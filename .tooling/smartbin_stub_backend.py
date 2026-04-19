from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
import asyncio
import random

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel


UTC = timezone.utc
RNG = random.Random(7)
NOW = datetime.now(UTC)


class BinDto(BaseModel):
    bin_id: str
    bin_name: str
    latitude: float
    longitude: float
    locality: str
    status: str
    last_seen_at: str | None = None
    installed_at: str | None = None
    last_waste_type: str | None = None
    total_events_today: int = 0


class WasteEventDto(BaseModel):
    event_id: str | None = None
    bin_id: str
    predicted_class: str
    confidence: float
    event_time: str
    uploaded_at: str | None = None
    source_device_id: str | None = None
    model_version: str | None = None


class CreateWasteEventRequest(BaseModel):
    bin_id: str
    predicted_class: str
    confidence: float
    event_time: str
    source_device_id: str
    model_version: str


RAW_CLASSES = [
    "battery",
    "clothes",
    "ewaste",
    "glass",
    "metal",
    "organic",
    "paper",
    "plastic",
    "shoes",
    "trash",
]


BIN_FIXTURES: list[dict[str, Any]] = [
    {
        "bin_id": "BIN-001",
        "bin_name": "Campus Gate Alpha",
        "latitude": 28.6139,
        "longitude": 77.2090,
        "locality": "Central Plaza",
        "status": "online",
        "installed_at": "2026-01-05T08:00:00Z",
    },
    {
        "bin_id": "BIN-002",
        "bin_name": "Library Walk",
        "latitude": 28.6152,
        "longitude": 77.2108,
        "locality": "Academic Block",
        "status": "online",
        "installed_at": "2026-01-08T08:00:00Z",
    },
    {
        "bin_id": "BIN-003",
        "bin_name": "Food Court South",
        "latitude": 28.6118,
        "longitude": 77.2069,
        "locality": "Food Court",
        "status": "degraded",
        "installed_at": "2026-01-11T08:00:00Z",
    },
    {
        "bin_id": "BIN-004",
        "bin_name": "Hostel Entrance",
        "latitude": 28.6174,
        "longitude": 77.2144,
        "locality": "Residential Zone",
        "status": "online",
        "installed_at": "2026-01-15T08:00:00Z",
    },
]


def _seed_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    base_counts = {
        "BIN-001": 31,
        "BIN-002": 29,
        "BIN-003": 42,
        "BIN-004": 26,
    }
    event_index = 0
    for bin_id, count in base_counts.items():
        for offset in range(count):
            timestamp = NOW - timedelta(days=6, hours=offset % 12, minutes=offset * 7 % 60)
            events.append(
                {
                    "event_id": f"EVT-{event_index:05d}",
                    "bin_id": bin_id,
                    "predicted_class": RAW_CLASSES[(offset + len(bin_id)) % len(RAW_CLASSES)],
                    "confidence": round(0.62 + ((offset % 30) / 100), 2),
                    "event_time": timestamp.isoformat().replace("+00:00", "Z"),
                    "uploaded_at": timestamp.isoformat().replace("+00:00", "Z"),
                    "source_device_id": "stub-backend",
                    "model_version": "stub-v1",
                },
            )
            event_index += 1
    return sorted(events, key=lambda item: item["event_time"])


EVENTS: list[dict[str, Any]] = _seed_events()
WS_CLIENTS: set[WebSocket] = set()

app = FastAPI(title="SmartBin Stub Backend")


def _today_event_count(bin_id: str) -> int:
    today = NOW.date()
    count = 0
    for event in EVENTS:
        if event["bin_id"] != bin_id:
            continue
        if datetime.fromisoformat(event["event_time"].replace("Z", "+00:00")).date() == today:
            count += 1
    return count


def _last_event(bin_id: str) -> dict[str, Any] | None:
    for event in reversed(EVENTS):
        if event["bin_id"] == bin_id:
            return event
    return None


def _materialize_bins() -> list[BinDto]:
    result: list[BinDto] = []
    for fixture in BIN_FIXTURES:
        last_event = _last_event(fixture["bin_id"])
        result.append(
            BinDto(
                **fixture,
                last_seen_at=last_event["event_time"] if last_event else None,
                last_waste_type=last_event["predicted_class"] if last_event else None,
                total_events_today=_today_event_count(fixture["bin_id"]),
            ),
        )
    return result


@app.get("/bins")
async def get_bins() -> list[dict[str, Any]]:
    return [item.model_dump() for item in _materialize_bins()]


@app.get("/bins/{bin_id}")
async def get_bin(bin_id: str) -> dict[str, Any]:
    for item in _materialize_bins():
        if item.bin_id == bin_id:
            return item.model_dump()
    raise RuntimeError(f"Unknown bin {bin_id}")


@app.get("/events")
async def get_events(
    bin_ids: str | None = Query(default=None),
    localities: str | None = Query(default=None),
    start: str = Query(...),
    end: str = Query(...),
) -> list[dict[str, Any]]:
    matched_bin_ids: set[str] = set()
    if localities:
        locality_set = {item.strip() for item in localities.split(",") if item.strip()}
        matched_bin_ids = {
            fixture["bin_id"]
            for fixture in BIN_FIXTURES
            if fixture["locality"] in locality_set
        }
    explicit_bin_ids = {item.strip() for item in (bin_ids or "").split(",") if item.strip()}
    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    results: list[dict[str, Any]] = []
    for event in EVENTS:
        event_dt = datetime.fromisoformat(event["event_time"].replace("Z", "+00:00"))
        if not (start_dt <= event_dt <= end_dt):
            continue
        if explicit_bin_ids and event["bin_id"] not in explicit_bin_ids:
            continue
        if matched_bin_ids and event["bin_id"] not in matched_bin_ids:
            continue
        results.append(event)
    return results


@app.post("/events")
async def post_event(request: CreateWasteEventRequest) -> dict[str, Any]:
    event = WasteEventDto(
        event_id=f"EVT-{len(EVENTS):05d}",
        bin_id=request.bin_id,
        predicted_class=request.predicted_class,
        confidence=request.confidence,
        event_time=request.event_time,
        uploaded_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        source_device_id=request.source_device_id,
        model_version=request.model_version,
    ).model_dump()
    EVENTS.append(event)
    await _broadcast_event(event)
    return event


@app.websocket("/events/stream")
async def events_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    WS_CLIENTS.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        WS_CLIENTS.discard(websocket)


async def _broadcast_event(event: dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    for client in WS_CLIENTS:
        try:
            await client.send_json(event)
        except Exception:
            stale.append(client)
    for client in stale:
        WS_CLIENTS.discard(client)


async def _background_events() -> None:
    while True:
        await asyncio.sleep(8)
        fixture = RNG.choice(BIN_FIXTURES)
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        event = WasteEventDto(
            event_id=f"EVT-{len(EVENTS):05d}",
            bin_id=fixture["bin_id"],
            predicted_class=RNG.choice(RAW_CLASSES),
            confidence=round(RNG.uniform(0.72, 0.96), 2),
            event_time=timestamp,
            uploaded_at=timestamp,
            source_device_id="stub-background",
            model_version="stub-v1",
        ).model_dump()
        EVENTS.append(event)
        await _broadcast_event(event)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_background_events())
