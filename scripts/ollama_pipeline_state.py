#!/usr/bin/env python3
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS images (
                raw_path TEXT PRIMARY KEY,
                filtered_path TEXT,
                category TEXT NOT NULL,
                item TEXT NOT NULL,
                query TEXT NOT NULL,
                source_engine TEXT,
                source_url TEXT,
                source_domain TEXT,
                downloaded_at TEXT,
                sha256 TEXT,
                phash TEXT,
                width INTEGER,
                height INTEGER,
                exact_dedupe_outcome TEXT,
                exact_duplicate_of TEXT,
                phash_dedupe_outcome TEXT,
                phash_duplicate_of TEXT,
                prefilter_decision TEXT,
                prefilter_score REAL,
                prefilter_reason TEXT,
                prefilter_details_json TEXT,
                class_decision TEXT,
                class_conf REAL,
                class_reason TEXT,
                photo_decision TEXT,
                photo_conf REAL,
                photo_reason TEXT,
                is_real_photo INTEGER,
                target_dominant INTEGER,
                has_humans INTEGER,
                has_major_clutter INTEGER,
                has_multiple_salient_objects INTEGER,
                is_infographic_or_render INTEGER,
                is_abnormal_artistic_case INTEGER,
                is_visually_clean INTEGER,
                is_trainworthy INTEGER,
                photo_details_json TEXT,
                final_decision TEXT,
                integration_status TEXT,
                dataset_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256);
            CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
            CREATE INDEX IF NOT EXISTS idx_images_final_decision ON images(final_decision);
            CREATE INDEX IF NOT EXISTS idx_images_prefilter_decision ON images(prefilter_decision);

            CREATE TABLE IF NOT EXISTS download_jobs (
                job_key TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                item TEXT NOT NULL,
                query TEXT NOT NULL,
                status TEXT NOT NULL,
                kept_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS domain_health (
                source_engine TEXT NOT NULL,
                source_domain TEXT NOT NULL,
                download_attempts INTEGER NOT NULL DEFAULT 0,
                download_successes INTEGER NOT NULL DEFAULT 0,
                download_failures INTEGER NOT NULL DEFAULT 0,
                low_res_rejections INTEGER NOT NULL DEFAULT 0,
                exact_duplicates INTEGER NOT NULL DEFAULT 0,
                phash_duplicates INTEGER NOT NULL DEFAULT 0,
                prefilter_rejections INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 0,
                rejected INTEGER NOT NULL DEFAULT 0,
                uncertain INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_engine, source_domain)
            );

            CREATE TABLE IF NOT EXISTS model_health (
                model_name TEXT NOT NULL,
                stage_name TEXT NOT NULL,
                calls INTEGER NOT NULL DEFAULT 0,
                successes INTEGER NOT NULL DEFAULT 0,
                failures INTEGER NOT NULL DEFAULT 0,
                total_latency_ms REAL NOT NULL DEFAULT 0,
                last_error TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (model_name, stage_name)
            );
            """
        )
        existing_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(images)").fetchall()
        }
        for column_name, column_type in (
            ("is_real_photo", "INTEGER"),
            ("target_dominant", "INTEGER"),
            ("has_humans", "INTEGER"),
            ("has_major_clutter", "INTEGER"),
            ("has_multiple_salient_objects", "INTEGER"),
            ("is_infographic_or_render", "INTEGER"),
            ("is_abnormal_artistic_case", "INTEGER"),
            ("is_visually_clean", "INTEGER"),
            ("is_trainworthy", "INTEGER"),
        ):
            if column_name not in existing_columns:
                conn.execute(f"ALTER TABLE images ADD COLUMN {column_name} {column_type}")


def upsert_image(db_path: Path, values: dict[str, Any]) -> None:
    payload = dict(values)
    now = utc_now()
    raw_path = payload.get("raw_path")
    if raw_path is None:
        raise ValueError("upsert_image requires raw_path")
    payload.setdefault("created_at", now)
    payload["updated_at"] = now
    with connect(db_path) as conn:
        existing = conn.execute("SELECT * FROM images WHERE raw_path = ?", (raw_path,)).fetchone()
        if existing is not None:
            for key in existing.keys():
                if key not in payload:
                    payload[key] = existing[key]
        columns = sorted(payload.keys())
        update_columns = [col for col in columns if col != "raw_path"]
        placeholders = ", ".join(f":{col}" for col in columns)
        conn.execute(
            f"""
            INSERT INTO images ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(raw_path) DO UPDATE SET
            {", ".join(f"{col}=excluded.{col}" for col in update_columns)}
            """,
            payload,
        )


def image_row(db_path: Path, raw_path: str) -> sqlite3.Row | None:
    with connect(db_path) as conn:
        return conn.execute("SELECT * FROM images WHERE raw_path = ?", (raw_path,)).fetchone()


def image_by_sha256(db_path: Path, sha256: str) -> sqlite3.Row | None:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM images WHERE sha256 = ? ORDER BY updated_at ASC LIMIT 1",
            (sha256,),
        ).fetchone()


def candidate_phash_rows(db_path: Path, category: str, phash: str) -> list[sqlite3.Row]:
    prefix = phash[:4]
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE category = ?
              AND phash IS NOT NULL
              AND phash != ''
              AND substr(phash, 1, 4) = ?
            ORDER BY updated_at ASC
            """,
            (category, prefix),
        ).fetchall()


def pending_prefilter_rows(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE exact_dedupe_outcome = 'unique'
              AND phash_dedupe_outcome = 'unique'
              AND prefilter_decision IS NULL
            ORDER BY category, item, raw_path
            """
        ).fetchall()


def pending_vlm_rows(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE exact_dedupe_outcome = 'unique'
              AND phash_dedupe_outcome = 'unique'
              AND prefilter_decision = 'accepted'
              AND final_decision IS NULL
            ORDER BY category, item, filtered_path, raw_path
            """
        ).fetchall()


def pending_integration_rows(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT * FROM images
            WHERE final_decision = 'accepted'
              AND (integration_status IS NULL OR integration_status = 'pending')
            ORDER BY category, item, filtered_path, raw_path
            """
        ).fetchall()


def all_domain_health_rows(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM domain_health ORDER BY source_engine, source_domain"
        ).fetchall()


def all_model_health_rows(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM model_health ORDER BY model_name, stage_name"
        ).fetchall()


def all_download_job_rows(db_path: Path) -> list[sqlite3.Row]:
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM download_jobs ORDER BY category, item"
        ).fetchall()


def mark_download_job(db_path: Path, category: str, item: str, query: str, status: str, kept_count: int = 0, last_error: str | None = None) -> None:
    now = utc_now()
    job_key = f"{category}::{item}"
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO download_jobs (job_key, category, item, query, status, kept_count, last_error, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET
                status=excluded.status,
                kept_count=excluded.kept_count,
                last_error=excluded.last_error,
                updated_at=excluded.updated_at
            """,
            (job_key, category, item, query, status, kept_count, last_error, now),
        )


def download_job_status(db_path: Path, category: str, item: str) -> str | None:
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM download_jobs WHERE job_key = ?",
            (f"{category}::{item}",),
        ).fetchone()
        return None if row is None else str(row["status"])


def bump_domain_health(
    db_path: Path,
    source_engine: str,
    source_domain: str,
    *,
    download_attempts: int = 0,
    download_successes: int = 0,
    download_failures: int = 0,
    low_res_rejections: int = 0,
    exact_duplicates: int = 0,
    phash_duplicates: int = 0,
    prefilter_rejections: int = 0,
    accepted: int = 0,
    rejected: int = 0,
    uncertain: int = 0,
    last_error: str | None = None,
) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO domain_health (
                source_engine, source_domain, download_attempts, download_successes, download_failures,
                low_res_rejections, exact_duplicates, phash_duplicates, prefilter_rejections,
                accepted, rejected, uncertain, last_error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_engine, source_domain) DO UPDATE SET
                download_attempts = domain_health.download_attempts + excluded.download_attempts,
                download_successes = domain_health.download_successes + excluded.download_successes,
                download_failures = domain_health.download_failures + excluded.download_failures,
                low_res_rejections = domain_health.low_res_rejections + excluded.low_res_rejections,
                exact_duplicates = domain_health.exact_duplicates + excluded.exact_duplicates,
                phash_duplicates = domain_health.phash_duplicates + excluded.phash_duplicates,
                prefilter_rejections = domain_health.prefilter_rejections + excluded.prefilter_rejections,
                accepted = domain_health.accepted + excluded.accepted,
                rejected = domain_health.rejected + excluded.rejected,
                uncertain = domain_health.uncertain + excluded.uncertain,
                last_error = COALESCE(excluded.last_error, domain_health.last_error),
                updated_at = excluded.updated_at
            """,
            (
                source_engine,
                source_domain,
                download_attempts,
                download_successes,
                download_failures,
                low_res_rejections,
                exact_duplicates,
                phash_duplicates,
                prefilter_rejections,
                accepted,
                rejected,
                uncertain,
                last_error,
                now,
            ),
        )


def bump_model_health(
    db_path: Path,
    model_name: str,
    stage_name: str,
    *,
    success: bool,
    latency_ms: float,
    error: str | None = None,
) -> None:
    now = utc_now()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_health (
                model_name, stage_name, calls, successes, failures, total_latency_ms, last_error, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name, stage_name) DO UPDATE SET
                calls = model_health.calls + 1,
                successes = model_health.successes + excluded.successes,
                failures = model_health.failures + excluded.failures,
                total_latency_ms = model_health.total_latency_ms + excluded.total_latency_ms,
                last_error = COALESCE(excluded.last_error, model_health.last_error),
                updated_at = excluded.updated_at
            """,
            (
                model_name,
                stage_name,
                1,
                1 if success else 0,
                0 if success else 1,
                float(latency_ms),
                error,
                now,
            ),
        )
