#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlsplit
from urllib.request import Request, urlopen

from PIL import Image

try:
    from dataset_utils import build_existing_hash_index, hash_file, load_metadata, save_metadata
    from eval_splits_no_aug import build_splits
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_utils import build_existing_hash_index, hash_file, load_metadata, save_metadata
    from eval_splits_no_aug import build_splits

from ollama_pipeline_defaults import (
    CATEGORY_THRESHOLD_PROFILES,
    CLASS_STAGE_PROMPT,
    DEFAULT_CLASS_SPEC,
    DEFAULT_DOWNLOADER_PROFILE,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_TEXT_GENERATION_PROFILE,
    DEFAULT_TEXT_MODEL,
    DEFAULT_VISION_GENERATION_PROFILE,
    DEFAULT_VISION_MODELS,
    NOISY_SUBCLASSES,
    PHOTO_STAGE_PROMPT,
    TEXT_CLASS_DISCOVERY_SYSTEM_PROMPT,
    TEXT_CLASS_DISCOVERY_USER_PROMPT,
    ThresholdProfile,
    VLM_SYSTEM_PROMPT,
)
from ollama_pipeline_state import (
    all_domain_health_rows,
    all_download_job_rows,
    all_model_health_rows,
    bump_domain_health,
    bump_model_health,
    candidate_phash_rows,
    connect,
    download_job_status,
    image_by_sha256,
    image_row,
    init_db,
    mark_download_job,
    pending_integration_rows,
    pending_prefilter_rows,
    pending_yolo_rows,
    pending_vlm_rows,
    upsert_image,
)

try:
    import imagehash
except Exception:
    imagehash = None

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
except Exception:
    torch = None
    CLIPModel = None
    CLIPProcessor = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from icrawler.builtin import BaiduImageCrawler, BingImageCrawler, GoogleImageCrawler
    from icrawler.downloader import ImageDownloader
    import icrawler.parser as icrawler_parser
except Exception:
    BaiduImageCrawler = None
    BingImageCrawler = None
    GoogleImageCrawler = None
    ImageDownloader = None
    icrawler_parser = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
YOLO_MODEL_ID = "yolov8n.pt"
PHASH_DISTANCE_THRESHOLD = 4
CLIP_PREFILTER_PROMPTS = {
    "dominant_item": "a clear real-world training photo where the target {category} {item} object is the dominant subject",
    "dominant_category": "a clean close-up training photo of a single {category} object",
    "has_human": "a photo dominated by a person or a human hand",
    "major_clutter": "a cluttered scene with many unrelated objects",
    "multiple_salient": "a busy scene with multiple equally important objects",
}
_CLIP_PREFILTER: dict[str, Any] = {"model": None, "processor": None, "device": None}
_YOLO_PREFILTER: dict[str, Any] = {"model": None, "device": None}


def patch_icrawler_parser_worker() -> None:
    if icrawler_parser is None:
        return
    if getattr(icrawler_parser.Parser.worker_exec, "_esd_patched", False):
        return

    def worker_exec(self, queue_timeout=2, req_timeout=5, max_retry=3, **kwargs):
        while True:
            if self.signal.get("reach_max_num"):
                break
            try:
                url = self.in_queue.get(timeout=queue_timeout)
            except icrawler_parser.queue.Empty:
                if self.signal.get("feeder_exited"):
                    break
                continue
            except Exception:
                continue

            retry = max_retry
            while retry > 0:
                try:
                    base_url = "{0.scheme}://{0.netloc}".format(icrawler_parser.urlsplit(url))
                    response = self.session.get(url, timeout=req_timeout, headers={"Referer": base_url})
                except Exception as exc:
                    self.logger.error("fetch failed for %s: %s", url, exc)
                else:
                    try:
                        parsed_tasks = self.parse(response, **kwargs) or []
                    except Exception as exc:
                        self.logger.error("parse failed for %s: %s", url, exc)
                        parsed_tasks = []
                    for task in parsed_tasks:
                        while not self.signal.get("reach_max_num"):
                            try:
                                if isinstance(task, dict):
                                    self.output(task, timeout=1)
                                elif isinstance(task, str):
                                    self.input(task, timeout=1)
                            except icrawler_parser.queue.Full:
                                time.sleep(1)
                            except Exception as exc:
                                self.logger.error("queue put failed for %s: %s", task, exc)
                            else:
                                break
                        if self.signal.get("reach_max_num"):
                            break
                    self.in_queue.task_done()
                    break
                finally:
                    retry -= 1

    worker_exec._esd_patched = True
    icrawler_parser.Parser.worker_exec = worker_exec
    logging.getLogger("parser").setLevel(logging.ERROR)
    logging.getLogger("downloader").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama-first end-to-end dataset curation and training pipeline.")
    parser.add_argument("--class-spec", default="", help="JSON string or JSON file path describing target classes and optional seed objects. Defaults to organic/metal/paper.")
    parser.add_argument("--output-root", default="review_downloads/ollama_end_to_end_pipeline")
    parser.add_argument("--dataset-root", default="Dataset_Final")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODELS[0])
    parser.add_argument("--vision-models", nargs="+", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--text-temperature", type=float, default=DEFAULT_TEXT_GENERATION_PROFILE.temperature)
    parser.add_argument("--vision-temperature", type=float, default=DEFAULT_VISION_GENERATION_PROFILE.temperature)
    parser.add_argument("--item-workers", type=int, default=DEFAULT_DOWNLOADER_PROFILE.item_workers)
    parser.add_argument("--engine-workers", type=int, default=DEFAULT_DOWNLOADER_PROFILE.engine_workers)
    parser.add_argument("--direct-bing-limit", type=int, default=DEFAULT_DOWNLOADER_PROFILE.direct_bing_limit)
    parser.add_argument("--google-limit", type=int, default=DEFAULT_DOWNLOADER_PROFILE.google_limit)
    parser.add_argument("--bing-limit", type=int, default=DEFAULT_DOWNLOADER_PROFILE.bing_limit)
    parser.add_argument("--baidu-limit", type=int, default=DEFAULT_DOWNLOADER_PROFILE.baidu_limit)
    parser.add_argument("--min-resolution", type=int, default=DEFAULT_DOWNLOADER_PROFILE.min_resolution)
    parser.add_argument("--yolo-model", default=YOLO_MODEL_ID)
    parser.add_argument("--integrate-accepted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-split-manifest", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata-file", default="dataset_metadata.json")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-ratios", default="0.9,0.05,0.05")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--training-command", default="")
    parser.add_argument("--pull-missing-models", action="store_true")
    parser.add_argument("--force", action="store_true", default=False)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging(logs_root: Path) -> None:
    ensure_dir(logs_root)
    logfile = logs_root / "pipeline.log"
    handlers = [
        logging.FileHandler(logfile, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def artifact_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "raw": output_root / "raw",
        "filtered": output_root / "filtered",
        "accepted": output_root / "accepted",
        "rejected": output_root / "rejected",
        "uncertain": output_root / "uncertain",
        "integrated": output_root / "integrated",
        "manifests": output_root / "manifests",
        "logs": output_root / "logs",
        "class_discovery": output_root / "manifests" / "class_discovery",
    }
    for path in dirs.values():
        ensure_dir(path)
    return dirs


def load_json_file_or_inline(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if not stripped:
        return {"classes": DEFAULT_CLASS_SPEC}
    if stripped[0] in "{[":
        return json.loads(stripped)
    candidate = Path(stripped)
    if len(stripped) < 512 and candidate.exists():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(stripped)


def normalize_class_spec(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    source = spec["classes"] if "classes" in spec and isinstance(spec["classes"], dict) else spec
    normalized: dict[str, dict[str, Any]] = {}
    for class_name, payload in source.items():
        if isinstance(payload, list):
            normalized[class_name] = {"seed_objects": list(payload), "description": ""}
        elif isinstance(payload, dict):
            normalized[class_name] = {
                "seed_objects": list(payload.get("seed_objects", payload.get("objects", []))),
                "description": str(payload.get("description", "")).strip(),
            }
        else:
            raise ValueError("Class spec entries must be either a list of seed objects or an object payload.")
    return normalized


def parse_split_ratios(spec: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in spec.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--split-ratios must contain exactly three comma-separated values.")
    total = sum(parts)
    if total <= 0.0:
        raise ValueError("--split-ratios must sum to a positive value.")
    return (parts[0] / total, parts[1] / total, parts[2] / total)


def ollama_headers() -> dict[str, str]:
    return {"Content-Type": "application/json"}


def ollama_chat(
    host: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    stream: bool = False,
    keep_alive: str = "30m",
    format_json: bool = False,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": stream,
        "keep_alive": keep_alive,
        "options": {
            "temperature": temperature,
        },
    }
    if format_json:
        payload["format"] = "json"
    last_error: Exception | None = None
    for attempt in range(3):
        request = Request(
            url=f"{host.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers=ollama_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=900) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, ValueError) as exc:
            last_error = exc
            if attempt == 2:
                break
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Ollama chat failed for model {model}: {last_error}")


def ollama_tags(host: str) -> set[str]:
    request = Request(url=f"{host.rstrip('/')}/api/tags", headers=ollama_headers(), method="GET")
    with urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {str(model.get("name", "")) for model in payload.get("models", [])}


def ensure_ollama_models(host: str, models: list[str], pull_missing: bool) -> None:
    logging.info("Checking Ollama models: %s", ", ".join(models))
    available = ollama_tags(host)
    missing = [model for model in models if model not in available]
    if not missing:
        logging.info("All requested Ollama models are already available.")
        return
    if not pull_missing:
        raise RuntimeError(f"Missing Ollama models: {missing}. Re-run with --pull-missing-models to fetch them.")
    for model in missing:
        logging.info("Pulling missing Ollama model: %s", model)
        subprocess.run(["ollama", "pull", model], check=True, cwd=str(repo_root()))
        logging.info("Finished pulling Ollama model: %s", model)


def parse_json_response(content: str) -> dict[str, Any]:
    content = content.strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        content = match.group(0)
    return json.loads(content)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def source_domain_from_url(url: str | None) -> str:
    if not url:
        return "unknown"
    try:
        return re.sub(r"^www\.", "", urlsplit(url).netloc) or "unknown"
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_image_phash(path: Path) -> str | None:
    if imagehash is None:
        return None
    try:
        with Image.open(path) as image:
            return str(imagehash.phash(image))
    except Exception:
        return None


def phash_distance(a: str | None, b: str | None) -> int | None:
    if imagehash is None or not a or not b:
        return None
    try:
        return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
    except Exception:
        return None


def save_image_from_bytes(data: bytes, output_dir: Path, prefix: str, index: int, min_resolution: int) -> tuple[Path | None, tuple[int, int] | None]:
    try:
        with Image.open(BytesIO(data)) as image:
            image.load()
            if image.size[0] < min_resolution or image.size[1] < min_resolution:
                return None, image.size
            fmt = (image.format or "JPEG").lower()
            if fmt == "jpeg":
                fmt = "jpg"
            ext = f".{fmt}" if fmt in {"jpg", "png", "webp", "bmp"} else ".jpg"
            path = output_dir / f"{prefix}_{index:06d}{ext}"
            image.convert("RGB").save(path)
            return path, image.size
    except Exception:
        return None, None


def bing_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.11 (KHTML, like Gecko) "
            "Chrome/23.0.1271.64 Safari/537.11"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
        "Accept-Encoding": "identity",
        "Accept-Language": "en-US,en;q=0.8",
        "Connection": "keep-alive",
    }


def fetch_bing_links(query: str, limit: int) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    page_size = 150
    pages = max(1, (limit + page_size - 1) // page_size)
    for page_idx in range(pages):
        request_url = (
            "https://www.bing.com/images/async?q="
            + quote_plus(query)
            + f"&first={page_idx * page_size}&count={page_size}&adlt=off&qft="
        )
        try:
            html = urlopen(Request(request_url, None, headers=bing_headers()), timeout=30).read().decode("utf8", errors="ignore")
        except Exception:
            continue
        page_links = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
        for link in page_links:
            if link in seen:
                continue
            seen.add(link)
            links.append(link)
            if len(links) >= limit:
                return links
    return links


def download_direct_bing(query: str, output_dir: Path, limit: int, min_resolution: int, db_path: Path) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    logging.info("[download/direct_bing] start query=%s limit=%d output=%s", query, limit, output_dir)
    links = fetch_bing_links(query, limit)
    logging.info("[download/direct_bing] fetched_links=%d query=%s", len(links), query)
    saved: list[dict[str, Any]] = []
    for index, link in enumerate(links, start=1):
        domain = source_domain_from_url(link)
        bump_domain_health(db_path, "direct_bing", domain, download_attempts=1)
        logging.info("[download/direct_bing] fetch %d/%d domain=%s", index, len(links), domain)
        try:
            data = urlopen(Request(link, None, headers=bing_headers()), timeout=30).read()
        except (HTTPError, URLError, OSError, ValueError, Exception):
            bump_domain_health(db_path, "direct_bing", domain, download_failures=1, last_error="download_failed")
            logging.info("[download/direct_bing] failed domain=%s url=%s", domain, link)
            continue
        path, size = save_image_from_bytes(data, output_dir, "direct_bing", index, min_resolution)
        if path is None:
            if size is not None:
                bump_domain_health(db_path, "direct_bing", domain, low_res_rejections=1)
                logging.info("[download/direct_bing] low_res domain=%s size=%s url=%s", domain, size, link)
            continue
        bump_domain_health(db_path, "direct_bing", domain, download_successes=1)
        logging.info("[download/direct_bing] saved path=%s size=%sx%s", path, size[0] if size else "?", size[1] if size else "?")
        saved.append(
            {
                "raw_path": str(path),
                "source_engine": "direct_bing",
                "source_url": link,
                "source_domain": domain,
                "downloaded_at": utc_now(),
                "width": size[0] if size else None,
                "height": size[1] if size else None,
            }
        )
    return saved


def download_icrawler_engine(engine_name: str, query: str, output_dir: Path, limit: int, db_path: Path) -> list[dict[str, Any]]:
    patch_icrawler_parser_worker()
    if limit <= 0:
        return []
    if ImageDownloader is None:
        return []
    logging.info("[download/%s] start query=%s limit=%d output=%s", engine_name, query, limit, output_dir)
    records: list[dict[str, Any]] = []

    class ProvenanceImageDownloader(ImageDownloader):
        def process_meta(self, task):  # type: ignore[override]
            if task.get("file_url"):
                domain = source_domain_from_url(str(task.get("file_url", "")))
                bump_domain_health(db_path, engine_name, domain, download_attempts=1)
            result = super().process_meta(task)
            if task.get("success") and task.get("filename"):
                file_url = str(task.get("file_url", ""))
                domain = source_domain_from_url(file_url)
                records.append(
                    {
                        "raw_path": str(output_dir / str(task["filename"])),
                        "source_engine": engine_name,
                        "source_url": file_url,
                        "source_domain": domain,
                        "downloaded_at": utc_now(),
                    }
                )
                bump_domain_health(db_path, engine_name, domain, download_successes=1)
            elif task.get("file_url"):
                bump_domain_health(
                    db_path,
                    engine_name,
                    source_domain_from_url(str(task.get("file_url", ""))),
                    download_failures=1,
                    last_error="icrawler_failed",
                )
            return result

    if engine_name == "google" and GoogleImageCrawler is not None:
        logging.info("[download/google] crawler starting")
        crawler = GoogleImageCrawler(
            downloader_cls=ProvenanceImageDownloader,
            storage={"root_dir": str(output_dir)},
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
        )
        crawler.crawl(keyword=query, max_num=limit, filters={"size": "large"})
    elif engine_name == "bing" and BingImageCrawler is not None:
        logging.info("[download/bing] crawler starting")
        crawler = BingImageCrawler(
            downloader_cls=ProvenanceImageDownloader,
            storage={"root_dir": str(output_dir)},
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
        )
        crawler.crawl(keyword=query, max_num=limit, filters={"size": "large"})
    elif engine_name == "baidu" and BaiduImageCrawler is not None:
        logging.info("[download/baidu] crawler starting")
        crawler = BaiduImageCrawler(
            downloader_cls=ProvenanceImageDownloader,
            storage={"root_dir": str(output_dir)},
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
        )
        crawler.crawl(keyword=query, max_num=limit)
    logging.info("[download/%s] crawler finished records=%d", engine_name, len(records))
    return records


def load_clip_prefilter() -> tuple[Any, Any, str] | tuple[None, None, str]:
    if CLIPModel is None or CLIPProcessor is None or torch is None:
        logging.info("CLIP prefilter unavailable; prefilter stage will default to accepted.")
        return None, None, "cpu"
    if _CLIP_PREFILTER["model"] is None or _CLIP_PREFILTER["processor"] is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info("Loading CLIP prefilter model=%s device=%s", CLIP_MODEL_ID, device)
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
        model.eval()
        model.to(device)
        _CLIP_PREFILTER["model"] = model
        _CLIP_PREFILTER["processor"] = processor
        _CLIP_PREFILTER["device"] = device
        logging.info("CLIP prefilter loaded successfully.")
    return _CLIP_PREFILTER["model"], _CLIP_PREFILTER["processor"], _CLIP_PREFILTER["device"]


def load_yolo_prefilter(model_id: str) -> tuple[Any, str] | tuple[None, str]:
    if YOLO is None:
        logging.info("YOLO prefilter unavailable; stage will default to accepted.")
        return None, "cpu"
    if _YOLO_PREFILTER["model"] is None:
        device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
        logging.info("Loading YOLO prefilter model=%s device=%s", model_id, device)
        model = YOLO(model_id)
        _YOLO_PREFILTER["model"] = model
        _YOLO_PREFILTER["device"] = device
        logging.info("YOLO prefilter loaded successfully.")
    return _YOLO_PREFILTER["model"], _YOLO_PREFILTER["device"]


def encode_image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def threshold_profile_for(category: str, item: str) -> tuple[float, float, float, float]:
    profile = CATEGORY_THRESHOLD_PROFILES.get(
        category,
        ThresholdProfile(class_accept=0.80, class_reject=0.45, photo_accept=0.82, photo_reject=0.48, noisy_bonus=0.0),
    )
    noisy_bonus = profile.noisy_bonus if item in NOISY_SUBCLASSES else 0.0
    return (
        min(1.0, profile.class_accept + noisy_bonus),
        profile.class_reject,
        min(1.0, profile.photo_accept + noisy_bonus),
        profile.photo_reject,
    )


def class_stage_prompt_for(category: str, item: str, classes: list[str]) -> str:
    allowed_classes_csv = ", ".join(classes)
    if category in CATEGORY_THRESHOLD_PROFILES:
        return CLASS_STAGE_PROMPT.format(
            category=category,
            item=item,
            allowed_classes=classes,
            allowed_classes_csv=allowed_classes_csv,
        )
    return (
        f"Target superclass: {category}\n"
        f"Target object subclass: {item}\n"
        f"Allowed classes: {allowed_classes_csv}\n\n"
        "Task:\n"
        "- Decide if this image is best described as the target superclass or not.\n"
        "- Keep only real camera photos of physical objects that genuinely belong to the target superclass.\n"
        "- Reject screenshots, vector graphics, infographics, diagrams, clipart, logos, CG renders, AI-generated images, or clearly synthetic compositions.\n\n"
        "Return JSON:\n"
        "{\n"
        f'  "decision": "<one of: {allowed_classes_csv}, reject, uncertain>",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "short reason"\n'
        "}\n"
    )


def photo_stage_prompt_for(category: str, item: str) -> str:
    if category in CATEGORY_THRESHOLD_PROFILES:
        return PHOTO_STAGE_PROMPT.format(category=category, item=item)
    return (
        f"Target superclass: {category}\n"
        f"Target object subclass: {item}\n\n"
        "Return strict JSON with the fields: decision, confidence, reason, is_real_photo, target_dominant, has_humans, "
        "has_major_clutter, has_multiple_salient_objects, is_infographic_or_render, is_abnormal_artistic_case, "
        "is_visually_clean, is_trainworthy.\n"
    )


def ask_vlm_stage(
    host: str,
    model: str,
    image_path: Path,
    prompt: str,
    temperature: float,
) -> dict[str, Any]:
    response = ollama_chat(
        host,
        model,
        messages=[
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt, "images": [encode_image_base64(image_path)]},
        ],
        temperature=temperature,
        keep_alive="30m",
        format_json=True,
    )
    content = response.get("message", {}).get("content", "{}")
    parsed = parse_json_response(content)
    parsed.setdefault("decision", "uncertain")
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("reason", "")
    parsed["confidence"] = safe_float(parsed.get("confidence"), 0.0)
    return parsed


def timed_vlm_stage(
    host: str,
    model: str,
    image_path: Path,
    prompt: str,
    temperature: float,
    db_path: Path,
    stage_name: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    logging.info("[vlm/stage] start model=%s stage=%s image=%s", model, stage_name, image_path)
    try:
        parsed = ask_vlm_stage(host, model, image_path, prompt, temperature)
    except Exception as exc:
        bump_model_health(db_path, model, stage_name, success=False, latency_ms=(time.perf_counter() - started) * 1000.0, error=str(exc))
        logging.info("[vlm/stage] failed model=%s stage=%s image=%s error=%s", model, stage_name, image_path, exc)
        raise
    bump_model_health(db_path, model, stage_name, success=True, latency_ms=(time.perf_counter() - started) * 1000.0)
    logging.info("[vlm/stage] done model=%s stage=%s image=%s", model, stage_name, image_path)
    return parsed


def copy_decision_file(src: Path, decision_root: Path, category: str, item: str) -> Path:
    dst_dir = decision_root / category / slugify(item)
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


def iter_image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def discover_objects(
    host: str,
    text_model: str,
    class_spec: dict[str, dict[str, Any]],
    *,
    temperature: float,
    db_path: Path,
) -> dict[str, dict[str, Any]]:
    logging.info("[class_discovery] start classes=%s", ", ".join(class_spec.keys()))
    prompt = TEXT_CLASS_DISCOVERY_USER_PROMPT.format(class_spec=json.dumps(class_spec, indent=2, sort_keys=True))
    started = time.perf_counter()
    try:
        response = ollama_chat(
            host,
            text_model,
            messages=[
                {"role": "system", "content": TEXT_CLASS_DISCOVERY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            keep_alive="30m",
            format_json=True,
        )
    except Exception as exc:
        bump_model_health(db_path, text_model, "class_discovery", success=False, latency_ms=(time.perf_counter() - started) * 1000.0, error=str(exc))
        raise
    bump_model_health(db_path, text_model, "class_discovery", success=True, latency_ms=(time.perf_counter() - started) * 1000.0)
    content = response.get("message", {}).get("content", "{}")
    parsed = parse_json_response(content)
    discovered = normalize_class_spec(parsed)
    logging.info("[class_discovery] discovered=%s", json.dumps(discovered, sort_keys=True))
    merged: dict[str, dict[str, Any]] = {}
    for class_name, payload in class_spec.items():
        discovered_payload = discovered.get(class_name, {"seed_objects": [], "description": ""})
        merged_objects: list[str] = []
        for item in payload.get("seed_objects", []) + discovered_payload.get("seed_objects", []):
            normalized = str(item).strip()
            if normalized and normalized not in merged_objects:
                merged_objects.append(normalized)
        merged[class_name] = {
            "description": payload.get("description", discovered_payload.get("description", "")),
            "seed_objects": merged_objects,
        }
    return merged


def register_downloaded_image(
    db_path: Path,
    raw_path: Path,
    filtered_root: Path,
    *,
    category: str,
    item: str,
    query: str,
    source_engine: str,
    source_url: str | None,
    source_domain: str | None,
    downloaded_at: str | None,
    min_resolution: int,
) -> str:
    existing = image_row(db_path, str(raw_path))
    if existing and existing["sha256"] and existing["exact_dedupe_outcome"] and existing["phash_dedupe_outcome"]:
        if existing["filtered_path"] and Path(str(existing["filtered_path"])).exists():
            logging.info("[dedupe/register] already_processed raw=%s exact=%s phash=%s", raw_path, existing["exact_dedupe_outcome"], existing["phash_dedupe_outcome"])
            return str(existing["phash_dedupe_outcome"])
        if str(existing["exact_dedupe_outcome"]) == "unique" and str(existing["phash_dedupe_outcome"]) == "unique":
            filtered_path = filtered_root / category / slugify(item) / raw_path.name
            ensure_dir(filtered_path.parent)
            if raw_path.exists() and not filtered_path.exists():
                shutil.copy2(raw_path, filtered_path)
            upsert_image(db_path, {"raw_path": str(raw_path), "filtered_path": str(filtered_path)})
            logging.info("[dedupe/register] restored filtered copy raw=%s filtered=%s", raw_path, filtered_path)
        return str(existing["phash_dedupe_outcome"])

    if not raw_path.exists():
        logging.info("[dedupe/register] missing raw=%s", raw_path)
        return "missing"

    try:
        with Image.open(raw_path) as image:
            image.load()
            width, height = image.size
    except Exception:
        logging.info("[dedupe/register] invalid image raw=%s", raw_path)
        raw_path.unlink(missing_ok=True)
        return "invalid"

    if width < min_resolution or height < min_resolution:
        bump_domain_health(db_path, source_engine, source_domain or "unknown", low_res_rejections=1)
        logging.info("[dedupe/register] low_res raw=%s size=%dx%d", raw_path, width, height)
        raw_path.unlink(missing_ok=True)
        return "low_res"

    sha256 = sha256_file(raw_path)
    phash = compute_image_phash(raw_path)

    exact_outcome = "unique"
    exact_duplicate_of = None
    existing_exact = image_by_sha256(db_path, sha256)
    if existing_exact is not None and str(existing_exact["raw_path"]) != str(raw_path):
        exact_outcome = "duplicate"
        exact_duplicate_of = str(existing_exact["raw_path"])
        bump_domain_health(db_path, source_engine, source_domain or "unknown", exact_duplicates=1)
        logging.info("[dedupe/register] exact_duplicate raw=%s duplicate_of=%s", raw_path, exact_duplicate_of)

    phash_outcome = "unique"
    phash_duplicate_of = None
    if exact_outcome == "unique" and phash:
        best_row = None
        best_distance = None
        for candidate in candidate_phash_rows(db_path, category, phash):
            if str(candidate["raw_path"]) == str(raw_path):
                continue
            candidate_distance = phash_distance(phash, str(candidate["phash"]))
            if candidate_distance is None:
                continue
            if best_distance is None or candidate_distance < best_distance:
                best_distance = candidate_distance
                best_row = candidate
        if best_row is not None and best_distance is not None and best_distance <= PHASH_DISTANCE_THRESHOLD:
            phash_outcome = "duplicate"
            phash_duplicate_of = str(best_row["raw_path"])
            bump_domain_health(db_path, source_engine, source_domain or "unknown", phash_duplicates=1)
            logging.info("[dedupe/register] phash_duplicate raw=%s duplicate_of=%s distance=%s", raw_path, phash_duplicate_of, best_distance)

    filtered_path: str | None = None
    if exact_outcome == "unique" and phash_outcome == "unique":
        dst = filtered_root / category / slugify(item) / raw_path.name
        ensure_dir(dst.parent)
        if not dst.exists():
            shutil.copy2(raw_path, dst)
        filtered_path = str(dst)
        logging.info("[dedupe/register] unique raw=%s filtered=%s", raw_path, filtered_path)

    upsert_image(
        db_path,
        {
            "raw_path": str(raw_path),
            "filtered_path": filtered_path,
            "category": category,
            "item": item,
            "query": query,
            "source_engine": source_engine,
            "source_url": source_url,
            "source_domain": source_domain,
            "downloaded_at": downloaded_at or utc_now(),
            "sha256": sha256,
            "phash": phash,
            "width": width,
            "height": height,
            "exact_dedupe_outcome": exact_outcome,
            "exact_duplicate_of": exact_duplicate_of,
            "phash_dedupe_outcome": phash_outcome,
            "phash_duplicate_of": phash_duplicate_of,
            "integration_status": "pending",
        },
    )
    return phash_outcome if exact_outcome == "unique" else exact_outcome


def download_query_bundle(
    category: str,
    item: str,
    args: argparse.Namespace,
    raw_root: Path,
    filtered_root: Path,
    db_path: Path,
) -> tuple[str, str, int]:
    query = f"{category} {item}"
    target_dir = raw_root / category / slugify(item)
    ensure_dir(target_dir)
    logging.info("[download/job] start category=%s item=%s query=%s", category, item, query)

    job_state = download_job_status(db_path, category, item)
    if job_state == "complete":
        logging.info("[download/job] already_complete category=%s item=%s", category, item)
        kept = 0
        with connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM images
                WHERE category = ? AND item = ?
                  AND exact_dedupe_outcome = 'unique'
                  AND phash_dedupe_outcome = 'unique'
                """,
                (category, item),
            ).fetchone()
            kept = int(row["count"]) if row else 0
        return category, item, kept

    mark_download_job(db_path, category, item, query, "in_progress")
    logging.info("[download/job] marked_in_progress category=%s item=%s", category, item)
    records_by_path: dict[str, dict[str, Any]] = {}

    def run_engine(engine_name: str) -> None:
        logging.info("[download/job] engine_start category=%s item=%s engine=%s", category, item, engine_name)
        engine_records: list[dict[str, Any]] = []
        if engine_name == "direct_bing":
            engine_records = download_direct_bing(query, target_dir, args.direct_bing_limit, args.min_resolution, db_path)
        elif engine_name == "google":
            engine_records = download_icrawler_engine("google", query, target_dir, args.google_limit, db_path)
        elif engine_name == "bing":
            engine_records = download_icrawler_engine("bing", query, target_dir, args.bing_limit, db_path)
        elif engine_name == "baidu":
            engine_records = download_icrawler_engine("baidu", query, target_dir, args.baidu_limit, db_path)
        for record in engine_records:
            records_by_path[record["raw_path"]] = record
        logging.info("[download/job] engine_end category=%s item=%s engine=%s records=%d", category, item, engine_name, len(engine_records))

    engines = ["direct_bing", "google", "bing", "baidu"]
    try:
        with ThreadPoolExecutor(max_workers=args.engine_workers) as pool:
            futures = [pool.submit(run_engine, engine) for engine in engines]
            for future in as_completed(futures):
                future.result()

        kept = 0
        for path in iter_image_files(target_dir):
            logging.info("[download/job] register_candidate category=%s item=%s path=%s", category, item, path)
            record = records_by_path.get(
                str(path),
                {
                    "raw_path": str(path),
                    "source_engine": "resume_scan",
                    "source_url": None,
                    "source_domain": "unknown",
                    "downloaded_at": utc_now(),
                },
            )
            outcome = register_downloaded_image(
                db_path,
                path,
                filtered_root,
                category=category,
                item=item,
                query=query,
                source_engine=str(record.get("source_engine", "unknown")),
                source_url=record.get("source_url"),
                source_domain=str(record.get("source_domain", "unknown")),
                downloaded_at=str(record.get("downloaded_at", utc_now())),
                min_resolution=args.min_resolution,
            )
            if outcome == "unique":
                kept += 1
            logging.info("[download/job] candidate_done path=%s outcome=%s kept=%d", path, outcome, kept)
        mark_download_job(db_path, category, item, query, "complete", kept_count=kept)
        logging.info("[download/job] complete category=%s item=%s kept=%d", category, item, kept)
        return category, item, kept
    except Exception as exc:
        mark_download_job(db_path, category, item, query, "failed", last_error=str(exc))
        logging.exception("[download/job] failed category=%s item=%s", category, item)
        raise


def run_clip_prefilter(
    db_path: Path,
    filtered_root: Path,
    rejected_root: Path,
    min_resolution: int,
) -> dict[str, int]:
    rows = pending_prefilter_rows(db_path)
    counts = Counter()
    model, processor, device = load_clip_prefilter()
    logging.info("[prefilter] pending_rows=%d", len(rows))
    for row in rows:
        raw_path = Path(str(row["raw_path"]))
        filtered_path = Path(str(row["filtered_path"])) if row["filtered_path"] else filtered_root / str(row["category"]) / slugify(str(row["item"])) / raw_path.name
        ensure_dir(filtered_path.parent)
        if row["filtered_path"] and raw_path.exists() and not filtered_path.exists():
            shutil.copy2(raw_path, filtered_path)

        source_engine = str(row["source_engine"] or "unknown")
        source_domain = str(row["source_domain"] or "unknown")
        if model is None or processor is None or torch is None:
            upsert_image(
                db_path,
                {
                    "raw_path": str(raw_path),
                    "filtered_path": str(filtered_path) if filtered_path.exists() else None,
                    "prefilter_decision": "accepted",
                    "prefilter_score": 1.0,
                    "prefilter_reason": "clip_prefilter_unavailable",
                    "prefilter_details_json": json.dumps({"target_dominant": True}),
                },
            )
            counts["accepted"] += 1
            logging.info("[prefilter] accepted_without_clip raw=%s", raw_path)
            continue

        started = time.perf_counter()
        logging.info("[prefilter] start raw=%s category=%s item=%s", raw_path, row["category"], row["item"])
        try:
            with Image.open(filtered_path if filtered_path.exists() else raw_path) as image:
                image = image.convert("RGB")
                if image.size[0] < min_resolution or image.size[1] < min_resolution:
                    decision = "rejected"
                    score = 0.0
                    reason = "low_resolution_after_register"
                    details = {
                        "target_dominant": False,
                        "has_humans": False,
                        "has_major_clutter": True,
                        "has_multiple_salient_objects": True,
                        "clip_prompt_probs": {},
                    }
                else:
                    prompts = [
                        CLIP_PREFILTER_PROMPTS["dominant_item"].format(category=row["category"], item=row["item"]),
                        CLIP_PREFILTER_PROMPTS["dominant_category"].format(category=row["category"]),
                        CLIP_PREFILTER_PROMPTS["has_human"],
                        CLIP_PREFILTER_PROMPTS["major_clutter"],
                        CLIP_PREFILTER_PROMPTS["multiple_salient"],
                    ]
                    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.inference_mode():
                        logits = model(**inputs).logits_per_image
                        probs = logits.softmax(dim=1).detach().cpu().tolist()[0]
                    dominant_score = max(probs[0], probs[1])
                    has_humans = probs[2] >= 0.28
                    has_major_clutter = probs[3] >= 0.30
                    has_multiple_salient_objects = probs[4] >= 0.30
                    target_dominant = dominant_score >= 0.24
                    details = {
                        "target_dominant": target_dominant,
                        "has_humans": has_humans,
                        "has_major_clutter": has_major_clutter,
                        "has_multiple_salient_objects": has_multiple_salient_objects,
                        "clip_prompt_probs": {
                            "dominant_item": probs[0],
                            "dominant_category": probs[1],
                            "has_human": probs[2],
                            "major_clutter": probs[3],
                            "multiple_salient": probs[4],
                        },
                    }
                    if target_dominant and not has_humans and not has_major_clutter and not has_multiple_salient_objects:
                        decision = "accepted"
                        score = dominant_score
                        reason = "target_dominant_enough"
                    else:
                        decision = "rejected"
                        score = dominant_score
                        reason = "prefilter_target_not_dominant_or_too_cluttered"
        except Exception as exc:
            bump_model_health(db_path, "clip-prefilter", "prefilter", success=False, latency_ms=(time.perf_counter() - started) * 1000.0, error=str(exc))
            logging.exception("prefilter failed for %s", raw_path)
            continue

        bump_model_health(db_path, "clip-prefilter", "prefilter", success=True, latency_ms=(time.perf_counter() - started) * 1000.0)
        updates: dict[str, Any] = {
            "raw_path": str(raw_path),
            "filtered_path": str(filtered_path) if filtered_path.exists() else None,
            "prefilter_decision": decision,
            "prefilter_score": score,
            "prefilter_reason": reason,
            "prefilter_details_json": json.dumps(details, sort_keys=True),
        }
        if decision == "rejected":
            updates["final_decision"] = "rejected"
            copy_decision_file(filtered_path if filtered_path.exists() else raw_path, rejected_root, str(row["category"]), str(row["item"]))
            bump_domain_health(db_path, source_engine, source_domain, prefilter_rejections=1, rejected=1)
        upsert_image(db_path, updates)
        counts[decision] += 1
        logging.info("[prefilter] done raw=%s decision=%s score=%.4f reason=%s", raw_path, decision, score, reason)
    return dict(counts)


def run_yolo_prefilter(
    db_path: Path,
    filtered_root: Path,
    rejected_root: Path,
    yolo_model_id: str,
) -> dict[str, int]:
    rows = pending_yolo_rows(db_path)
    counts = Counter()
    model, device = load_yolo_prefilter(yolo_model_id)
    logging.info("[yolo_prefilter] pending_rows=%d", len(rows))
    for row in rows:
        raw_path = Path(str(row["raw_path"]))
        filtered_path = Path(str(row["filtered_path"])) if row["filtered_path"] else filtered_root / str(row["category"]) / slugify(str(row["item"])) / raw_path.name
        ensure_dir(filtered_path.parent)
        if row["filtered_path"] and raw_path.exists() and not filtered_path.exists():
            shutil.copy2(raw_path, filtered_path)

        source_engine = str(row["source_engine"] or "unknown")
        source_domain = str(row["source_domain"] or "unknown")
        if model is None:
            upsert_image(
                db_path,
                {
                    "raw_path": str(raw_path),
                    "yolo_prefilter_decision": "accepted",
                    "yolo_prefilter_score": 1.0,
                    "yolo_prefilter_reason": "yolo_prefilter_unavailable",
                    "yolo_prefilter_details_json": json.dumps({"detected": []}, sort_keys=True),
                },
            )
            counts["accepted"] += 1
            logging.info("[yolo_prefilter] accepted_without_yolo raw=%s", raw_path)
            continue

        started = time.perf_counter()
        logging.info("[yolo_prefilter] start raw=%s category=%s item=%s", raw_path, row["category"], row["item"])
        try:
            image_source = filtered_path if filtered_path.exists() else raw_path
            with Image.open(image_source) as image:
                width, height = image.size
            results = model.predict(source=str(image_source), verbose=False, imgsz=640, conf=0.25, device=device)
            detections: list[dict[str, Any]] = []
            if results:
                result = results[0]
                names = getattr(result, "names", None) or getattr(model, "names", {})
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    for box in boxes:
                        xyxy = box.xyxy[0].tolist()
                        left, top, right, bottom = [float(v) for v in xyxy]
                        area_ratio = max(0.0, (right - left) * (bottom - top) / max(width * height, 1))
                        cls_idx = int(box.cls[0].item() if hasattr(box.cls[0], "item") else box.cls[0])
                        label = names.get(cls_idx, str(cls_idx)) if isinstance(names, dict) else names[cls_idx]
                        detections.append(
                            {
                                "label": str(label),
                                "confidence": float(box.conf[0].item() if hasattr(box.conf[0], "item") else box.conf[0]),
                                "area_ratio": area_ratio,
                            }
                        )

            person_boxes = [det for det in detections if det["label"] == "person"]
            max_person_area = max((det["area_ratio"] for det in person_boxes), default=0.0)
            max_box_area = max((det["area_ratio"] for det in detections), default=0.0)
            total_box_area = sum(det["area_ratio"] for det in detections)
            detection_count = len(detections)

            decision = "accepted"
            reason = "no_dominant_human_or_clutter_detected"
            if max_person_area >= 0.03:
                decision = "rejected"
                reason = "person_dominates_frame"
            elif detection_count >= 10:
                decision = "rejected"
                reason = "too_many_detections"
            elif detection_count >= 5 and total_box_area > 0.85:
                decision = "rejected"
                reason = "scene_too_crowded"
            elif detection_count >= 3 and max_box_area < 0.05:
                decision = "rejected"
                reason = "no_dominant_object_box"

            score = 1.0 - min(1.0, total_box_area)
            details = {
                "image_size": [width, height],
                "detection_count": detection_count,
                "max_person_area_ratio": max_person_area,
                "max_box_area_ratio": max_box_area,
                "total_box_area_ratio": total_box_area,
                "detections": detections,
            }
        except Exception as exc:
            bump_model_health(db_path, "yolo-prefilter", "prefilter", success=False, latency_ms=(time.perf_counter() - started) * 1000.0, error=str(exc))
            logging.exception("[yolo_prefilter] failed raw=%s", raw_path)
            continue

        bump_model_health(db_path, "yolo-prefilter", "prefilter", success=True, latency_ms=(time.perf_counter() - started) * 1000.0)
        updates: dict[str, Any] = {
            "raw_path": str(raw_path),
            "yolo_prefilter_decision": decision,
            "yolo_prefilter_score": score,
            "yolo_prefilter_reason": reason,
            "yolo_prefilter_details_json": json.dumps(details, sort_keys=True),
        }
        if decision == "rejected":
            updates["prefilter_decision"] = "rejected"
            updates["prefilter_reason"] = f"yolo_{reason}"
            updates["final_decision"] = "rejected"
            copy_decision_file(filtered_path if filtered_path.exists() else raw_path, rejected_root, str(row["category"]), str(row["item"]))
            bump_domain_health(db_path, source_engine, source_domain, prefilter_rejections=1, rejected=1)
        upsert_image(db_path, updates)
        counts[decision] += 1
        logging.info("[yolo_prefilter] done raw=%s decision=%s score=%.4f reason=%s", raw_path, decision, score, reason)
    return dict(counts)


def normalize_photo_stage(parsed: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(parsed)
    normalized["confidence"] = safe_float(parsed.get("confidence"), 0.0)
    for key in (
        "is_real_photo",
        "target_dominant",
        "has_humans",
        "has_major_clutter",
        "has_multiple_salient_objects",
        "is_infographic_or_render",
        "is_abnormal_artistic_case",
        "is_visually_clean",
        "is_trainworthy",
    ):
        normalized[key] = safe_bool(parsed.get(key))
    return normalized


def judge_single_image(
    host: str,
    model: str,
    image_path: Path,
    category: str,
    item: str,
    classes: list[str],
    temperature: float,
    db_path: Path,
) -> dict[str, Any]:
    logging.info("[vlm/judge] start image=%s category=%s item=%s model=%s", image_path, category, item, model)
    class_prompt = class_stage_prompt_for(category, item, classes)
    photo_prompt = photo_stage_prompt_for(category, item)
    class_result = timed_vlm_stage(host, model, image_path, class_prompt, temperature, db_path, "class_stage")
    photo_result = normalize_photo_stage(timed_vlm_stage(host, model, image_path, photo_prompt, temperature, db_path, "trainworthiness_stage"))
    class_accept, class_reject, photo_accept, photo_reject = threshold_profile_for(category, item)

    class_decision = str(class_result.get("decision", "uncertain")).strip().lower()
    photo_decision = str(photo_result.get("decision", "uncertain")).strip().lower()
    class_conf = safe_float(class_result.get("confidence"), 0.0)
    photo_conf = safe_float(photo_result.get("confidence"), 0.0)

    hard_quality_pass = (
        photo_result["is_real_photo"]
        and photo_result["target_dominant"]
        and not photo_result["has_humans"]
        and not photo_result["has_major_clutter"]
        and not photo_result["has_multiple_salient_objects"]
        and not photo_result["is_infographic_or_render"]
        and not photo_result["is_abnormal_artistic_case"]
        and photo_result["is_visually_clean"]
        and photo_result["is_trainworthy"]
    )
    hard_quality_fail = (
        photo_result["is_infographic_or_render"]
        or photo_result["is_abnormal_artistic_case"]
        or photo_result["has_humans"]
        or photo_result["has_major_clutter"]
        or photo_result["has_multiple_salient_objects"]
        or not photo_result["is_real_photo"]
        or not photo_result["is_visually_clean"]
        or not photo_result["is_trainworthy"]
    )

    final_decision = "uncertain"
    if class_decision == category and class_conf >= class_accept and photo_decision == "photo" and photo_conf >= photo_accept and hard_quality_pass:
        final_decision = "accepted"
    elif (class_decision == "reject" and class_conf >= class_reject) or (photo_decision == "synthetic" and photo_conf >= photo_reject) or (hard_quality_fail and photo_conf >= photo_reject):
        final_decision = "rejected"
    logging.info(
        "[vlm/judge] done image=%s class=%s(%.4f) photo=%s(%.4f) final=%s",
        image_path,
        class_decision,
        class_conf,
        photo_decision,
        photo_conf,
        final_decision,
    )

    return {
        "path": str(image_path),
        "category": category,
        "item": item,
        "class_stage": class_result,
        "photo_stage": photo_result,
        "thresholds": {
            "class_accept": class_accept,
            "class_reject": class_reject,
            "photo_accept": photo_accept,
            "photo_reject": photo_reject,
        },
        "final_keep": final_decision == "accepted",
        "final_decision": final_decision,
    }


def run_vlm_pass(
    host: str,
    model: str,
    db_path: Path,
    accepted_root: Path,
    rejected_root: Path,
    uncertain_root: Path,
    categories: list[str],
    temperature: float,
    manifests_root: Path,
) -> dict[str, Any]:
    decisions_path = manifests_root / "decisions.jsonl"
    counts = Counter()
    pending_rows = pending_vlm_rows(db_path)
    logging.info("[vlm] pending_rows=%d model=%s", len(pending_rows), model)
    with decisions_path.open("a", encoding="utf-8") as handle:
        for row in pending_rows:
            image_path = Path(str(row["filtered_path"] or row["raw_path"]))
            if not image_path.exists():
                logging.info("[vlm] skip_missing image=%s", image_path)
                continue
            logging.info("[vlm] start image=%s category=%s item=%s", image_path, row["category"], row["item"])
            try:
                result = judge_single_image(
                    host,
                    model,
                    image_path,
                    str(row["category"]),
                    str(row["item"]),
                    categories,
                    temperature,
                    db_path,
                )
            except Exception:
                logging.exception("VLM failed for %s", image_path)
                continue

            decision = str(result["final_decision"])
            details_json = json.dumps(result["photo_stage"], sort_keys=True)
            upsert_image(
                db_path,
                {
                    "raw_path": str(row["raw_path"]),
                    "class_decision": result["class_stage"].get("decision"),
                    "class_conf": safe_float(result["class_stage"].get("confidence"), 0.0),
                    "class_reason": result["class_stage"].get("reason"),
                    "photo_decision": result["photo_stage"].get("decision"),
                    "photo_conf": safe_float(result["photo_stage"].get("confidence"), 0.0),
                    "photo_reason": result["photo_stage"].get("reason"),
                    "is_real_photo": int(result["photo_stage"].get("is_real_photo", False)),
                    "target_dominant": int(result["photo_stage"].get("target_dominant", False)),
                    "has_humans": int(result["photo_stage"].get("has_humans", False)),
                    "has_major_clutter": int(result["photo_stage"].get("has_major_clutter", False)),
                    "has_multiple_salient_objects": int(result["photo_stage"].get("has_multiple_salient_objects", False)),
                    "is_infographic_or_render": int(result["photo_stage"].get("is_infographic_or_render", False)),
                    "is_abnormal_artistic_case": int(result["photo_stage"].get("is_abnormal_artistic_case", False)),
                    "is_visually_clean": int(result["photo_stage"].get("is_visually_clean", False)),
                    "is_trainworthy": int(result["photo_stage"].get("is_trainworthy", False)),
                    "photo_details_json": details_json,
                    "final_decision": decision,
                    "integration_status": "pending" if decision == "accepted" else "not_applicable",
                },
            )
            decision_root = {"accepted": accepted_root, "rejected": rejected_root, "uncertain": uncertain_root}[decision]
            dst = copy_decision_file(image_path, decision_root, str(row["category"]), str(row["item"]))
            logging.info("[vlm] saved image=%s decision=%s dst=%s", image_path, decision, dst)
            bump_domain_health(
                db_path,
                str(row["source_engine"] or "unknown"),
                str(row["source_domain"] or "unknown"),
                accepted=1 if decision == "accepted" else 0,
                rejected=1 if decision == "rejected" else 0,
                uncertain=1 if decision == "uncertain" else 0,
            )
            handle.write(json.dumps(result) + "\n")
            handle.flush()
            counts[decision] += 1
            logging.info("[vlm] done image=%s decision=%s counts=%s", image_path, decision, dict(counts))

    with connect(db_path) as conn:
        summary_row = conn.execute(
            """
            SELECT
              SUM(CASE WHEN final_decision = 'accepted' THEN 1 ELSE 0 END) AS accepted,
              SUM(CASE WHEN final_decision = 'rejected' THEN 1 ELSE 0 END) AS rejected,
              SUM(CASE WHEN final_decision = 'uncertain' THEN 1 ELSE 0 END) AS uncertain
            FROM images
            """
        ).fetchone()
    accepted = int(summary_row["accepted"] or 0)
    rejected = int(summary_row["rejected"] or 0)
    uncertain = int(summary_row["uncertain"] or 0)
    total = accepted + rejected + uncertain
    summary = {
        "model": model,
        "counts": {"accepted": accepted, "rejected": rejected, "uncertain": uncertain},
        "new_counts": dict(counts),
        "total_decided": total,
        "accepted_pct": (100.0 * accepted / max(total, 1)),
    }
    (manifests_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def integrate_dataset_with_metadata(
    db_path: Path,
    integrated_root: Path,
    dataset_root: Path,
    metadata_file: str,
    source_root_name: str,
) -> dict[str, Any]:
    ensure_dir(dataset_root)
    ensure_dir(integrated_root)
    metadata_path = dataset_root / metadata_file
    metadata = load_metadata(metadata_path)
    metadata_paths = {str(entry.get("file_path", "")).replace("\\", "/") for entry in metadata}
    existing_hashes = build_existing_hash_index(dataset_root)
    logging.info("[dataset] integration_start dataset_root=%s metadata=%s", dataset_root, metadata_path)

    copied = 0
    skipped_duplicate = 0
    per_class = Counter()
    new_records: list[dict[str, Any]] = []

    for row in pending_integration_rows(db_path):
        category = str(row["category"])
        if category not in {"organic", "metal", "paper"}:
            upsert_image(db_path, {"raw_path": str(row["raw_path"]), "integration_status": "unsupported_class"})
            logging.info("[dataset] unsupported_class raw=%s category=%s", row["raw_path"], category)
            continue
        image_path = Path(str(row["filtered_path"] or row["raw_path"]))
        if not image_path.exists():
            logging.info("[dataset] missing_image raw=%s", row["raw_path"])
            continue
        digest = str(row["sha256"] or hash_file(image_path))
        if digest in existing_hashes:
            skipped_duplicate += 1
            upsert_image(
                db_path,
                {
                    "raw_path": str(row["raw_path"]),
                    "integration_status": "skipped_duplicate",
                    "dataset_path": existing_hashes[digest][1],
                },
            )
            logging.info("[dataset] skipped_duplicate raw=%s digest=%s existing=%s", row["raw_path"], digest, existing_hashes[digest][1])
            continue

        item = str(row["item"])
        dst_category_dir = dataset_root / category
        ensure_dir(dst_category_dir)
        stem = f"ollama_{category}_{slugify(item)}_{digest[:12]}"
        destination = dst_category_dir / f"{stem}{image_path.suffix.lower()}"
        suffix = 1
        while destination.exists():
            destination = dst_category_dir / f"{stem}_{suffix}{image_path.suffix.lower()}"
            suffix += 1
        shutil.copy2(image_path, destination)

        integrated_copy = integrated_root / category / slugify(item) / destination.name
        ensure_dir(integrated_copy.parent)
        if not integrated_copy.exists():
            shutil.copy2(image_path, integrated_copy)

        relative_destination = destination.relative_to(dataset_root).as_posix()
        metadata_key = f"{dataset_root.name}/{relative_destination}"
        metadata_paths.add(metadata_key)
        existing_hashes[digest] = (category, relative_destination)
        copied += 1
        per_class[category] += 1
        new_records.append(
            {
                "file_path": metadata_key,
                "label": category,
                "source_dataset": "ollama_pipeline",
                "source_tree": source_root_name,
                "source_class": category,
                "source_subclass": item.replace("_", " "),
                "source_path": image_path.as_posix(),
                "source_sha256": digest,
                "integration_rule": "single_model_accepted",
            }
        )
        upsert_image(
            db_path,
            {
                "raw_path": str(row["raw_path"]),
                "integration_status": "integrated",
                "dataset_path": relative_destination,
            },
        )
        logging.info("[dataset] integrated raw=%s dst=%s label=%s item=%s", row["raw_path"], relative_destination, category, item)

    if new_records:
        metadata.extend(new_records)
        save_metadata(metadata_path, metadata)
        logging.info("[dataset] metadata_updated records=%d", len(new_records))

    return {
        "dataset_root": str(dataset_root),
        "metadata_path": str(metadata_path),
        "copied": copied,
        "skipped_duplicate": skipped_duplicate,
        "per_class": dict(per_class),
    }


def write_split_manifest(dataset_root: Path, manifests_root: Path, seed: int, ratios: tuple[float, float, float]) -> dict[str, Any]:
    split_root = manifests_root / "split_manifest"
    ensure_dir(split_root)
    logging.info("[split] building dataset_root=%s seed=%d ratios=%s", dataset_root, seed, ratios)
    train_samples, val_samples, test_samples, class_names = build_splits(dataset_root, seed=seed, ratios=ratios)
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    def summarize(samples: list[tuple[str, int]]) -> dict[str, Any]:
        counts = Counter(idx_to_class[target] for _, target in samples)
        return {"count": len(samples), "per_class": dict(counts)}

    summary = {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "class_names": class_names,
        "train": summarize(train_samples),
        "val": summarize(val_samples),
        "test": summarize(test_samples),
    }
    (split_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for split_name, samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        lines = [str(Path(path).resolve().relative_to(dataset_root.resolve())) for path, _ in samples]
        (split_root / f"{split_name}_paths.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logging.info("[split] %s count=%d", split_name, len(samples))
    return summary


def export_health_snapshots(db_path: Path, manifests_root: Path, logs_root: Path) -> None:
    ensure_dir(manifests_root)
    ensure_dir(logs_root)
    logging.info("[health] exporting snapshots")
    (logs_root / "domain_health.json").write_text(
        json.dumps([dict(row) for row in all_domain_health_rows(db_path)], indent=2) + "\n",
        encoding="utf-8",
    )
    (logs_root / "model_health.json").write_text(
        json.dumps([dict(row) for row in all_model_health_rows(db_path)], indent=2) + "\n",
        encoding="utf-8",
    )
    (manifests_root / "download_jobs.json").write_text(
        json.dumps([dict(row) for row in all_download_job_rows(db_path)], indent=2) + "\n",
        encoding="utf-8",
    )
    with connect(db_path) as conn:
        summary_row = conn.execute(
            """
            SELECT
              COUNT(*) AS total_rows,
              SUM(CASE WHEN exact_dedupe_outcome = 'unique' AND phash_dedupe_outcome = 'unique' THEN 1 ELSE 0 END) AS unique_rows,
              SUM(CASE WHEN final_decision = 'accepted' THEN 1 ELSE 0 END) AS accepted,
              SUM(CASE WHEN final_decision = 'rejected' THEN 1 ELSE 0 END) AS rejected,
              SUM(CASE WHEN final_decision = 'uncertain' THEN 1 ELSE 0 END) AS uncertain,
              SUM(CASE WHEN integration_status = 'integrated' THEN 1 ELSE 0 END) AS integrated
            FROM images
            """
        ).fetchone()
    (manifests_root / "provenance_summary.json").write_text(
        json.dumps(dict(summary_row) if summary_row is not None else {}, indent=2) + "\n",
        encoding="utf-8",
    )


def run_training(training_command: str) -> None:
    if not training_command.strip():
        raise ValueError("--training-command must be set when --run-training is enabled")
    logging.info("[training] command=%s", training_command)
    subprocess.run(training_command, shell=True, check=True, cwd=str(repo_root()))


def main() -> int:
    args = parse_args()
    if args.vision_models:
        args.vision_model = args.vision_models[0]

    class_spec = normalize_class_spec(load_json_file_or_inline(args.class_spec))
    output_root = (repo_root() / args.output_root).resolve()
    if args.force and output_root.exists():
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s", force=True)
        logging.info("Force removing output root: %s", output_root)
        shutil.rmtree(output_root)

    dirs = artifact_dirs(output_root)
    setup_logging(dirs["logs"])
    logging.info("Pipeline starting output_root=%s dataset_root=%s", output_root, (repo_root() / args.dataset_root).resolve())
    logging.info(
        "Config text_model=%s vision_model=%s item_workers=%d engine_workers=%d min_resolution=%d integrate_accepted=%s write_split_manifest=%s",
        args.text_model,
        args.vision_model,
        args.item_workers,
        args.engine_workers,
        args.min_resolution,
        args.integrate_accepted,
        args.write_split_manifest,
    )
    logging.info(
        "Limits direct_bing=%d google=%d bing=%d baidu=%d temperatures text=%.3f vision=%.3f",
        args.direct_bing_limit,
        args.google_limit,
        args.bing_limit,
        args.baidu_limit,
        args.text_temperature,
        args.vision_temperature,
    )
    db_path = dirs["manifests"] / "pipeline.sqlite"
    init_db(db_path)
    logging.info("SQLite manifest initialized at %s", db_path)

    dataset_root = (repo_root() / args.dataset_root).resolve()
    required_models = [args.text_model, args.vision_model]
    ensure_ollama_models(args.ollama_host, required_models, args.pull_missing_models)

    discovered_path = dirs["class_discovery"] / "discovered_classes.json"
    if discovered_path.exists():
        logging.info("Reusing discovered classes from %s", discovered_path)
        discovered = normalize_class_spec(json.loads(discovered_path.read_text(encoding="utf-8")))
    else:
        logging.info("Discovering classes via text model")
        discovered = discover_objects(
            args.ollama_host,
            args.text_model,
            class_spec,
            temperature=args.text_temperature,
            db_path=db_path,
        )
        discovered_path.write_text(json.dumps(discovered, indent=2) + "\n", encoding="utf-8")
        logging.info("Saved discovered classes to %s", discovered_path)

    jobs = [(category, item) for category, payload in discovered.items() for item in payload.get("seed_objects", [])]
    logging.info("Download jobs planned=%d", len(jobs))
    with ThreadPoolExecutor(max_workers=max(1, args.item_workers)) as pool:
        futures = [
            pool.submit(download_query_bundle, category, item, args, dirs["raw"], dirs["filtered"], db_path)
            for category, item in jobs
        ]
        for future in as_completed(futures):
            category, item, kept = future.result()
            logging.info("[download] %s/%s kept=%d", category, item, kept)

    logging.info("Starting CLIP prefilter")
    prefilter_summary = run_clip_prefilter(db_path, dirs["filtered"], dirs["rejected"], args.min_resolution)
    logging.info("[prefilter] %s", json.dumps(prefilter_summary, sort_keys=True))

    logging.info("Starting YOLO prefilter model=%s", args.yolo_model)
    yolo_summary = run_yolo_prefilter(db_path, dirs["filtered"], dirs["rejected"], args.yolo_model)
    logging.info("[yolo_prefilter] %s", json.dumps(yolo_summary, sort_keys=True))

    categories = list(discovered.keys())
    logging.info("Starting VLM pass model=%s categories=%s", args.vision_model, categories)
    vlm_summary = run_vlm_pass(
        args.ollama_host,
        args.vision_model,
        db_path,
        dirs["accepted"],
        dirs["rejected"],
        dirs["uncertain"],
        categories,
        args.vision_temperature,
        dirs["manifests"],
    )
    logging.info("[vlm] %s", json.dumps(vlm_summary, sort_keys=True))

    if args.integrate_accepted:
        logging.info("Starting dataset integration into %s", dataset_root)
        integration_summary = integrate_dataset_with_metadata(
            db_path,
            dirs["integrated"],
            dataset_root,
            args.metadata_file,
            output_root.name,
        )
        (dirs["manifests"] / "integration_summary.json").write_text(json.dumps(integration_summary, indent=2) + "\n", encoding="utf-8")
        logging.info("[dataset] %s", json.dumps(integration_summary, sort_keys=True))
        if args.write_split_manifest:
            logging.info("Writing split manifest")
            split_summary = write_split_manifest(
                dataset_root,
                dirs["manifests"],
                seed=args.split_seed,
                ratios=parse_split_ratios(args.split_ratios),
            )
            logging.info("[split] %s", json.dumps(split_summary, sort_keys=True))

    export_health_snapshots(db_path, dirs["manifests"], dirs["logs"])
    logging.info("Exported health snapshots")

    if args.run_training:
        run_training(args.training_command)

    logging.info("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
