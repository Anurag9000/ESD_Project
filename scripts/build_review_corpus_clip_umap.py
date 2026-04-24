#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from io import BytesIO
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import URLError, HTTPError
from urllib.parse import quote_plus, urlsplit
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import umap
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA
from torchvision.transforms.functional import InterpolationMode
from transformers.generation.utils import GenerationConfig, GenerationMixin

try:
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover - optional backend
    AutoModel = None
    AutoProcessor = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    Qwen2_5_VLForConditionalGeneration = None


matplotlib.use("Agg")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


PAPER_ITEMS = [
    "envelopes",
    "flyers",
    "brochures",
    "receipts",
    "documents",
    "cardboard",
    "notepads",
    "postcards",
    "wrapping paper",
    "paper bags",
    "cardboard boxes carton",
    "cereal boxes",
    "sticky notes",
    "notebooks",
    "newspapers",
    "magazines",
    "books",
    "paper cups",
    "paper plates",
    "tissue paper",
    "napkins",
    "paper towels",
    "calendars",
    "posters",
    "labels",
    "shopping receipts",
    "cards",
    "greeting cards",
    "index cards",
    "file folders",
    "folders",
    "mailers",
    "tickets",
    "forms",
    "letters",
    "checks",
    "invoices",
    "booklets",
    "manuals",
    "charts",
    "printouts",
    "paper packaging",
]

METAL_ITEMS = [
    "cans",
    "tin boxes",
    "bottle caps",
    "lids",
    "foil",
    "trays",
    "pans",
    "keys",
    "coins",
    "screws",
    "nails",
    "bolts",
    "nuts",
    "chains",
    "wire hangers",
    "tools",
    "utensils",
    "bottles",
    "scissors",
    "knives",
    "pliers",
    "spoons",
    "forks",
    "cutlery",
    "plates",
    "taps",
    "washers",
    "clamps",
    "hinges",
    "wrenches",
    "pots",
    "ladles",
    "spatulas",
    "grates",
    "rails",
    "rods",
    "pipes",
    "hooks",
    "locks",
    "latches",
    "hammers",
    "screwdrivers",
]

CATEGORY_ITEMS: dict[str, list[str]] = {
    "paper": PAPER_ITEMS,
    "metal": METAL_ITEMS,
}

CATEGORY_PROMPTS: dict[str, list[str]] = {
    "paper": [
        "a photo of paper objects",
        "a photo of cardboard and paper materials",
        "a photo of office paper, paper packaging, and paper goods",
    ],
    "metal": [
        "a photo of metal objects",
        "a photo of metal tools, utensils, and hardware",
        "a photo of metal cans, keys, coins, screws, bolts, and nuts",
    ],
}

ITEM_PROMPT_TEMPLATES = [
    "a photo of {}",
    "a close-up photo of {}",
    "a product photo of {}",
    "an image of {}",
    "a real-world photograph of {} for a training dataset",
    "a camera-captured example of {} suitable for supervised dataset curation",
    "a natural object photo of {} that should be kept in a curated training dataset",
]

PHOTO_PROMPTS_BY_CATEGORY: dict[str, list[str]] = {
    "paper": [
        "a real camera photo of a physical paper object",
        "a handheld mobile photo of a real printed paper item",
        "a real-life photograph of a tangible paper product with printed text or graphics",
        "a genuine camera image of a physical cardboard or paper object",
    ],
    "metal": [
        "a real camera photo of a physical metal object",
        "a handheld photo of a real metallic everyday object",
        "a real-life photograph of a tangible metal tool, utensil, or hardware object",
        "a genuine camera image of a physical reflective metal object",
    ],
}

NON_PHOTO_PROMPTS = [
    "an infographic that is not a camera photo",
    "a screenshot",
    "a digital illustration",
    "a vector graphic",
    "a rendered 3d object",
    "an ai generated image",
    "a synthetic image",
    "a cgi render",
    "clipart",
    "an icon set",
    "a stock mockup render",
    "a flat graphic design composition",
]

VLM_SYSTEM_PROMPT = (
    "You are a strict dataset curation judge for a supervised image classification pipeline. "
    "Your task is to decide whether a single image is a clean, real-world training example for either "
    "the paper superclass or the metal superclass. "
    "Reject borderline, synthetic, infographic-like, diagram-like, screenshot-like, rendered, clipart, "
    "logo, collage, or AI-generated images. "
    "Return only valid JSON with no markdown and no extra text."
)

VLM_CLASS_PROMPT_BY_CATEGORY = {
    "paper": (
        "Target superclass: paper.\n"
        "Keep only real camera photos of physical paper or cardboard objects. "
        "Valid examples include envelopes, flyers, brochures, receipts, documents, cardboard, notepads, "
        "postcards, paper bags, books, newspapers, magazines, paper cups, paper plates, and paper packaging. "
        "Printed text, labels, or background clutter are okay only if the image is still clearly a real photo "
        "of a paper or cardboard object. "
        "Reject anything that looks like an infographic, chart, poster design, screenshot, logo, vector art, "
        "CG render, AI-generated image, or synthetic composition.\n"
        "Answer with exactly one token: paper, metal, or reject."
    ),
    "metal": (
        "Target superclass: metal.\n"
        "Keep only real camera photos of physical metal objects. "
        "Valid examples include cans, keys, coins, screws, nails, bolts, nuts, tools, utensils, bottles, "
        "scissors, knives, pliers, spoons, forks, cutlery, plates, pans, trays, taps, washers, clamps, hinges, "
        "wrenches, pots, ladles, spatulas, grates, rails, rods, pipes, hooks, locks, latches, hammers, and screwdrivers. "
        "Reject anything that looks like an infographic, chart, poster design, screenshot, logo, vector art, "
        "CG render, AI-generated image, or synthetic composition.\n"
        "Answer with exactly one token: paper, metal, or reject."
    ),
}

VLM_PHOTO_PROMPT_BY_CATEGORY = {
    "paper": (
        "Target superclass: paper.\n"
        "Decide whether the image is a genuine camera photo of a real physical paper or cardboard object. "
        "Printed text, labels, or clutter are okay if the image is still a real photo. "
        "Reject infographic-like, diagram-like, screenshot-like, logo-like, vector art, CG render, AI-generated, "
        "or synthetic images.\n"
        "Answer with exactly one token: photo or reject."
    ),
    "metal": (
        "Target superclass: metal.\n"
        "Decide whether the image is a genuine camera photo of a real physical metal object. "
        "Reflective surfaces and clutter are okay if the image is still a real photo. "
        "Reject infographic-like, diagram-like, screenshot-like, logo-like, vector art, CG render, AI-generated, "
        "or synthetic images.\n"
        "Answer with exactly one token: photo or reject."
    ),
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class ItemResult:
    category: str
    item: str
    raw_count: int
    kept_count: int
    rejected_count: int


@dataclass(frozen=True)
class RawSample:
    category: str
    item: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download, CLIP-filter, and UMAP a paper/metal review corpus.")
    parser.add_argument("--output-root", default="review_downloads/paper_metal_clip_review")
    parser.add_argument("--limit-per-query", type=int, default=1000)
    parser.add_argument("--download-buffer", type=int, default=1200)
    parser.add_argument(
        "--item-workers",
        type=int,
        default=min(20, max(4, os.cpu_count() or 20)),
        help="Parallel item jobs.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=min(64, max(16, (os.cpu_count() or 20) * 2)),
        help="Parallel image download workers per item.",
    )
    parser.add_argument(
        "--page-workers",
        type=int,
        default=min(32, max(8, os.cpu_count() or 20)),
        help="Parallel Bing page fetch workers per query.",
    )
    parser.add_argument("--max-query-pages", type=int, default=80, help="Hard cap on Bing image result pages fetched per query.")
    parser.add_argument(
        "--idle-page-rounds",
        type=int,
        default=3,
        help="Stop a query after this many consecutive page batches produce no new URLs.",
    )
    parser.add_argument("--min-resolution", type=int, default=224)
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--clip-pretrained", default="openai")
    parser.add_argument("--clip-batch-size", type=int, default=1365)
    parser.add_argument("--clip-min-prob", type=float, default=0.75)
    parser.add_argument("--clip-min-margin", type=float, default=0.35)
    parser.add_argument("--photo-min-prob", type=float, default=0.55)
    parser.add_argument("--photo-min-margin", type=float, default=0.05)
    parser.add_argument("--judge-backend", choices=["clip", "vlm"], default="clip")
    parser.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--vlm-quantization", choices=["fp16", "8bit", "4bit"], default="fp16")
    parser.add_argument("--vlm-batch-size", type=int, default=2)
    parser.add_argument("--vlm-max-new-tokens", type=int, default=64)
    parser.add_argument("--vlm-min-confidence", type=float, default=0.80)
    parser.add_argument("--vlm-min-photo-confidence", type=float, default=0.80)
    parser.add_argument("--vlm-min-pixels", type=int, default=224 * 224)
    parser.add_argument("--vlm-max-pixels", type=int, default=512 * 512)
    parser.add_argument("--vlm-max-gpu-memory-gib", type=float, default=5.5)
    parser.add_argument("--vlm-max-cpu-memory-gib", type=float, default=48.0)
    parser.add_argument("--internvl-input-size", type=int, default=448)
    parser.add_argument("--internvl-max-num", type=int, default=1)
    parser.add_argument("--umap-thumb-size", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download-only", action="store_true", help="Only download raw images; skip CLIP filtering and UMAP.")
    parser.add_argument("--clip-only", action="store_true", help="Skip downloads and run the configured judge on existing raw files.")
    parser.add_argument("--startup-vram-limit-mib", type=int, default=6144)
    parser.add_argument("--startup-vram-guard-seconds", type=int, default=120)
    parser.add_argument("--startup-vram-poll-seconds", type=float, default=2.0)
    parser.add_argument("--force", action="store_true", help="Rebuild the output root from scratch.")
    return parser.parse_args()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def image_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [path for path in sorted(root.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_EXTS]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def fetch_bing_page_links(query: str, page_index: int, page_size: int, adult: str = "off") -> tuple[int, list[str]]:
    headers = bing_headers()
    request_url = (
        "https://www.bing.com/images/async?q="
        + quote_plus(query)
        + f"&first={page_index * page_size}&count={page_size}&adlt={adult}&qft="
    )
    try:
        response = urlopen(Request(request_url, None, headers=headers), timeout=30)
        html = response.read().decode("utf8", errors="ignore")
    except Exception as exc:
        print(f"[download] bing page error for {query}: {exc}")
        return page_index, []
    if not html:
        return page_index, []
    page_links = re.findall(r'murl&quot;:&quot;(.*?)&quot;', html)
    return page_index, page_links


def fetch_bing_image_links(
    query: str,
    limit: int,
    page_size: int = 150,
    adult: str = "off",
    page_workers: int = 1,
) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()
    page_count = max(1, math.ceil(limit / page_size))

    page_workers = max(1, page_workers)
    with ThreadPoolExecutor(max_workers=page_workers) as pool:
        for _, page_links in sorted(
            pool.map(lambda page_index: fetch_bing_page_links(query, page_index, page_size, adult), range(page_count)),
            key=lambda x: x[0],
        ):
            if not page_links:
                continue
            for link in page_links:
                if link not in seen:
                    seen.add(link)
                    links.append(link)
                    if len(links) >= limit:
                        return links
    return links


def download_image_candidate(
    link: str,
    output_dir: Path,
    index: int,
    min_resolution: int,
    timeout: int,
) -> Path | None:
    headers = bing_headers()
    try:
        request = Request(link, None, headers=headers)
        data = urlopen(request, timeout=timeout).read()
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
            if width < min_resolution or height < min_resolution:
                return None
            fmt = (image.format or "JPEG").lower()
            if fmt == "jpeg":
                fmt = "jpg"
            ext = f".{fmt}" if fmt in {"jpg", "png", "webp", "bmp", "gif", "tiff"} else ".jpg"
            dst = output_dir / f"Image_{index:06d}{ext}"
            image.convert("RGB").save(dst)
            return dst
    except (HTTPError, URLError, OSError, ValueError):
        return None


def download_and_clean_item(
    category: str,
    item: str,
    raw_root: Path,
    limit: int,
    min_resolution: int,
    download_buffer: int,
    download_workers: int,
    page_workers: int,
    max_query_pages: int,
    idle_page_rounds: int,
) -> tuple[str, str, Path | None, list[Path]]:
    query = f"{category} {item}"
    category_dir = raw_root / category
    final_dir = category_dir / slugify(item)
    ensure_dir(final_dir)
    print(f"[download] {query}")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    ensure_dir(final_dir)

    page_size = 150
    page_workers = max(1, page_workers)
    download_workers = max(1, download_workers)
    max_query_pages = max(1, max_query_pages)
    idle_page_rounds = max(1, idle_page_rounds)
    seen_links: set[str] = set()
    saved_paths: list[Path] = []
    pending_downloads: set = set()

    def drain_completed() -> None:
        nonlocal pending_downloads, saved_paths
        done = [future for future in pending_downloads if future.done()]
        for future in done:
            pending_downloads.discard(future)
            try:
                path = future.result()
            except Exception:
                path = None
            if path is not None:
                saved_paths.append(path)

    path_lock = threading.Lock()
    next_index = 1

    def reserve_index() -> int:
        nonlocal next_index
        with path_lock:
            value = next_index
            next_index += 1
            return value

    def worker(link: str) -> Path | None:
        return download_image_candidate(link, final_dir, reserve_index(), min_resolution, timeout=30)

    with ThreadPoolExecutor(max_workers=download_workers) as download_pool:
        pages_fetched = 0
        idle_rounds = 0
        while len(saved_paths) < limit and pages_fetched < max_query_pages:
            round_count = min(page_workers, max_query_pages - pages_fetched)
            try:
                with ThreadPoolExecutor(max_workers=page_workers) as page_pool:
                    page_futures = [
                        page_pool.submit(fetch_bing_page_links, query, pages_fetched + offset, page_size)
                        for offset in range(round_count)
                    ]
                    new_links_this_round = 0
                    for page_future in as_completed(page_futures):
                        try:
                            _, page_links = page_future.result()
                        except Exception:
                            page_links = []
                        for link in page_links:
                            if link in seen_links:
                                continue
                            seen_links.add(link)
                            new_links_this_round += 1
                            pending_downloads.add(download_pool.submit(worker, link))
                    pages_fetched += round_count
            except Exception as exc:
                print(f"[download] error scheduling pages for {query}: {exc}")
                break

            drain_completed()

            # If the source is noisy, keep searching beyond the original buffer until we either
            # hit the target or Bing stops yielding new URLs.
            minimum_link_budget = max(download_buffer, limit * 4)
            if len(seen_links) < minimum_link_budget and pages_fetched < max_query_pages:
                idle_rounds = 0
            elif new_links_this_round == 0:
                idle_rounds += 1
            else:
                idle_rounds = 0

            while pending_downloads and len(saved_paths) < limit and len(pending_downloads) >= download_workers:
                future = pending_downloads.pop()
                try:
                    path = future.result()
                except Exception:
                    path = None
                if path is not None:
                    saved_paths.append(path)

            if idle_rounds >= idle_page_rounds and not pending_downloads:
                break

        while pending_downloads and len(saved_paths) < limit:
            future = pending_downloads.pop()
            try:
                path = future.result()
            except Exception:
                path = None
            if path is not None:
                saved_paths.append(path)

        for future in pending_downloads:
            future.cancel()

    saved_paths = sorted(saved_paths)
    return category, item, final_dir, saved_paths


def load_clip(device: torch.device, model_name: str, pretrained: str):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device=device)
    model.eval()
    return model, preprocess, tokenizer


def load_vlm(model_name: str, min_pixels: int, max_pixels: int, max_gpu_memory_gib: float, max_cpu_memory_gib: float):
    if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None:
        raise RuntimeError(
            "transformers is required for --judge-backend vlm. Install the repo requirements or run `pip install transformers accelerate`."
        )
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        max_memory={0: f"{max_gpu_memory_gib:.1f}GiB", "cpu": f"{max_cpu_memory_gib:.1f}GiB"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, processor


def load_vlm_quantized(
    model_name: str,
    min_pixels: int,
    max_pixels: int,
    max_gpu_memory_gib: float,
    max_cpu_memory_gib: float,
    quantization: str,
):
    if AutoProcessor is None or Qwen2_5_VLForConditionalGeneration is None or BitsAndBytesConfig is None:
        raise RuntimeError(
            "transformers with bitsandbytes support is required for --vlm-quantization 8bit/4bit. "
            "Install the repo requirements or run `pip install transformers accelerate bitsandbytes`."
        )
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixels, max_pixels=max_pixels)
    quantization_config = None
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
        "max_memory": {0: f"{max_gpu_memory_gib:.1f}GiB", "cpu": f"{max_cpu_memory_gib:.1f}GiB"},
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    model_kwargs["quantization_config"] = quantization_config
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    model.eval()
    return model, processor


def load_internvl(model_name: str, max_gpu_memory_gib: float, max_cpu_memory_gib: float):
    if AutoModel is None or AutoTokenizer is None:
        raise RuntimeError(
            "transformers is required for --judge-backend vlm. Install the repo requirements or run `pip install transformers accelerate`."
        )
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map="auto",
        max_memory={0: f"{max_gpu_memory_gib:.1f}GiB", "cpu": f"{max_cpu_memory_gib:.1f}GiB"},
    ).eval()
    if not hasattr(model.language_model, "generate"):
        patched_cls = type(
            f"Patched{model.language_model.__class__.__name__}",
            (model.language_model.__class__, GenerationMixin),
            {},
        )
        model.language_model.__class__ = patched_cls
    if getattr(model.language_model, "generation_config", None) is None:
        model.language_model.generation_config = GenerationConfig.from_model_config(model.language_model.config)
    model.language_model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def vlm_parse_json(text: str) -> dict[str, object] | None:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").strip().removesuffix("```").strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def build_vlm_messages(category: str, path: Path) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(path)},
                {"type": "text", "text": VLM_CLASS_PROMPT_BY_CATEGORY[category]},
            ],
        },
    ]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: set[tuple[int, int]], width: int, height: int, image_size: int):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Image.Image, min_num: int = 1, max_num: int = 1, image_size: int = 448, use_thumbnail: bool = False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_file: Path, input_size: int = 448, max_num: int = 1) -> torch.Tensor:
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch.stack(pixel_values)


def encode_texts(model, tokenizer, prompts: list[str], device: torch.device) -> torch.Tensor:
    tokens = tokenizer(prompts).to(device)
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        text_features = model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)
    return text_features


def build_category_prompt_bank(model, tokenizer, device: torch.device) -> tuple[list[str], torch.Tensor]:
    category_names = sorted(CATEGORY_PROMPTS.keys())
    prompt_rows: list[torch.Tensor] = []
    for category in category_names:
        text_features = encode_texts(model, tokenizer, CATEGORY_PROMPTS[category], device)
        prompt_rows.append(F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1))
    return category_names, torch.cat(prompt_rows, dim=0)


def build_photo_prompt_bank(model, tokenizer, device: torch.device) -> dict[str, torch.Tensor]:
    non_photo_features = encode_texts(model, tokenizer, NON_PHOTO_PROMPTS, device)
    non_photo_mean = F.normalize(non_photo_features.mean(dim=0), dim=-1)
    bank: dict[str, torch.Tensor] = {}
    for category, prompts in PHOTO_PROMPTS_BY_CATEGORY.items():
        photo_features = encode_texts(model, tokenizer, prompts, device)
        bank[category] = torch.stack(
            [
                F.normalize(photo_features.mean(dim=0), dim=-1),
                non_photo_mean,
            ],
            dim=0,
        )
    return bank


def start_startup_vram_guard(limit_mib: int, guard_seconds: int, poll_seconds: float) -> None:
    if guard_seconds <= 0 or limit_mib <= 0:
        return

    def worker() -> None:
        deadline = time.time() + guard_seconds
        while time.time() < deadline:
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                values = [int(line.strip()) for line in output.splitlines() if line.strip()]
                if values and max(values) > limit_mib:
                    print(f"[guard] startup VRAM limit exceeded: {max(values)} MiB > {limit_mib} MiB; aborting")
                    os._exit(2)
            except Exception:
                pass
            time.sleep(poll_seconds)

    threading.Thread(target=worker, daemon=True).start()


def classify_images(
    model,
    preprocess,
    device: torch.device,
    paths: list[Path],
    category: str,
    item_name: str,
    category_names: list[str],
    category_prompt_bank: torch.Tensor,
    photo_prompt_bank: dict[str, torch.Tensor],
    batch_size: int,
    min_prob: float,
    min_margin: float,
    photo_min_prob: float,
    photo_min_margin: float,
) -> tuple[list[Path], list[dict[str, float]]]:
    target_index = category_names.index(category)

    kept: list[Path] = []
    rows: list[dict[str, float]] = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch_images: list[torch.Tensor] = []
        valid_paths: list[Path] = []
        for path in batch_paths:
            try:
                with Image.open(path) as image:
                    batch_images.append(preprocess(image.convert("RGB")))
                    valid_paths.append(path)
            except Exception:
                continue
        if not batch_images:
            continue
        image_tensor = torch.stack(batch_images).to(device=device)
        autocast_enabled = device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            image_features = model.encode_image(image_tensor).float()
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ category_prompt_bank.T
            probs = torch.softmax(logits, dim=1)
            photo_logits = 100.0 * image_features @ photo_prompt_bank[category].T
            photo_probs = torch.softmax(photo_logits, dim=1)

        for path, prob_row, photo_prob_row in zip(valid_paths, probs.cpu().numpy(), photo_probs.cpu().numpy(), strict=True):
            target_prob = float(prob_row[target_index])
            other_prob = float(np.partition(prob_row, -2)[-2]) if len(prob_row) > 1 else 0.0
            margin = target_prob - other_prob
            predicted_index = int(np.argmax(prob_row))
            predicted_category = category_names[predicted_index]
            photo_prob = float(photo_prob_row[0])
            photo_other = float(photo_prob_row[1]) if len(photo_prob_row) > 1 else 0.0
            photo_margin = photo_prob - photo_other
            rows.append(
                {
                    "path": str(path),
                    "predicted_category": predicted_category,
                    "target_category": category,
                    "source_item": item_name,
                    "target_prob": target_prob,
                    "other_prob": other_prob,
                    "margin": margin,
                    "photo_prob": photo_prob,
                    "photo_other_prob": photo_other,
                    "photo_margin": photo_margin,
                }
            )
            if (
                predicted_category == category
                and target_prob >= min_prob
                and margin >= min_margin
                and (photo_prob >= photo_min_prob or photo_margin >= photo_min_margin)
            ):
                kept.append(path)
    return kept, rows


def classify_samples_global(
    model,
    preprocess,
    device: torch.device,
    samples: list[RawSample],
    category_names: list[str],
    category_prompt_bank: torch.Tensor,
    photo_prompt_bank: dict[str, torch.Tensor],
    batch_size: int,
    min_prob: float,
    min_margin: float,
    photo_min_prob: float,
    photo_min_margin: float,
) -> tuple[dict[Path, bool], dict[Path, dict[str, object]]]:
    keep_by_path: dict[Path, bool] = {}
    row_by_path: dict[Path, dict[str, object]] = {}

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch_images: list[torch.Tensor] = []
        valid_samples: list[RawSample] = []
        for sample in batch_samples:
            try:
                with Image.open(sample.path) as image:
                    batch_images.append(preprocess(image.convert("RGB")))
                    valid_samples.append(sample)
            except Exception:
                continue
        if not batch_images:
            continue

        image_tensor = torch.stack(batch_images).to(device=device)
        autocast_enabled = device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            image_features = model.encode_image(image_tensor).float()
            image_features = F.normalize(image_features, dim=-1)
            logits = 100.0 * image_features @ category_prompt_bank.T
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        photo_probs_by_category: dict[str, np.ndarray] = {}
        for category in {sample.category for sample in valid_samples}:
            category_indices = [idx for idx, sample in enumerate(valid_samples) if sample.category == category]
            category_features = image_features[category_indices]
            photo_logits = 100.0 * category_features @ photo_prompt_bank[category].T
            category_photo_probs = torch.softmax(photo_logits, dim=1).cpu().numpy()
            for local_pos, sample_idx in enumerate(category_indices):
                photo_probs_by_category[sample_idx] = category_photo_probs[local_pos]

        for idx, sample in enumerate(valid_samples):
            target_index = category_names.index(sample.category)
            prob_row = probs[idx]
            photo_prob_row = photo_probs_by_category[idx]
            target_prob = float(prob_row[target_index])
            other_prob = float(np.partition(prob_row, -2)[-2]) if len(prob_row) > 1 else 0.0
            margin = target_prob - other_prob
            predicted_index = int(np.argmax(prob_row))
            predicted_category = category_names[predicted_index]
            photo_prob = float(photo_prob_row[0])
            photo_other = float(photo_prob_row[1]) if len(photo_prob_row) > 1 else 0.0
            photo_margin = photo_prob - photo_other
            keep = (
                predicted_category == sample.category
                and target_prob >= min_prob
                and margin >= min_margin
                and (photo_prob >= photo_min_prob or photo_margin >= photo_min_margin)
            )
            row_by_path[sample.path] = {
                "path": str(sample.path),
                "predicted_category": predicted_category,
                "target_category": sample.category,
                "source_item": sample.item,
                "target_prob": target_prob,
                "other_prob": other_prob,
                "margin": margin,
                "photo_prob": photo_prob,
                "photo_other_prob": photo_other,
                "photo_margin": photo_margin,
            }
            keep_by_path[sample.path] = keep

    return keep_by_path, row_by_path


def classify_samples_vlm(
    model,
    processor,
    samples: list[RawSample],
    batch_size: int,
    min_confidence: float,
    min_photo_confidence: float,
) -> tuple[dict[Path, bool], dict[Path, dict[str, object]]]:
    keep_by_path: dict[Path, bool] = {}
    row_by_path: dict[Path, dict[str, object]] = {}
    batch_size = max(1, batch_size)

    tokenizer = processor.tokenizer
    label_token_ids = {
        "paper": tokenizer.encode("paper", add_special_tokens=False)[0],
        "metal": tokenizer.encode("metal", add_special_tokens=False)[0],
        "reject": tokenizer.encode("reject", add_special_tokens=False)[0],
        "photo": tokenizer.encode("photo", add_special_tokens=False)[0],
    }

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        batch_images: list[Image.Image] = []
        class_texts: list[str] = []
        photo_texts: list[str] = []
        valid_samples: list[RawSample] = []
        for sample in batch_samples:
            try:
                with Image.open(sample.path) as image:
                    batch_images.append(image.convert("RGB"))
                class_texts.append(
                    processor.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "path": str(sample.path)},
                                    {"type": "text", "text": VLM_CLASS_PROMPT_BY_CATEGORY[sample.category]},
                                ],
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                photo_texts.append(
                    processor.apply_chat_template(
                        [
                            {
                                "role": "system",
                                "content": [{"type": "text", "text": VLM_SYSTEM_PROMPT}],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "path": str(sample.path)},
                                    {"type": "text", "text": VLM_PHOTO_PROMPT_BY_CATEGORY[sample.category]},
                                ],
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                valid_samples.append(sample)
            except Exception:
                continue
        if not valid_samples:
            continue

        class_inputs = processor(text=class_texts, images=batch_images, padding=True, return_tensors="pt")
        photo_inputs = processor(text=photo_texts, images=batch_images, padding=True, return_tensors="pt")
        input_device = getattr(model, "device", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        if hasattr(class_inputs, "to"):
            class_inputs = class_inputs.to(input_device)
            photo_inputs = photo_inputs.to(input_device)
        else:
            class_inputs = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in class_inputs.items()}
            photo_inputs = {key: value.to(input_device) if hasattr(value, "to") else value for key, value in photo_inputs.items()}

        with torch.inference_mode():
            class_logits = model(**class_inputs).logits[:, -1, :]
            photo_logits = model(**photo_inputs).logits[:, -1, :]

        class_token_indices = torch.tensor(
            [label_token_ids["paper"], label_token_ids["metal"], label_token_ids["reject"]],
            device=class_logits.device,
        )
        photo_token_indices = torch.tensor(
            [label_token_ids["photo"], label_token_ids["reject"]],
            device=photo_logits.device,
        )

        class_selected = class_logits.index_select(dim=1, index=class_token_indices)
        class_probs = torch.softmax(class_selected, dim=1).cpu().numpy()
        photo_selected = photo_logits.index_select(dim=1, index=photo_token_indices)
        photo_probs = torch.softmax(photo_selected, dim=1).cpu().numpy()

        for idx, sample in enumerate(valid_samples):
            class_prob_row = class_probs[idx]
            photo_prob_row = photo_probs[idx]
            class_order = ["paper", "metal", "reject"]
            predicted_category = class_order[int(np.argmax(class_prob_row))]
            class_conf = float(np.max(class_prob_row))
            class_other = float(np.partition(class_prob_row, -2)[-2]) if len(class_prob_row) > 1 else 0.0
            class_margin = class_conf - class_other
            photo_conf = float(photo_prob_row[0])
            photo_other = float(photo_prob_row[1]) if len(photo_prob_row) > 1 else 0.0
            photo_margin = photo_conf - photo_other
            keep = (
                predicted_category == sample.category
                and class_conf >= min_confidence
                and class_margin >= 0.0
                and photo_conf >= min_photo_confidence
                and photo_margin >= 0.0
            )
            row_by_path[sample.path] = {
                "path": str(sample.path),
                "predicted_category": predicted_category,
                "target_category": sample.category,
                "source_item": sample.item,
                "class_confidence": class_conf,
                "class_other_confidence": class_other,
                "class_margin": class_margin,
                "real_photo_confidence": photo_conf,
                "real_photo_other_confidence": photo_other,
                "real_photo_margin": photo_margin,
                "final_keep": keep,
            }
            keep_by_path[sample.path] = keep

    return keep_by_path, row_by_path


def classify_samples_internvl(
    model,
    tokenizer,
    samples: list[RawSample],
    batch_size: int,
    input_size: int,
    max_num: int,
) -> tuple[dict[Path, bool], dict[Path, dict[str, object]]]:
    keep_by_path: dict[Path, bool] = {}
    row_by_path: dict[Path, dict[str, object]] = {}
    batch_size = max(1, batch_size)
    generation_config = dict(max_new_tokens=8, do_sample=False, top_p=1.0)

    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        pixel_values_list: list[torch.Tensor] = []
        num_patches_list: list[int] = []
        questions: list[str] = []
        valid_samples: list[RawSample] = []
        for sample in batch_samples:
            try:
                pixel_values = load_internvl_image(sample.path, input_size=input_size, max_num=max_num)
            except Exception:
                continue
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
            questions.append(
                (
                    f"Target superclass: {sample.category}.\n"
                    f"You are curating a clean training dataset for a supervised model.\n"
                    f"Keep only genuine real-world photographs of {sample.category} objects/materials.\n"
                    f"Reject infographic, diagram, chart, screenshot, logo, collage, vector art, rendered, CGI, AI-generated, or borderline images.\n"
                    f"Answer with exactly one token: {sample.category} or reject."
                )
            )
            valid_samples.append(sample)
        if not valid_samples:
            continue

        pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.float16).cuda()
        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

        for sample, response in zip(valid_samples, responses, strict=True):
            response_text = str(response).strip().lower()
            predicted_category = sample.category if response_text == sample.category else "reject"
            keep = predicted_category == sample.category
            row_by_path[sample.path] = {
                "path": str(sample.path),
                "predicted_category": predicted_category,
                "target_category": sample.category,
                "source_item": sample.item,
                "final_keep": keep,
                "response": response_text,
            }
            keep_by_path[sample.path] = keep

    return keep_by_path, row_by_path


def save_image_with_prefix(src: Path, dst_dir: Path, prefix: str, index: int) -> Path:
    ensure_dir(dst_dir)
    ext = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTS else ".jpg"
    dst = dst_dir / f"{prefix}_{index:05d}{ext}"
    shutil.copy2(src, dst)
    return dst


def gather_category_images(root: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {"paper": [], "metal": []}
    for category in grouped:
        category_dir = root / category
        if category_dir.exists():
            for item_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                grouped[category].extend(image_paths(item_dir))
    return grouped


def gather_raw_item_images(root: Path) -> list[tuple[str, str, Path, list[Path]]]:
    staged: list[tuple[str, str, Path, list[Path]]] = []
    for category in ("paper", "metal"):
        category_dir = root / category
        if not category_dir.exists():
            continue
        for item_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            normalized = item_dir.name.replace("_", " ")
            prefix = f"{category} "
            if normalized.lower().startswith(prefix):
                normalized = normalized[len(prefix) :]
            staged.append((category, normalized, item_dir, image_paths(item_dir)))
    return staged


def gather_raw_samples(root: Path) -> list[RawSample]:
    samples: list[RawSample] = []
    for category, item, _item_dir, paths in gather_raw_item_images(root):
        for path in paths:
            samples.append(RawSample(category=category, item=item, path=path))
    return samples


def persist_judged_samples(
    category: str,
    item: str,
    batch_samples: list[RawSample],
    keep_by_path: dict[Path, bool],
    row_by_path: dict[Path, dict[str, object]],
    accepted_root: Path,
    rejected_root: Path,
    scores_handle,
    accepted_index: dict[str, int],
) -> tuple[int, int]:
    item_accepted_dir = accepted_root / category / slugify(item)
    item_rejected_dir = rejected_root / category / slugify(item)
    ensure_dir(item_accepted_dir)
    ensure_dir(item_rejected_dir)

    accepted_count = 0
    rejected_count = 0
    for sample in batch_samples:
        row = row_by_path.get(sample.path)
        if row is not None:
            scores_handle.write(json.dumps({"category": category, "item": item, **row}) + "\n")
        if keep_by_path.get(sample.path, False):
            accepted_index[category] += 1
            prefix = "METAL" if category == "metal" else "PAPER"
            dst_name = (
                f"{prefix}_{slugify(item)}_{accepted_index[category]:05d}{sample.path.suffix.lower() or '.jpg'}"
                if category == "metal"
                else f"{slugify(item)}_{accepted_index[category]:05d}{sample.path.suffix.lower() or '.jpg'}"
            )
            shutil.copy2(sample.path, item_accepted_dir / dst_name)
            accepted_count += 1
        else:
            shutil.copy2(sample.path, item_rejected_dir / sample.path.name)
            rejected_count += 1
    scores_handle.flush()
    return accepted_count, rejected_count


def render_umap_thumbnail_map(
    embeddings: np.ndarray,
    thumbnails: list[np.ndarray],
    labels: list[str],
    output_path: Path,
    thumb_size: int,
    title: str,
) -> None:
    if embeddings.shape[0] == 0:
        raise RuntimeError("No embeddings available for UMAP rendering.")

    pca_dim = min(50, embeddings.shape[1], max(2, embeddings.shape[0] - 1))
    if pca_dim < embeddings.shape[1]:
        embeddings = PCA(n_components=pca_dim, random_state=42, svd_solver="randomized").fit_transform(embeddings)

    coords = umap.UMAP(
        n_components=2,
        random_state=42,
        init="spectral",
        n_neighbors=min(30, max(2, len(embeddings) - 1)),
        min_dist=0.05,
        metric="euclidean",
        n_jobs=min(20, os.cpu_count() or 20),
    ).fit_transform(embeddings)

    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)
    norm_x = (coords[:, 0] - x_min) / x_span
    norm_y = (coords[:, 1] - y_min) / y_span

    render_count = len(coords)
    cols = max(1, int(math.ceil(math.sqrt(render_count * (x_span / max(y_span, 1e-6))))))
    rows = max(1, int(math.ceil(render_count / cols)))
    cell = max(thumb_size + 4, 8)
    pad = 2
    top_margin = 92
    left_margin = 24
    canvas_w = left_margin * 2 + cols * cell
    canvas_h = top_margin + rows * cell + 24
    canvas = Image.new("RGB", (canvas_w, canvas_h), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        title_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    palette = matplotlib.colormaps.get_cmap("tab10").resampled(len(set(labels)) or 1)
    label_to_index = {label: idx for idx, label in enumerate(sorted(set(labels)))}
    occupied = np.zeros((rows, cols), dtype=bool)

    def find_free_slot(target_r: int, target_c: int) -> tuple[int, int]:
        if not occupied[target_r, target_c]:
            return target_r, target_c
        max_radius = max(rows, cols)
        for radius in range(1, max_radius):
            r0 = max(0, target_r - radius)
            r1 = min(rows - 1, target_r + radius)
            c0 = max(0, target_c - radius)
            c1 = min(cols - 1, target_c + radius)
            for c in range(c0, c1 + 1):
                if not occupied[r0, c]:
                    return r0, c
                if not occupied[r1, c]:
                    return r1, c
            for r in range(r0 + 1, r1):
                if not occupied[r, c0]:
                    return r, c0
                if not occupied[r, c1]:
                    return r, c1
        return target_r, target_c

    for idx in range(len(coords)):
        target_r = int(round(norm_y[idx] * max(rows - 1, 0)))
        target_c = int(round(norm_x[idx] * max(cols - 1, 0)))
        slot_r, slot_c = find_free_slot(target_r, target_c)
        occupied[slot_r, slot_c] = True
        x = left_margin + slot_c * cell + pad
        y = top_margin + slot_r * cell + pad
        thumb = Image.fromarray(thumbnails[idx])
        canvas.paste(thumb, (x, y))
        outline = tuple(int(v * 255) for v in palette(label_to_index[labels[idx]])[:3])
        draw.rectangle([x - 1, y - 1, x + thumb_size + 1, y + thumb_size + 1], outline=outline, width=2)

    draw.text((left_margin, 18), title, fill=(245, 245, 245), font=title_font)
    draw.text(
        (left_margin, 42),
        f"All {len(coords):,} accepted samples are plotted with {thumb_size}x{thumb_size} thumbnails.",
        fill=(210, 210, 210),
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_clip_umap(output_root: Path, accepted_root: Path, clip_model, preprocess, tokenizer, device: torch.device, thumb_size: int) -> None:
    all_paths: list[Path] = []
    all_labels: list[str] = []
    for category in ("paper", "metal"):
        category_dir = accepted_root / category
        if not category_dir.exists():
            continue
        for item_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            for path in image_paths(item_dir):
                all_paths.append(path)
                all_labels.append(category)

    if not all_paths:
        raise RuntimeError("No accepted images found for UMAP generation.")

    batch_size = 384
    embeddings: list[np.ndarray] = []
    thumbnails: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda"
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        for start in range(0, len(all_paths), batch_size):
            batch_paths = all_paths[start : start + batch_size]
            batch_images: list[torch.Tensor] = []
            valid_paths: list[Path] = []
            for path in batch_paths:
                try:
                    with Image.open(path) as image:
                        img = image.convert("RGB")
                        batch_images.append(preprocess(img))
                        valid_paths.append(path)
                        thumb = img.resize((thumb_size, thumb_size), Image.BILINEAR)
                        thumbnails.append(np.array(thumb))
                except Exception:
                    continue
            if not batch_images:
                continue
            tensor = torch.stack(batch_images).to(device=device)
            feats = clip_model.encode_image(tensor).float()
            feats = F.normalize(feats, dim=-1).cpu().numpy()
            embeddings.append(feats)

    emb_np = np.concatenate(embeddings, axis=0).astype(np.float32, copy=False)
    thumb_np = thumbnails[: len(emb_np)]
    labels = all_labels[: len(emb_np)]

    np.savez_compressed(output_root / "clip_embeddings_umap.npz", embeddings=emb_np, labels=np.asarray(labels))
    with (output_root / "clip_embeddings_umap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "label", "path"])
        for idx, (label, path) in enumerate(zip(labels, all_paths[: len(emb_np)], strict=True)):
            writer.writerow([idx, label, str(path)])

    render_umap_thumbnail_map(
        emb_np,
        thumb_np,
        labels,
        output_root / "clip_umap_thumbnail_map.png",
        thumb_size=thumb_size,
        title="CLIP UMAP Thumbnail Map of the Refined Paper/Metal Review Corpus",
    )


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    raw_root = output_root / "raw"
    accepted_root = output_root / "accepted"
    rejected_root = output_root / "rejected"
    manifest_path = output_root / "review_manifest.json"
    scores_path = output_root / "clip_scores.jsonl"

    if args.download_only and args.clip_only:
        raise SystemExit("--download-only and --clip-only are mutually exclusive")

    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    ensure_dir(output_root)
    ensure_dir(raw_root)
    ensure_dir(accepted_root)
    ensure_dir(rejected_root)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    clip_model = None
    preprocess = None
    tokenizer = None
    category_names = None
    category_prompt_bank = None
    photo_prompt_bank = None
    vlm_model = None
    vlm_processor = None
    internvl_tokenizer = None
    if not args.download_only:
        if args.judge_backend == "clip":
            print(f"[clip] loading {args.clip_model} / {args.clip_pretrained} on {device}")
        else:
            print(f"[vlm] loading {args.vlm_model} on {device}")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            start_startup_vram_guard(
                args.startup_vram_limit_mib,
                args.startup_vram_guard_seconds,
                args.startup_vram_poll_seconds,
            )
        if args.judge_backend == "clip":
            clip_model, preprocess, tokenizer = load_clip(device, args.clip_model, args.clip_pretrained)
            category_names, category_prompt_bank = build_category_prompt_bank(clip_model, tokenizer, device)
            photo_prompt_bank = build_photo_prompt_bank(clip_model, tokenizer, device)
        else:
            if "internvl" in args.vlm_model.lower():
                vlm_model, internvl_tokenizer = load_internvl(
                    args.vlm_model,
                    args.vlm_max_gpu_memory_gib,
                    args.vlm_max_cpu_memory_gib,
                )
            elif args.vlm_quantization != "fp16":
                vlm_model, vlm_processor = load_vlm_quantized(
                    args.vlm_model,
                    args.vlm_min_pixels,
                    args.vlm_max_pixels,
                    args.vlm_max_gpu_memory_gib,
                    args.vlm_max_cpu_memory_gib,
                    args.vlm_quantization,
                )
            else:
                vlm_model, vlm_processor = load_vlm(
                    args.vlm_model,
                    args.vlm_min_pixels,
                    args.vlm_max_pixels,
                    args.vlm_max_gpu_memory_gib,
                    args.vlm_max_cpu_memory_gib,
                )

    item_results: list[ItemResult] = []
    scores_handle = scores_path.open("w", encoding="utf-8")
    accepted_index = {"paper": 0, "metal": 0}

    try:
        if args.clip_only:
            mode_name = "CLIP" if args.judge_backend == "clip" else "VLM"
            print(f"[{args.judge_backend}] running {mode_name}-only pass over existing raw files in {raw_root}")
            raw_samples = gather_raw_samples(raw_root)
            if args.judge_backend == "clip":
                keep_by_path, row_by_path = classify_samples_global(
                    clip_model,
                    preprocess,
                    device,
                    raw_samples,
                    category_names,
                    category_prompt_bank,
                    photo_prompt_bank,
                    args.clip_batch_size,
                    args.clip_min_prob,
                    args.clip_min_margin,
                    args.photo_min_prob,
                    args.photo_min_margin,
                )
                staged_results = gather_raw_item_images(raw_root)
                for category, item, _staged_dir, kept_preclean in staged_results:
                    item_accepted_dir = accepted_root / category / slugify(item)
                    item_rejected_dir = rejected_root / category / slugify(item)
                    ensure_dir(item_accepted_dir)
                    ensure_dir(item_rejected_dir)
                    accepted_count = 0
                    rejected_count = 0
                    for path in kept_preclean:
                        row = row_by_path.get(path)
                        if row is not None:
                            scores_handle.write(json.dumps({"category": category, "item": item, **row}) + "\n")
                        if path in kept_paths:
                            accepted_index[category] += 1
                            prefix = "METAL" if category == "metal" else "PAPER"
                            dst_name = (
                                f"{prefix}_{slugify(item)}_{accepted_index[category]:05d}{path.suffix.lower() or '.jpg'}"
                                if category == "metal"
                                else f"{slugify(item)}_{accepted_index[category]:05d}{path.suffix.lower() or '.jpg'}"
                            )
                            shutil.copy2(path, item_accepted_dir / dst_name)
                            accepted_count += 1
                        else:
                            shutil.copy2(path, item_rejected_dir / path.name)
                            rejected_count += 1
                    item_results.append(ItemResult(category, item, len(kept_preclean), accepted_count, rejected_count))
                    print(f"[{args.judge_backend}] {category}/{item}: raw={len(kept_preclean)} kept={accepted_count} rejected={rejected_count}")
            else:
                staged_results = gather_raw_item_images(raw_root)
                for category, item, _staged_dir, kept_preclean in staged_results:
                    accepted_count = 0
                    rejected_count = 0
                    for start in range(0, len(kept_preclean), args.vlm_batch_size):
                        batch_paths = kept_preclean[start : start + args.vlm_batch_size]
                        batch_samples = [RawSample(category=category, item=item, path=path) for path in batch_paths]
                        if not batch_samples:
                            continue
                        if internvl_tokenizer is not None:
                            keep_by_path, row_by_path = classify_samples_internvl(
                                vlm_model,
                                internvl_tokenizer,
                                batch_samples,
                                args.vlm_batch_size,
                                args.internvl_input_size,
                                args.internvl_max_num,
                            )
                        else:
                            keep_by_path, row_by_path = classify_samples_vlm(
                                vlm_model,
                                vlm_processor,
                                batch_samples,
                                args.vlm_batch_size,
                                args.vlm_min_confidence,
                                args.vlm_min_photo_confidence,
                            )
                        batch_accepted, batch_rejected = persist_judged_samples(
                            category,
                            item,
                            batch_samples,
                            keep_by_path,
                            row_by_path,
                            accepted_root,
                            rejected_root,
                            scores_handle,
                            accepted_index,
                        )
                        accepted_count += batch_accepted
                        rejected_count += batch_rejected
                        print(
                            f"[{args.judge_backend}] {category}/{item} batch={start // args.vlm_batch_size + 1} "
                            f"size={len(batch_samples)} kept={batch_accepted} rejected={batch_rejected}"
                        )
                    item_results.append(ItemResult(category, item, len(kept_preclean), accepted_count, rejected_count))
                    print(f"[{args.judge_backend}] {category}/{item}: raw={len(kept_preclean)} kept={accepted_count} rejected={rejected_count}")
        else:
            download_jobs: list[tuple[str, str]] = [(category, item) for category, items in CATEGORY_ITEMS.items() for item in items]
            max_workers = min(args.item_workers, len(download_jobs))
            print(f"[download] running {len(download_jobs)} item jobs with {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(
                        download_and_clean_item,
                        category,
                        item,
                        raw_root,
                        args.limit_per_query,
                        args.min_resolution,
                        args.download_buffer,
                        args.download_workers,
                        args.page_workers,
                        args.max_query_pages,
                        args.idle_page_rounds,
                    )
                    for category, item in download_jobs
                ]
                for future in as_completed(futures):
                    category, item, staged_dir, kept_preclean = future.result()
                    if staged_dir is None:
                        item_results.append(ItemResult(category, item, 0, 0, 0))
                        continue
                    if args.download_only:
                        item_results.append(ItemResult(category, item, len(kept_preclean), 0, 0))
                        print(f"[download] {category}/{item}: raw={len(kept_preclean)}")
                        continue
                    if args.judge_backend == "clip":
                        kept_paths, rows = classify_images(
                            clip_model,
                            preprocess,
                            device,
                            kept_preclean,
                            category,
                            item,
                            category_names,
                            category_prompt_bank,
                            photo_prompt_bank,
                            args.clip_batch_size,
                            args.clip_min_prob,
                            args.clip_min_margin,
                            args.photo_min_prob,
                            args.photo_min_margin,
                        )
                        row_by_path = {Path(row["path"]): row for row in rows}
                    else:
                        raw_samples = [RawSample(category=category, item=item, path=path) for path in kept_preclean]
                        if internvl_tokenizer is not None:
                            keep_by_path, row_by_path = classify_samples_internvl(
                                vlm_model,
                                internvl_tokenizer,
                                raw_samples,
                                args.vlm_batch_size,
                                args.internvl_input_size,
                                args.internvl_max_num,
                            )
                        else:
                            keep_by_path, row_by_path = classify_samples_vlm(
                                vlm_model,
                                vlm_processor,
                                raw_samples,
                                args.vlm_batch_size,
                                args.vlm_min_confidence,
                                args.vlm_min_photo_confidence,
                            )
                        kept_paths = [path for path in kept_preclean if keep_by_path.get(path, False)]
                    item_accepted_dir = accepted_root / category / slugify(item)
                    item_rejected_dir = rejected_root / category / slugify(item)
                    ensure_dir(item_accepted_dir)
                    ensure_dir(item_rejected_dir)
                    kept_set = {path.resolve() for path in kept_paths}
                    accepted_count = 0
                    rejected_count = 0
                    for path in kept_preclean:
                        row = row_by_path.get(path)
                        if row is not None:
                            scores_handle.write(json.dumps({"category": category, "item": item, **row}) + "\n")
                        if path.resolve() in kept_set:
                            accepted_index[category] += 1
                            prefix = "METAL" if category == "metal" else "PAPER"
                            dst_name = (
                                f"{prefix}_{slugify(item)}_{accepted_index[category]:05d}{path.suffix.lower() or '.jpg'}"
                                if category == "metal"
                                else f"{slugify(item)}_{accepted_index[category]:05d}{path.suffix.lower() or '.jpg'}"
                            )
                            shutil.copy2(path, item_accepted_dir / dst_name)
                            accepted_count += 1
                        else:
                            shutil.copy2(path, item_rejected_dir / path.name)
                            path.unlink(missing_ok=True)
                            rejected_count += 1
                    shutil.rmtree(staged_dir, ignore_errors=True)
                    item_results.append(ItemResult(category, item, len(kept_preclean), accepted_count, rejected_count))
                    print(f"[{args.judge_backend}] {category}/{item}: raw={len(kept_preclean)} kept={accepted_count} rejected={rejected_count}")
    finally:
        scores_handle.close()

    accepted_images = image_paths(accepted_root / "paper") + image_paths(accepted_root / "metal")
    manifest = {
        "output_root": str(output_root),
        "raw_root": str(raw_root),
        "accepted_root": str(accepted_root),
        "rejected_root": str(rejected_root),
        "download_only": args.download_only,
        "clip_only": args.clip_only,
        "judge_backend": args.judge_backend,
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "vlm_model": args.vlm_model,
        "vlm_quantization": args.vlm_quantization,
        "filter_mode": "superclass_real_photo",
        "limit_per_query": args.limit_per_query,
        "download_buffer": args.download_buffer,
        "min_resolution": args.min_resolution,
        "clip_min_prob": args.clip_min_prob,
        "clip_min_margin": args.clip_min_margin,
        "vlm_min_confidence": args.vlm_min_confidence,
        "vlm_min_photo_confidence": args.vlm_min_photo_confidence,
        "vlm_batch_size": args.vlm_batch_size,
        "num_items": sum(len(items) for items in CATEGORY_ITEMS.values()),
        "raw_count": sum(result.raw_count for result in item_results),
        "kept_count": sum(result.kept_count for result in item_results),
        "rejected_count": sum(result.rejected_count for result in item_results),
        "accepted_image_count": len(accepted_images),
        "item_results": [result.__dict__ for result in item_results],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[clip] wrote {manifest_path}")

    if args.download_only:
        print("[download] finished raw ingestion only; skipping CLIP and UMAP")
        return 0

    if args.judge_backend == "clip":
        if accepted_images:
            print(f"[umap] building CLIP UMAP for {len(accepted_images):,} accepted images")
            build_clip_umap(output_root, accepted_root, clip_model, preprocess, tokenizer, device, args.umap_thumb_size)
            print(f"[umap] wrote {output_root / 'clip_umap_thumbnail_map.png'}")
        else:
            print("[umap] skipped; no accepted images")
    else:
        print("[umap] skipped; VLM judge mode does not build CLIP embeddings")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
