#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"

DEFAULT_TEXT_MODEL = "deepseek-r1:8b"

DEFAULT_VISION_MODELS = (
    "qwen2.5vl:3b",
)

DEFAULT_CLASS_SPEC: dict[str, dict[str, list[str] | str]] = {
    "organic": {
        "description": "Biodegradable real-world organic waste and natural matter.",
        "seed_objects": [
            "food scraps",
            "fruit peels",
            "vegetable scraps",
            "leaves",
            "grass clippings",
            "coffee grounds",
            "eggshells",
            "garden waste",
            "compost",
            "leftover food",
        ],
    },
    "metal": {
        "description": "Real-world metallic objects and waste items.",
        "seed_objects": [
            "cans",
            "keys",
            "coins",
            "screws",
            "bolts",
            "nuts",
            "tools",
            "utensils",
            "foil",
            "trays",
        ],
    },
    "paper": {
        "description": "Real-world paper and cardboard objects and waste items.",
        "seed_objects": [
            "envelopes",
            "flyers",
            "brochures",
            "receipts",
            "documents",
            "cardboard",
            "notepads",
            "paper bags",
            "books",
            "newspapers",
        ],
    },
}


@dataclass(frozen=True)
class ThresholdProfile:
    class_accept: float
    class_reject: float
    photo_accept: float
    photo_reject: float
    noisy_bonus: float = 0.0


CATEGORY_THRESHOLD_PROFILES: dict[str, ThresholdProfile] = {
    "organic": ThresholdProfile(
        class_accept=0.80,
        class_reject=0.45,
        photo_accept=0.82,
        photo_reject=0.48,
        noisy_bonus=0.03,
    ),
    "paper": ThresholdProfile(
        class_accept=0.82,
        class_reject=0.45,
        photo_accept=0.84,
        photo_reject=0.50,
        noisy_bonus=0.05,
    ),
    "metal": ThresholdProfile(
        class_accept=0.80,
        class_reject=0.45,
        photo_accept=0.82,
        photo_reject=0.48,
        noisy_bonus=0.03,
    ),
}


NOISY_SUBCLASSES = frozenset(
    {
        "flyers",
        "brochures",
        "posters",
        "cards",
        "greeting cards",
        "index cards",
        "booklets",
        "manuals",
        "charts",
        "printouts",
        "labels",
        "documents",
        "receipts",
        "shopping receipts",
        "postcards",
        "wrapping paper",
        "paper packaging",
        "food scraps",
        "fruit peels",
        "vegetable scraps",
        "garden waste",
        "leftover food",
        "compost",
    }
)


TEXT_CLASS_DISCOVERY_SYSTEM_PROMPT = (
    "You are a dataset class expansion planner. "
    "Given target material classes and optional seed objects, return the most common real-world physical object types "
    "that should be searched for building a clean image dataset. "
    "Return strict JSON only."
)


TEXT_CLASS_DISCOVERY_USER_PROMPT = """
You are helping build an image dataset for material classification.

The user provides target classes. For each target class:
- infer the most common real-world physical objects that belong to that class
- keep the list focused on common, concrete, photographable objects
- avoid abstract concepts, scenes, non-physical media, logos, screenshots, and synthetic-only artifacts
- include the provided seed objects as long as they are plausible
- do not include objects that are too ambiguous across material classes unless the material cue is usually obvious

Return JSON with exactly this shape:
{{
  "classes": {{
    "<class_name>": {{
      "objects": ["object 1", "object 2", "..."],
      "rationale": "short sentence"
    }}
  }}
}}

Target class specification:
{class_spec}
""".strip()


VLM_SYSTEM_PROMPT = (
    "You are a strict image-dataset curator. "
    "You must classify real-world images conservatively for supervised training. "
    "Return JSON only."
)


CLASS_STAGE_PROMPT = """
Target superclass: {category}
Target object subclass: {item}
Allowed classes: {allowed_classes}

Task:
- Decide if this image is best described as the target superclass or not.
- The image should represent a real-world physical object that belongs to the target superclass.
- For organic: real photographed biodegradable natural matter, food waste, plant waste, compost, leaves, peels, and leftovers are valid.
- For paper: real photographed paper/cardboard objects are valid even if they contain printed text, logos, packaging art, or graphics.
- For metal: real photographed metal objects are valid even if reflective, cluttered, or worn.
- Reject screenshots, infographics, vector graphics, icons, renders, CGI, AI-generated images, synthetic compositions, or category-mismatched images.

Return JSON:
{{
  "decision": "<one of: {allowed_classes_csv}, reject, uncertain>",
  "confidence": 0.0,
  "reason": "short reason"
}}

Confidence must be between 0.0 and 1.0.
""".strip()


PHOTO_STAGE_PROMPT = """
Target superclass: {category}
Target object subclass: {item}

Task:
- You are judging whether this image is suitable training data for a pristine, clean, learnable real-world material dataset.
- The goal is to keep only images that will help a classifier learn robust, generalizable visual features for the target superclass.
- First decide whether this is a genuine camera photo of a physical real-world object, not an infographic, screenshot, digital design, render, CGI image, AI-generated image, mockup, clipart, or diagram.
- Then judge whether the image is clean and train-worthy:
  - the target object/material should be visually dominant or clearly central
  - the image should not be dominated by humans, hands, faces, bodies, pets, or unrelated foreground subjects
  - the image should not contain many other salient objects that would confuse the label
  - the image should not be a novelty, sculpture, artwork, collage, costume, or abnormal construction made out of the target material
  - the image should not be extremely blurry, badly cropped, heavily occluded, excessively noisy, low-information, or otherwise poor supervision
  - the image may contain normal background clutter, printed text, packaging art, reflections, labels, or perspective distortion if the target object remains clearly learnable
- Be conservative. If the image is borderline, mixed, confusing, or likely to hurt training quality, return `uncertain` instead of `photo`.
- Use `photo` only when the image is both:
  - a real-world camera photo
  - clean, representative, and learnable training data for the target superclass
- Use `synthetic` when the image is synthetic, infographic-like, badly noisy, strongly cluttered, dominated by other subjects, abnormal, or otherwise poor training data.

Return JSON:
{{
  "decision": "<one of: photo, synthetic, uncertain>",
  "confidence": 0.0,
  "reason": "short reason",
  "is_real_photo": true,
  "target_dominant": true,
  "has_humans": false,
  "has_major_clutter": false,
  "has_multiple_salient_objects": false,
  "is_infographic_or_render": false,
  "is_abnormal_artistic_case": false,
  "is_visually_clean": true,
  "is_trainworthy": true
}}

Confidence must be between 0.0 and 1.0.
""".strip()


@dataclass(frozen=True)
class DownloaderProfile:
    direct_bing_limit: int = 1250
    google_limit: int = 1250
    bing_limit: int = 1250
    baidu_limit: int = 1250
    min_resolution: int = 224
    item_workers: int = 8
    engine_workers: int = 4


DEFAULT_DOWNLOADER_PROFILE = DownloaderProfile()


@dataclass(frozen=True)
class OllamaGenerationProfile:
    temperature: float = 0.1
    top_p: float = 0.9
    num_predict: int = 256
    keep_alive: str = "30m"


DEFAULT_TEXT_GENERATION_PROFILE = OllamaGenerationProfile(
    temperature=0.2,
    top_p=0.95,
    num_predict=512,
    keep_alive="30m",
)


DEFAULT_VISION_GENERATION_PROFILE = OllamaGenerationProfile(
    temperature=0.0,
    top_p=0.9,
    num_predict=128,
    keep_alive="30m",
)
