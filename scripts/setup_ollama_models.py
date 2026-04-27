#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess

from ollama_pipeline_defaults import DEFAULT_TEXT_MODEL, DEFAULT_VISION_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull the default Ollama text and vision models used by the repo.")
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--vision-models", nargs="+", default=list(DEFAULT_VISION_MODELS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = [args.text_model] + list(args.vision_models)
    for model in models:
        print(f"[ollama] pulling {model}")
        subprocess.run(["ollama", "pull", model], check=True)
    print("[ollama] all required models are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
