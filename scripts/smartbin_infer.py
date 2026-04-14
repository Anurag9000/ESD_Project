#!/usr/bin/env python3
"""
SmartBin Edge Inference — Raspberry Pi 4 deployment script.
Uses the ONNX model (best.onnx + best.onnx.data) for zero-accuracy-loss inference.

Requirements (install once on RPi):
    pip install onnxruntime pillow numpy

Usage:
    # Classify a single image:
    python3 smartbin_infer.py --image /path/to/image.jpg

    # Run on a folder of images:
    python3 smartbin_infer.py --folder /path/to/images/

    # Run on USB camera (continuous):
    python3 smartbin_infer.py --camera 0
"""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# ── Constants (must match training) ──────────────────────────────────────────
IMAGE_SIZE  = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = [
    'clothes', 'ewaste', 'glass',
    'hard_plastic', 'metal', 'organic', 'paper', 'soft_plastic'
]


def preprocess(image: Image.Image) -> np.ndarray:
    """Centre-crop + normalize exactly as the training eval pipeline does."""
    # Resize shortest edge to IMAGE_SIZE
    w, h = image.size
    if w < h:
        new_w, new_h = IMAGE_SIZE, int(h * IMAGE_SIZE / w)
    else:
        new_w, new_h = int(w * IMAGE_SIZE / h), IMAGE_SIZE
    image = image.resize((new_w, new_h), Image.BILINEAR)

    # Centre crop
    left  = (new_w - IMAGE_SIZE) // 2
    top   = (new_h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # To float32 numpy, HWC → CHW, normalize
    arr = np.array(image, dtype=np.float32) / 255.0   # [H, W, 3]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD         # normalize
    arr = arr.transpose(2, 0, 1)                        # [3, H, W]
    return arr[np.newaxis]                              # [1, 3, H, W]


def load_session(model_path: str) -> ort.InferenceSession:
    """Load the ONNX model. best.onnx.data must be in the same directory."""
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4   # RPi4 has 4 cores
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=['CPUExecutionProvider']
    )


def classify(session: ort.InferenceSession, image: Image.Image) -> tuple[str, float, float]:
    """
    Returns: (class_name, confidence_pct, inference_ms)
    """
    tensor = preprocess(image.convert('RGB'))
    t0 = time.perf_counter()
    logits = session.run(['logits'], {'image': tensor})[0][0]   # shape [8]
    ms = (time.perf_counter() - t0) * 1000

    # Softmax
    exp_logits = np.exp(logits - logits.max())
    probs = exp_logits / exp_logits.sum()

    pred_idx  = int(probs.argmax())
    confidence = float(probs[pred_idx]) * 100
    return CLASS_NAMES[pred_idx], confidence, ms


def main():
    p = argparse.ArgumentParser(description='SmartBin Edge Inference')
    p.add_argument('--model',   default='best.onnx',
                   help='Path to best.onnx  (best.onnx.data must be in same dir)')
    p.add_argument('--image',   help='Path to a single image file')
    p.add_argument('--folder',  help='Path to a folder of images')
    p.add_argument('--camera',  type=int, default=None,
                   help='Camera index for live webcam inference (e.g. 0)')
    args = p.parse_args()

    print(f'Loading model: {args.model}')
    session = load_session(args.model)
    print('Model loaded ✅\n')

    # ── Single image ─────────────────────────────────────────────────────────
    if args.image:
        img = Image.open(args.image)
        label, conf, ms = classify(session, img)
        print(f'  File  : {args.image}')
        print(f'  Class : {label}')
        print(f'  Conf  : {conf:.1f}%')
        print(f'  Time  : {ms:.0f} ms')

    # ── Folder of images ─────────────────────────────────────────────────────
    elif args.folder:
        exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
        paths = [p for p in Path(args.folder).iterdir()
                 if p.suffix.lower() in exts]
        print(f'Found {len(paths)} images in {args.folder}\n')
        times = []
        for path in sorted(paths):
            img = Image.open(path)
            label, conf, ms = classify(session, img)
            times.append(ms)
            print(f'  {path.name:<40} → {label:<12} ({conf:.1f}%)  [{ms:.0f}ms]')
        if times:
            print(f'\nAvg inference: {np.mean(times):.0f} ms | '
                  f'Min: {np.min(times):.0f} ms | Max: {np.max(times):.0f} ms')

    # ── Live camera ──────────────────────────────────────────────────────────
    elif args.camera is not None:
        try:
            import cv2
        except ImportError:
            print('ERROR: Install opencv  →  pip install opencv-python-headless')
            return
        cap = cv2.VideoCapture(args.camera)
        print(f'Streaming camera {args.camera}. Press Ctrl+C to stop.\n')
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                label, conf, ms = classify(session, img)
                print(f'\r  {label:<12} ({conf:.1f}%)  [{ms:.0f}ms]   ', end='', flush=True)
        except KeyboardInterrupt:
            print('\nStopped.')
        finally:
            cap.release()

    else:
        p.print_help()


if __name__ == '__main__':
    main()
