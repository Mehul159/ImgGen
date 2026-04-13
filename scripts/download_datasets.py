"""
Phase 1 — Download datasets from Hugging Face.

Datasets:
  - diffusers/dog-example               → subject images for DreamBooth
  - multimodalart/faces-prior-preservation → regularisation images
  - Artificio/WikiArt (stream, first 1k) → style LoRA dataset
  - poloclub/diffusiondb (2m_first_1k)  → ControlNet conditioning pairs

Video clips (OpenVid-1M) are skipped to stay within Kaggle's 57 GB disk.
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from datasets import load_dataset, load_from_disk
from configs.default import (
    SUBJECT_DIR, REG_DIR, STYLE_DIR, CONTROLNET_DIR,
)


def download_subject():
    if SUBJECT_DIR.exists() and any(SUBJECT_DIR.iterdir()):
        print(f"[skip] Subject dataset already at {SUBJECT_DIR}")
        return
    print("[1/4] Downloading subject dataset (diffusers/dog-example)…")
    ds = load_dataset("diffusers/dog-example", split="train")
    ds.save_to_disk(str(SUBJECT_DIR))
    print(f"  → Saved {len(ds)} samples to {SUBJECT_DIR}")


def download_regularisation():
    if REG_DIR.exists() and any(REG_DIR.iterdir()):
        print(f"[skip] Regularisation dataset already at {REG_DIR}")
        return
    print("[2/4] Downloading regularisation faces…")
    ds = load_dataset("multimodalart/faces-prior-preservation", split="train")
    ds.save_to_disk(str(REG_DIR))
    print(f"  → Saved {len(ds)} samples to {REG_DIR}")


def download_style():
    if STYLE_DIR.exists() and any(STYLE_DIR.iterdir()):
        print(f"[skip] Style dataset already at {STYLE_DIR}")
        return
    print("[3/4] Downloading style dataset (WikiArt, first 1000)…")
    stream = load_dataset("Artificio/WikiArt", split="train", streaming=True)
    subset = list(stream.take(1000))

    from datasets import Dataset, Features, Value, Image as HFImage
    features = Features({"image": HFImage(), "text": Value("string")})
    rows = []
    for s in subset:
        img = s.get("image")
        caption = s.get("artist", "unknown")
        if img is not None:
            rows.append({"image": img, "text": caption})

    ds = Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(STYLE_DIR))
    print(f"  → Saved {len(ds)} samples to {STYLE_DIR}")


def download_controlnet_pairs():
    if CONTROLNET_DIR.exists() and any(CONTROLNET_DIR.iterdir()):
        print(f"[skip] ControlNet dataset already at {CONTROLNET_DIR}")
        return
    print("[4/4] Downloading ControlNet conditioning pairs (LAION-Art)…")
    stream = load_dataset("laion/laion-art", split="train", streaming=True)
    subset = list(stream.take(1000))

    from datasets import Dataset, Features, Value, Image as HFImage
    features = Features({"url": Value("string"), "text": Value("string")})
    rows = [{"url": s.get("URL", ""), "text": s.get("TEXT", "")} for s in subset if s.get("URL")]
    ds = Dataset.from_list(rows, features=features)
    ds.save_to_disk(str(CONTROLNET_DIR))
    print(f"  → Saved {len(ds)} samples to {CONTROLNET_DIR}")


def verify():
    all_ok = True
    for name, path in [
        ("subject_images", SUBJECT_DIR),
        ("reg_images", REG_DIR),
        ("style_images", STYLE_DIR),
        ("controlnet_pairs", CONTROLNET_DIR),
    ]:
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    download_subject()
    download_regularisation()
    download_style()
    download_controlnet_pairs()

    print("\n── Verification ──")
    if verify():
        print("\nAll datasets downloaded successfully.")
    else:
        print("\nSome datasets are missing. Re-run or check errors above.")
