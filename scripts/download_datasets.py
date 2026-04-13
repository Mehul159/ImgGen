"""
Phase 1 — Download all datasets from Hugging Face.

Datasets:
  - diffusers/dog-example               → subject images for DreamBooth
  - multimodalart/faces-prior-preservation → regularisation images
  - Artificio/WikiArt (stream, first 5k) → style LoRA dataset
  - HighCWu/diffusion-db-2m-first-1k    → ControlNet conditioning pairs
  - nkp37/OpenVid-1M (first shard)      → video clips for motion LoRA
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from datasets import load_dataset, load_from_disk
from huggingface_hub import snapshot_download
from configs.default import (
    SUBJECT_DIR, REG_DIR, STYLE_DIR, CONTROLNET_DIR, VIDEO_DIR,
)


def download_subject():
    if SUBJECT_DIR.exists() and any(SUBJECT_DIR.iterdir()):
        print(f"[skip] Subject dataset already at {SUBJECT_DIR}")
        return
    print("[1/5] Downloading subject dataset (diffusers/dog-example)…")
    ds = load_dataset("diffusers/dog-example", split="train")
    ds.save_to_disk(str(SUBJECT_DIR))
    print(f"  → Saved {len(ds)} samples to {SUBJECT_DIR}")


def download_regularisation():
    if REG_DIR.exists() and any(REG_DIR.iterdir()):
        print(f"[skip] Regularisation dataset already at {REG_DIR}")
        return
    print("[2/5] Downloading regularisation faces…")
    ds = load_dataset("multimodalart/faces-prior-preservation", split="train")
    ds.save_to_disk(str(REG_DIR))
    print(f"  → Saved {len(ds)} samples to {REG_DIR}")


def download_style():
    if STYLE_DIR.exists() and any(STYLE_DIR.iterdir()):
        print(f"[skip] Style dataset already at {STYLE_DIR}")
        return
    print("[3/5] Downloading style dataset (WikiArt, first 5000)…")
    stream = load_dataset("Artificio/WikiArt", split="train", streaming=True)
    subset = list(stream.take(5000))

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
    print("[4/5] Downloading ControlNet conditioning pairs…")
    ds = load_dataset("poloclub/diffusiondb", "2m_first_1k", split="train")
    ds.save_to_disk(str(CONTROLNET_DIR))
    print(f"  → Saved {len(ds)} samples to {CONTROLNET_DIR}")


def download_video_clips():
    if VIDEO_DIR.exists() and any(VIDEO_DIR.iterdir()):
        print(f"[skip] Video dataset already at {VIDEO_DIR}")
        return
    print("[5/5] Downloading video clips (OpenVid-1M first shard)…")
    snapshot_download(
        repo_id="nkp37/OpenVid-1M",
        repo_type="dataset",
        local_dir=str(VIDEO_DIR),
        ignore_patterns=["*.parquet"],
        allow_patterns=["data/train-00000*", "README.md"],
    )
    print(f"  → Saved to {VIDEO_DIR}")


def verify():
    all_ok = True
    for name, path in [
        ("subject_images", SUBJECT_DIR),
        ("reg_images", REG_DIR),
        ("style_images", STYLE_DIR),
        ("controlnet_pairs", CONTROLNET_DIR),
        ("video_clips", VIDEO_DIR),
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
    download_video_clips()

    print("\n── Verification ──")
    if verify():
        print("\nAll datasets downloaded successfully.")
    else:
        print("\nSome datasets are missing. Re-run or check errors above.")
