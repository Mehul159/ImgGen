"""
Phase 3 — Data Preprocessing.

1. Load subject images and auto-caption with BLIP-2 (trigger token prepended).
2. Resize to 1024×1024 and save processed images + metadata.jsonl.
3. Extract ControlNet conditioning maps: OpenPose, MiDaS depth, Canny edges.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from PIL import Image
from datasets import load_from_disk
from configs.default import SUBJECT_DIR, PROCESSED_DIR, TRIGGER_TOKEN


def autocaption_images():
    """Load BLIP-2 and generate captions for every subject image."""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BLIP-2 (Salesforce/blip2-opt-2.7b)…")
    blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading subject dataset from {SUBJECT_DIR}…")
    ds = load_from_disk(str(SUBJECT_DIR))

    metadata = []
    for i, sample in enumerate(ds):
        img: Image.Image = sample["image"].convert("RGB")
        img = img.resize((1024, 1024), Image.LANCZOS)

        inputs = blip_proc(images=img, return_tensors="pt").to("cuda", torch.float16)
        out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_proc.decode(out[0], skip_special_tokens=True)
        caption = f"a photo of {TRIGGER_TOKEN}, {caption}"

        img_path = PROCESSED_DIR / f"img_{i:04d}.jpg"
        img.save(str(img_path), quality=95)
        metadata.append({"file_name": str(img_path), "text": caption})
        print(f"  [{i+1}/{len(ds)}] {caption[:80]}…")

    meta_path = PROCESSED_DIR / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    print(f"Saved {len(metadata)} captioned images to {PROCESSED_DIR}")
    print(f"Metadata written to {meta_path}")
    return metadata


def extract_controlnet_maps():
    """Extract pose, depth, and canny conditioning maps for each image."""
    from controlnet_aux import OpenposeDetector, MidasDetector, CannyDetector

    print("\nExtracting ControlNet conditioning maps…")
    ds = load_from_disk(str(SUBJECT_DIR))

    pose_det = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    depth_det = MidasDetector.from_pretrained("lllyasviel/ControlNet")
    canny_det = CannyDetector()

    for i, sample in enumerate(ds):
        img = sample["image"].convert("RGB").resize((1024, 1024))

        pose_path = PROCESSED_DIR / f"pose_{i:04d}.png"
        depth_path = PROCESSED_DIR / f"depth_{i:04d}.png"
        canny_path = PROCESSED_DIR / f"canny_{i:04d}.png"

        try:
            pose_det(img).save(str(pose_path))
        except Exception as e:
            print(f"  [warn] Pose extraction failed for image {i}: {e}")

        try:
            depth_det(img).save(str(depth_path))
        except Exception as e:
            print(f"  [warn] Depth extraction failed for image {i}: {e}")

        try:
            canny_det(img, low_threshold=100, high_threshold=200).save(str(canny_path))
        except Exception as e:
            print(f"  [warn] Canny extraction failed for image {i}: {e}")

        print(f"  [{i+1}/{len(ds)}] Conditioning maps saved")

    print("ControlNet map extraction complete.")


if __name__ == "__main__":
    autocaption_images()
    extract_controlnet_maps()
    print("\nPreprocessing complete.")
