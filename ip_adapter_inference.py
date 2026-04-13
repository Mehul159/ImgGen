"""
Phase 7 — IP-Adapter Integration.

Uses CLIP image embeddings injected into cross-attention for
image-guided generation. Combines with subject LoRA for identity
preservation.

Downloads IP-Adapter SDXL weights from h94/IP-Adapter on first run.
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from configs.default import (
    SDXL_PATH, SDXL_HUB_ID, resolve_model,
    SUBJECT_LORA_PATH, PROCESSED_DIR, OUTPUT_DIR,
    MODEL_DIR, TRIGGER_TOKEN, ENABLE_CPU_OFFLOAD, DEFAULT_STEPS, DEFAULT_SEED,
)

IP_ADAPTER_DIR = MODEL_DIR / "ip-adapter"


def download_ip_adapter():
    """Download IP-Adapter SDXL weights if not present."""
    weight_path = IP_ADAPTER_DIR / "sdxl_models" / "ip-adapter_sdxl.bin"
    if weight_path.exists():
        print("[skip] IP-Adapter weights already downloaded")
        return

    print("Downloading IP-Adapter SDXL weights…")
    hf_hub_download(
        repo_id="h94/IP-Adapter",
        filename="sdxl_models/ip-adapter_sdxl.bin",
        local_dir=str(IP_ADAPTER_DIR),
    )
    hf_hub_download(
        repo_id="h94/IP-Adapter",
        filename="models/image_encoder/config.json",
        local_dir=str(IP_ADAPTER_DIR),
    )
    print("  IP-Adapter weights downloaded.")


def run_inference():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_ip_adapter()

    print("Loading SDXL pipeline…")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        resolve_model(SDXL_PATH, SDXL_HUB_ID), torch_dtype=torch.float16,
    )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    print("Loading IP-Adapter…")
    pipe.load_ip_adapter(
        str(IP_ADAPTER_DIR),
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",
    )
    pipe.set_ip_adapter_scale(0.7)

    if SUBJECT_LORA_PATH.exists():
        pipe.load_lora_weights(str(SUBJECT_LORA_PATH), adapter_name="subject")
        pipe.set_adapters(["subject"], adapter_weights=[1.0])
        print("  Subject LoRA loaded")

    ref_path = PROCESSED_DIR / "img_0000.jpg"
    if ref_path.exists():
        ref_image = load_image(str(ref_path))
    else:
        print(f"  [warn] Reference image not found at {ref_path}, using blank")
        ref_image = Image.new("RGB", (1024, 1024), (128, 128, 128))

    prompt = f"a photo of {TRIGGER_TOKEN} in a futuristic city, cinematic"
    print(f"Generating: {prompt}")

    result = pipe(
        prompt=prompt,
        ip_adapter_image=ref_image,
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=6.0,
        generator=torch.manual_seed(DEFAULT_SEED),
    ).images[0]

    out_path = OUTPUT_DIR / "ip_adapter_result.png"
    result.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_inference()
