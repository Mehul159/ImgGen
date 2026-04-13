"""
Phase 2 — Verify / pre-warm model cache.

Models are loaded from HuggingFace Hub on demand (no bulk download).
This script optionally pre-warms the HF cache so later phases don't
wait for downloads. Only SDXL is used — FLUX and CogVideoX are too
large for Kaggle/Colab disk limits.

Estimated disk usage:
  SDXL base (fp16 safetensors) : ~6.5 GB (in HF cache)
  SDXL VAE fix                 : ~150 MB
  Total                        : ~7 GB
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.default import (
    SDXL_PATH, SDXL_VAE_PATH,
    SDXL_HUB_ID, SDXL_VAE_HUB_ID,
    resolve_model,
)


def verify():
    print("── Model Status ──")
    sdxl_src = resolve_model(SDXL_PATH, SDXL_HUB_ID)
    vae_src = resolve_model(SDXL_VAE_PATH, SDXL_VAE_HUB_ID)
    is_local_sdxl = sdxl_src == str(SDXL_PATH)
    is_local_vae = vae_src == str(SDXL_VAE_PATH)
    print(f"  SDXL base  : {'LOCAL ' + str(SDXL_PATH) if is_local_sdxl else 'HUB ' + SDXL_HUB_ID}")
    print(f"  SDXL VAE   : {'LOCAL ' + str(SDXL_VAE_PATH) if is_local_vae else 'HUB ' + SDXL_VAE_HUB_ID}")
    print()
    print("Models will be downloaded from HuggingFace Hub on first use.")
    print("No bulk download needed — this saves ~100 GB of disk space.")


def prewarm():
    """Pre-download SDXL into HF cache (optional, saves time in later phases)."""
    from huggingface_hub import snapshot_download
    print("Pre-warming SDXL in HF cache (fp16 safetensors only)…")
    snapshot_download(
        repo_id=SDXL_HUB_ID,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
        ignore_patterns=["*.bin", "*.ckpt", "*.msgpack"],
    )
    print("  SDXL cached.")
    snapshot_download(
        repo_id=SDXL_VAE_HUB_ID,
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
    )
    print("  SDXL VAE cached.")
    print("\nPre-warm complete. Models ready for instant loading.")


if __name__ == "__main__":
    if "--prewarm" in sys.argv:
        prewarm()
    else:
        verify()
