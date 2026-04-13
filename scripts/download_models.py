"""
Phase 2 — Download base models from Hugging Face.

Models:
  - black-forest-labs/FLUX.1-dev           → primary DiT model (requires license)
  - stabilityai/stable-diffusion-xl-base-1.0 → SDXL fallback
  - madebyollin/sdxl-vae-fp16-fix         → SDXL VAE (prevents washed colours)
  - THUDM/CogVideoX-5b                    → video generation
"""

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from huggingface_hub import snapshot_download
from configs.default import FLUX_PATH, SDXL_PATH, SDXL_VAE_PATH, COGVIDEO_PATH


def download_flux():
    if FLUX_PATH.exists() and any(FLUX_PATH.iterdir()):
        print(f"[skip] FLUX.1-dev already at {FLUX_PATH}")
        return
    print("[1/4] Downloading FLUX.1-dev (requires license acceptance)…")
    print("       Accept license at: https://huggingface.co/black-forest-labs/FLUX.1-dev")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_dir=str(FLUX_PATH),
        ignore_patterns=["*.gguf"],
    )
    print(f"  → Saved to {FLUX_PATH}")


def download_sdxl():
    if SDXL_PATH.exists() and any(SDXL_PATH.iterdir()):
        print(f"[skip] SDXL 1.0 already at {SDXL_PATH}")
        return
    print("[2/4] Downloading SDXL 1.0 base…")
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir=str(SDXL_PATH),
    )
    print(f"  → Saved to {SDXL_PATH}")


def download_sdxl_vae():
    if SDXL_VAE_PATH.exists() and any(SDXL_VAE_PATH.iterdir()):
        print(f"[skip] SDXL VAE fix already at {SDXL_VAE_PATH}")
        return
    print("[3/4] Downloading SDXL VAE fp16 fix…")
    snapshot_download(
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        local_dir=str(SDXL_VAE_PATH),
    )
    print(f"  → Saved to {SDXL_VAE_PATH}")


def download_cogvideo():
    if COGVIDEO_PATH.exists() and any(COGVIDEO_PATH.iterdir()):
        print(f"[skip] CogVideoX-5b already at {COGVIDEO_PATH}")
        return
    print("[4/4] Downloading CogVideoX-5b…")
    snapshot_download(
        repo_id="THUDM/CogVideoX-5b",
        local_dir=str(COGVIDEO_PATH),
    )
    print(f"  → Saved to {COGVIDEO_PATH}")


def verify():
    all_ok = True
    for name, path in [
        ("FLUX.1-dev", FLUX_PATH),
        ("SDXL-base", SDXL_PATH),
        ("SDXL-VAE-fix", SDXL_VAE_PATH),
        ("CogVideoX-5b", COGVIDEO_PATH),
    ]:
        exists = path.exists() and any(path.iterdir()) if path.exists() else False
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {name}: {path}")
        if not exists:
            all_ok = False
    return all_ok


if __name__ == "__main__":
    download_flux()
    download_sdxl()
    download_sdxl_vae()
    download_cogvideo()

    print("\n── Verification ──")
    if verify():
        print("\nAll models downloaded successfully.")
    else:
        print("\nSome models are missing. Check errors above.")
        print("Note: FLUX.1-dev requires license acceptance at HuggingFace.")
