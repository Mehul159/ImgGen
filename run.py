"""
Kaggle/Colab runner — install deps and run any phase.

Disk-optimised: uses SDXL only (~7 GB), loads from HuggingFace Hub
on demand. Total footprint ~20 GB, fits within Kaggle's 57 GB limit.

Usage:
    python run.py setup          # install dependencies
    python run.py info           # show environment + disk info
    python run.py phase1         # download datasets (~3 GB)
    python run.py phase2         # verify models (downloaded on demand)
    python run.py phase3         # preprocess (BLIP-2 + ControlNet maps)
    python run.py phase4         # train subject LoRA
    python run.py phase5         # train style LoRA
    python run.py phase6         # ControlNet inference
    python run.py phase7         # IP-Adapter inference
    python run.py phase8         # multi-LoRA engine test
    python run.py phase9         # AnimateDiff video
    python run.py app            # launch Gradio UI
    python run.py disk           # show disk usage breakdown
    python run.py clean-cache    # clear HuggingFace cache to free space
"""

import sys
import subprocess
import shutil


def run(cmd):
    subprocess.check_call(cmd, shell=True)


def setup():
    run(
        f"{sys.executable} -m pip install -q "
        '"diffusers>=0.30" "accelerate>=0.34" "peft>=0.12" '
        '"datasets>=2.20" "huggingface_hub>=0.24" '
        "controlnet_aux==0.0.9 bitsandbytes einops tomesd "
        '"imageio[ffmpeg]" safetensors sentencepiece'
    )
    print("\nDependencies installed.")


def info():
    import torch
    from configs.default import ROOT, SDXL_PATH, SDXL_HUB_ID, resolve_model

    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Root:    {ROOT}")
    src = resolve_model(SDXL_PATH, SDXL_HUB_ID)
    print(f"SDXL:    {src}")
    disk_usage()


def disk_usage():
    import os
    from pathlib import Path

    print("\n── Disk Usage ──")
    total, used, free = shutil.disk_usage("/")
    print(f"  Total: {total / 1e9:.1f} GB  Used: {used / 1e9:.1f} GB  Free: {free / 1e9:.1f} GB")

    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if hf_cache.exists():
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
        print(f"  HF cache: {cache_size / 1e9:.1f} GB ({hf_cache})")

    from configs.default import ROOT, DATA_DIR, MODEL_DIR, LORA_DIR, OUTPUT_DIR
    for label, path in [("data/", DATA_DIR), ("models/", MODEL_DIR),
                        ("lora_weights/", LORA_DIR), ("outputs/", OUTPUT_DIR)]:
        if path.exists():
            sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"  {label:15s} {sz / 1e9:.1f} GB")


def clean_cache():
    import os
    from pathlib import Path

    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    if hf_cache.exists():
        cache_size = sum(f.stat().st_size for f in hf_cache.rglob("*") if f.is_file())
        print(f"HF cache: {cache_size / 1e9:.1f} GB at {hf_cache}")
        shutil.rmtree(hf_cache)
        print("Cache cleared.")
    else:
        print("No HF cache found.")


PHASES = {
    "setup": setup,
    "info": info,
    "disk": disk_usage,
    "clean-cache": clean_cache,
    "phase1": lambda: run(f"{sys.executable} scripts/download_datasets.py"),
    "phase2": lambda: run(f"{sys.executable} scripts/download_models.py"),
    "phase3": lambda: run(f"{sys.executable} preprocess.py"),
    "phase4": lambda: run(f"{sys.executable} train_subject_lora.py"),
    "phase5": lambda: run(f"{sys.executable} train_style_lora.py"),
    "phase6": lambda: run(f"{sys.executable} controlnet_inference.py"),
    "phase7": lambda: run(f"{sys.executable} ip_adapter_inference.py"),
    "phase8": lambda: run(f"{sys.executable} lora_engine.py"),
    "phase9": lambda: run(f"{sys.executable} video_animatediff.py"),
    "app": lambda: run(f"{sys.executable} app.py"),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in PHASES:
        print(__doc__)
        sys.exit(1)
    PHASES[sys.argv[1]]()
