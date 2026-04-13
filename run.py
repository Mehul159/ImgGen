"""
Kaggle/Colab runner — install deps and run any phase.

Usage:
    python run.py setup          # install dependencies
    python run.py phase1         # download datasets
    python run.py phase2         # download models
    python run.py phase3         # preprocess (BLIP-2 + ControlNet maps)
    python run.py phase4         # train subject LoRA
    python run.py phase5         # train style LoRA
    python run.py phase6         # ControlNet inference
    python run.py phase7         # IP-Adapter inference
    python run.py phase8         # multi-LoRA engine test
    python run.py phase9         # AnimateDiff video
    python run.py app            # launch Gradio UI
    python run.py info           # show environment info
"""

import sys
import subprocess


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
    from configs.default import ROOT, SDXL_PATH, FLUX_PATH

    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:     {torch.cuda.get_device_name(0)}")
        print(f"VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Root:    {ROOT}")
    print(f"SDXL:    {'found' if SDXL_PATH.exists() else 'not downloaded'}")
    print(f"FLUX:    {'found' if FLUX_PATH.exists() else 'not downloaded'}")


PHASES = {
    "setup": setup,
    "info": info,
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
