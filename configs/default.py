"""
Central configuration for the entire pipeline.
Edit values here rather than in individual scripts.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
SUBJECT_DIR = DATA_DIR / "subject_images"
REG_DIR = DATA_DIR / "reg_images"
STYLE_DIR = DATA_DIR / "style_images"
CONTROLNET_DIR = DATA_DIR / "controlnet_pairs"
VIDEO_DIR = DATA_DIR / "video_clips"
PROCESSED_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT / "models"
SDXL_PATH = MODEL_DIR / "sdxl-base"
SDXL_VAE_PATH = MODEL_DIR / "sdxl-vae-fix"

LORA_DIR = ROOT / "lora_weights"
SUBJECT_LORA_PATH = LORA_DIR / "subject_lora"
STYLE_LORA_PATH = LORA_DIR / "style_lora"

OUTPUT_DIR = ROOT / "outputs"

# ── Hub Model IDs (loaded on demand, no bulk download) ─────
SDXL_HUB_ID = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_VAE_HUB_ID = "madebyollin/sdxl-vae-fp16-fix"

# ── Training ───────────────────────────────────────────────
TRIGGER_TOKEN = "sks person"
STYLE_TRIGGER = "in the style of artx"

SUBJECT_LORA_RANK = 64
SUBJECT_LORA_ALPHA = 64
SUBJECT_LR = 1e-4
SUBJECT_STEPS = 1000
SUBJECT_BATCH_SIZE = 1
SUBJECT_GRAD_ACCUM = 4

STYLE_LORA_RANK = 32
STYLE_LORA_ALPHA = 32
STYLE_LR = 5e-5
STYLE_STEPS = 500

# ── Inference ──────────────────────────────────────────────
DEFAULT_STEPS = 28
DEFAULT_GUIDANCE = 3.5
DEFAULT_SEED = 42

# ── Hardware ───────────────────────────────────────────────
DTYPE = "bfloat16"  # "float16" for older GPUs
ENABLE_CPU_OFFLOAD = True  # enable for <16GB VRAM
ENABLE_SEQUENTIAL_OFFLOAD = False  # enable for <8GB VRAM


def resolve_model(local_path: Path, hub_id: str) -> str:
    """Return local path if it exists, otherwise the hub ID for from_pretrained()."""
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path)
    return hub_id
