"""
Phase 9b — AnimateDiff Motion LoRA.

Uses AnimateDiff motion modules with LoRA for lightweight
video generation from SDXL. Combines motion LoRA with
subject LoRA for identity-consistent animation.

Downloads motion adapter from guoyww/animatediff on first run.
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from configs.default import (
    SDXL_PATH, SUBJECT_LORA_PATH, MODEL_DIR, OUTPUT_DIR,
    ENABLE_CPU_OFFLOAD, TRIGGER_TOKEN,
)

ANIMATEDIFF_ADAPTER_PATH = MODEL_DIR / "animatediff-motion-adapter"
ANIMATEDIFF_LORA_PATH = MODEL_DIR / "animatediff-motion-lora"


def download_animatediff():
    if not (ANIMATEDIFF_ADAPTER_PATH.exists() and any(ANIMATEDIFF_ADAPTER_PATH.iterdir())):
        print("Downloading AnimateDiff motion adapter…")
        snapshot_download(
            repo_id="guoyww/animatediff-motion-adapter-v1-5-2",
            local_dir=str(ANIMATEDIFF_ADAPTER_PATH),
        )

    if not (ANIMATEDIFF_LORA_PATH.exists() and any(ANIMATEDIFF_LORA_PATH.iterdir())):
        print("Downloading AnimateDiff motion LoRA…")
        snapshot_download(
            repo_id="guoyww/animatediff-motion-lora-v1-5-2",
            local_dir=str(ANIMATEDIFF_LORA_PATH),
        )


def generate_animation():
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from diffusers.utils import export_to_gif

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_animatediff()

    print("Loading AnimateDiff pipeline…")
    adapter = MotionAdapter.from_pretrained(str(ANIMATEDIFF_ADAPTER_PATH))

    pipe = AnimateDiffPipeline.from_pretrained(
        str(SDXL_PATH),
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_schedule="linear",
        clip_sample=False,
        timestep_spacing="linspace",
        steps_offset=1,
    )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if ANIMATEDIFF_LORA_PATH.exists():
        pipe.load_lora_weights(str(ANIMATEDIFF_LORA_PATH), adapter_name="pan-left")
        print("  Motion LoRA loaded")
    if SUBJECT_LORA_PATH.exists():
        pipe.load_lora_weights(str(SUBJECT_LORA_PATH), adapter_name="subject")
        print("  Subject LoRA loaded")

    active_adapters = []
    active_weights = []
    if ANIMATEDIFF_LORA_PATH.exists():
        active_adapters.append("pan-left")
        active_weights.append(0.8)
    if SUBJECT_LORA_PATH.exists():
        active_adapters.append("subject")
        active_weights.append(1.0)
    if active_adapters:
        pipe.set_adapters(active_adapters, adapter_weights=active_weights)

    prompt = f"{TRIGGER_TOKEN} walking through a forest, cinematic, 4k"
    print(f"Generating animation: {prompt}")

    output = pipe(
        prompt=prompt,
        num_frames=16,
        guidance_scale=7.5,
        num_inference_steps=25,
        generator=torch.manual_seed(42),
    )
    frames = output.frames[0]

    gif_path = OUTPUT_DIR / "animation.gif"
    export_to_gif(frames, str(gif_path))
    print(f"Saved: {gif_path}")


if __name__ == "__main__":
    generate_animation()
