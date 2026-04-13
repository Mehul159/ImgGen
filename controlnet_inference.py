"""
Phase 6 — ControlNet Integration.

Multi-ControlNet inference (pose + canny + depth) on SDXL with
subject + style LoRA adapters hot-swapped in.

ControlNet models downloaded on first run:
  - thibaud/controlnet-openpose-sdxl-1.0
  - diffusers/controlnet-canny-sdxl-1.0
  - diffusers/controlnet-depth-sdxl-1.0
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
)
from controlnet_aux import OpenposeDetector, CannyDetector, MidasDetector
from configs.default import (
    SDXL_PATH, SDXL_VAE_PATH, SUBJECT_LORA_PATH, STYLE_LORA_PATH,
    PROCESSED_DIR, OUTPUT_DIR, TRIGGER_TOKEN, STYLE_TRIGGER,
    ENABLE_CPU_OFFLOAD, DEFAULT_STEPS, DEFAULT_SEED,
)


def load_controlnets():
    print("Loading ControlNet models…")
    controlnets = [
        ControlNetModel.from_pretrained(
            "thibaud/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16,
        ),
        ControlNetModel.from_pretrained(
            "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16,
        ),
        ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16,
        ),
    ]
    return controlnets


def build_pipeline(controlnets):
    print("Building SDXL ControlNet pipeline…")
    vae = AutoencoderKL.from_pretrained(
        str(SDXL_VAE_PATH), torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        str(SDXL_PATH),
        controlnet=controlnets,
        vae=vae,
        torch_dtype=torch.float16,
    )

    if ENABLE_CPU_OFFLOAD:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if SUBJECT_LORA_PATH.exists():
        pipe.load_lora_weights(str(SUBJECT_LORA_PATH), adapter_name="subject")
        print("  Loaded subject LoRA")
    if STYLE_LORA_PATH.exists():
        pipe.load_lora_weights(str(STYLE_LORA_PATH), adapter_name="style")
        print("  Loaded style LoRA")

    adapters = []
    weights = []
    if SUBJECT_LORA_PATH.exists():
        adapters.append("subject")
        weights.append(1.0)
    if STYLE_LORA_PATH.exists():
        adapters.append("style")
        weights.append(0.6)
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)

    return pipe


def extract_conditioning(source_img: Image.Image):
    """Generate pose, canny, and depth maps from a source image."""
    pose_det = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    canny_det = CannyDetector()
    depth_det = MidasDetector.from_pretrained("lllyasviel/ControlNet")

    pose_img = pose_det(source_img)
    canny_img = canny_det(source_img, low_threshold=100, high_threshold=200)
    depth_img = depth_det(source_img)

    return [pose_img, canny_img, depth_img]


def run_inference():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    controlnets = load_controlnets()
    pipe = build_pipeline(controlnets)

    source_path = PROCESSED_DIR / "img_0000.jpg"
    if not source_path.exists():
        print(f"Source image not found at {source_path}. Using a blank 1024×1024.")
        source_img = Image.new("RGB", (1024, 1024), (128, 128, 128))
    else:
        source_img = Image.open(str(source_path)).convert("RGB")

    print("Extracting conditioning maps…")
    cond_images = extract_conditioning(source_img)

    prompt = (
        f"a photo of {TRIGGER_TOKEN} walking in Tokyo, "
        f"{STYLE_TRIGGER}, cinematic lighting, 4k"
    )
    negative_prompt = "ugly, blurry, low quality, deformed, disfigured"

    print(f"Generating with prompt: {prompt[:80]}…")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=cond_images,
        controlnet_conditioning_scale=[0.8, 0.5, 0.5],
        num_inference_steps=DEFAULT_STEPS,
        guidance_scale=7.5,
        generator=torch.manual_seed(DEFAULT_SEED),
    ).images[0]

    out_path = OUTPUT_DIR / "controlnet_result.png"
    result.save(str(out_path))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run_inference()
