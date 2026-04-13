"""
Phase 11 — Gradio Demo UI.

Full-featured web interface for the DreamBooth LoRA Studio:
  - Text-to-image with adjustable LoRA adapter weights
  - ControlNet conditioning (pose/canny/depth from uploaded image)
  - IP-Adapter image-guided generation
  - AnimateDiff video generation tab
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import gradio as gr
from PIL import Image
from pathlib import Path
from configs.default import (
    OUTPUT_DIR, TRIGGER_TOKEN, STYLE_TRIGGER,
    DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_SEED,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

engine = None


def get_engine():
    global engine
    if engine is None:
        from lora_engine import LoRAEngine
        engine = LoRAEngine()
    return engine


def generate_image(prompt, subject_weight, style_weight, steps, guidance, seed):
    eng = get_engine()
    adapters = {}
    if subject_weight > 0:
        adapters["subject"] = subject_weight
    if style_weight > 0:
        adapters["style"] = style_weight

    img = eng.generate(
        prompt=prompt,
        adapters=adapters if adapters else None,
        num_steps=int(steps),
        guidance=guidance,
        seed=int(seed),
    )
    return img


def generate_with_controlnet(
    prompt, source_image, cn_pose_scale, cn_canny_scale, cn_depth_scale,
    subject_weight, style_weight, steps, guidance, seed,
):
    if source_image is None:
        return None

    from controlnet_inference import load_controlnets, build_pipeline, extract_conditioning

    pipe = build_pipeline(load_controlnets())
    source_img = Image.fromarray(source_image).convert("RGB").resize((1024, 1024))
    cond_images = extract_conditioning(source_img)

    result = pipe(
        prompt=prompt,
        negative_prompt="ugly, blurry, low quality, deformed",
        image=cond_images,
        controlnet_conditioning_scale=[cn_pose_scale, cn_canny_scale, cn_depth_scale],
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        generator=torch.manual_seed(int(seed)),
    ).images[0]

    return result


def generate_animation_ui(prompt, num_frames, steps, guidance, seed):
    from diffusers.utils import export_to_gif
    from video_animatediff import download_animatediff, ANIMATEDIFF_ADAPTER_PATH, ANIMATEDIFF_LORA_PATH
    from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
    from configs.default import SDXL_PATH, SUBJECT_LORA_PATH

    download_animatediff()
    adapter = MotionAdapter.from_pretrained(str(ANIMATEDIFF_ADAPTER_PATH))
    pipe = AnimateDiffPipeline.from_pretrained(
        str(SDXL_PATH), motion_adapter=adapter, torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config, beta_schedule="linear",
        clip_sample=False, timestep_spacing="linspace", steps_offset=1,
    )
    pipe.enable_model_cpu_offload()

    output = pipe(
        prompt=prompt,
        num_frames=int(num_frames),
        guidance_scale=guidance,
        num_inference_steps=int(steps),
        generator=torch.manual_seed(int(seed)),
    )
    frames = output.frames[0]
    gif_path = str(OUTPUT_DIR / "gradio_animation.gif")
    export_to_gif(frames, gif_path)
    return gif_path


def build_ui():
    with gr.Blocks(
        title="DreamBooth LoRA Studio",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# DreamBooth LoRA Studio\n"
            "Personalized image & video generation with hot-swappable LoRA adapters, "
            "ControlNet conditioning, and IP-Adapter guidance."
        )

        with gr.Tabs():
            # ── Tab 1: Text-to-Image ────────────────────────
            with gr.TabItem("Text to Image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_t2i = gr.Textbox(
                            label="Prompt",
                            value=f"a photo of {TRIGGER_TOKEN} on Mars, {STYLE_TRIGGER}",
                            lines=3,
                        )
                        with gr.Row():
                            subj_w = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Subject LoRA")
                            style_w = gr.Slider(0, 1.5, value=0.6, step=0.05, label="Style LoRA")
                        with gr.Row():
                            steps_t2i = gr.Slider(10, 50, value=DEFAULT_STEPS, step=1, label="Steps")
                            guidance_t2i = gr.Slider(1.0, 15.0, value=DEFAULT_GUIDANCE, step=0.5, label="Guidance")
                        seed_t2i = gr.Number(value=DEFAULT_SEED, label="Seed", precision=0)
                        btn_t2i = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        out_t2i = gr.Image(label="Output", type="pil")

                btn_t2i.click(
                    generate_image,
                    [prompt_t2i, subj_w, style_w, steps_t2i, guidance_t2i, seed_t2i],
                    out_t2i,
                )

            # ── Tab 2: ControlNet ───────────────────────────
            with gr.TabItem("ControlNet"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_cn = gr.Textbox(
                            label="Prompt",
                            value=f"a photo of {TRIGGER_TOKEN} walking in Tokyo, cinematic",
                            lines=3,
                        )
                        source_img = gr.Image(label="Source Image (for pose/depth/canny)", type="numpy")
                        with gr.Row():
                            cn_pose = gr.Slider(0, 1.5, value=0.8, step=0.05, label="Pose Scale")
                            cn_canny = gr.Slider(0, 1.5, value=0.5, step=0.05, label="Canny Scale")
                            cn_depth = gr.Slider(0, 1.5, value=0.5, step=0.05, label="Depth Scale")
                        with gr.Row():
                            cn_subj = gr.Slider(0, 1.5, value=1.0, step=0.05, label="Subject LoRA")
                            cn_style = gr.Slider(0, 1.5, value=0.6, step=0.05, label="Style LoRA")
                        with gr.Row():
                            steps_cn = gr.Slider(10, 50, value=30, step=1, label="Steps")
                            guidance_cn = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance")
                        seed_cn = gr.Number(value=DEFAULT_SEED, label="Seed", precision=0)
                        btn_cn = gr.Button("Generate with ControlNet", variant="primary")
                    with gr.Column(scale=1):
                        out_cn = gr.Image(label="Output", type="pil")

                btn_cn.click(
                    generate_with_controlnet,
                    [prompt_cn, source_img, cn_pose, cn_canny, cn_depth,
                     cn_subj, cn_style, steps_cn, guidance_cn, seed_cn],
                    out_cn,
                )

            # ── Tab 3: AnimateDiff Video ────────────────────
            with gr.TabItem("Video (AnimateDiff)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        prompt_vid = gr.Textbox(
                            label="Prompt",
                            value=f"{TRIGGER_TOKEN} walking through a forest, cinematic, 4k",
                            lines=3,
                        )
                        num_frames = gr.Slider(8, 32, value=16, step=1, label="Frames")
                        with gr.Row():
                            steps_vid = gr.Slider(10, 50, value=25, step=1, label="Steps")
                            guidance_vid = gr.Slider(1, 15, value=7.5, step=0.5, label="Guidance")
                        seed_vid = gr.Number(value=DEFAULT_SEED, label="Seed", precision=0)
                        btn_vid = gr.Button("Generate Animation", variant="primary")
                    with gr.Column(scale=1):
                        out_vid = gr.Image(label="Animation (GIF)")

                btn_vid.click(
                    generate_animation_ui,
                    [prompt_vid, num_frames, steps_vid, guidance_vid, seed_vid],
                    out_vid,
                )

        gr.Markdown(
            "---\n"
            f"**Trigger token:** `{TRIGGER_TOKEN}` · "
            f"**Style trigger:** `{STYLE_TRIGGER}` · "
            "Built with Diffusers + PEFT + Gradio"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)
