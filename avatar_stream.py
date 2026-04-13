"""
Phase 10 — Real-Time Avatar Streaming (StreamDiffusion).

Webcam → latent space → avatar generation at < 200ms per frame.
Uses StreamDiffusion with LCM-LoRA for latency consistency
and the subject LoRA for identity preservation.

Requires:
  - Webcam (cv2.VideoCapture(0))
  - >= 8GB VRAM for real-time performance
  - streamdiffusion package
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import cv2
import numpy as np
from configs.default import (
    SDXL_PATH, SUBJECT_LORA_PATH, TRIGGER_TOKEN, ENABLE_CPU_OFFLOAD,
)


def run_stream():
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.image_utils import postprocess_image
    from diffusers import AutoPipelineForImage2Image

    print("Loading SDXL img2img pipeline…")
    pipe = AutoPipelineForImage2Image.from_pretrained(
        str(SDXL_PATH), torch_dtype=torch.float16,
    ).to("cuda")

    if SUBJECT_LORA_PATH.exists():
        pipe.load_lora_weights(str(SUBJECT_LORA_PATH), adapter_name="subject")
        pipe.set_adapters(["subject"], adapter_weights=[1.0])
        print("  Subject LoRA loaded")

    stream = StreamDiffusion(
        pipe,
        t_index_list=[32, 45],
        torch_dtype=torch.float16,
        cfg_type="none",
    )
    stream.load_lcm_lora()
    stream.fuse_lora()

    prompt = f"a photo of {TRIGGER_TOKEN}, studio lighting, sharp focus"
    stream.prepare(prompt, num_inference_steps=50)

    print("Starting webcam stream… Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Exiting.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = stream(rgb)
        out_np = postprocess_image(output, output_type="np")

        if isinstance(out_np, np.ndarray):
            out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
        else:
            out_bgr = frame

        cv2.imshow("Avatar Stream", out_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream ended.")


if __name__ == "__main__":
    run_stream()
