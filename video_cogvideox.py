"""
Phase 9a — CogVideoX-5b Fine-tuning with LoRA.

Trains temporal attention LoRA on short video clips for
identity-consistent video generation. Uses PEFT on the
DiT transformer blocks.

Requires >= 24GB VRAM (A100/H100 recommended).
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from configs.default import COGVIDEO_PATH, VIDEO_DIR, LORA_DIR

COGVIDEO_LORA_PATH = LORA_DIR / "cogvideo_lora"


def train():
    from diffusers import CogVideoXPipeline

    COGVIDEO_LORA_PATH.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=2)

    print(f"Loading CogVideoX-5b from {COGVIDEO_PATH}…")
    pipe = CogVideoXPipeline.from_pretrained(
        str(COGVIDEO_PATH), torch_dtype=torch.bfloat16,
    )
    transformer = pipe.transformer

    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=32,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.print_trainable_parameters()

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=1e-5, weight_decay=1e-2)
    transformer, optimizer = accelerator.prepare(transformer, optimizer)

    print("\nCogVideoX LoRA training setup complete.")
    print("Full training requires a video DataLoader processing (B, T, C, H, W) tensors.")
    print(f"Video clips directory: {VIDEO_DIR}")
    print(f"LoRA will be saved to: {COGVIDEO_LORA_PATH}")

    # Placeholder training loop structure:
    # for epoch in range(num_epochs):
    #     for batch in video_loader:
    #         video_latents = pipe.vae.encode(batch["video"]).latent_dist.sample()
    #         noise = torch.randn_like(video_latents)
    #         timesteps = torch.randint(0, 1000, (B,))
    #         noisy = pipe.scheduler.add_noise(video_latents, noise, timesteps)
    #         pred = transformer(noisy, timestep=timesteps, encoder_hidden_states=...).sample
    #         loss = F.mse_loss(pred, noise)
    #         accelerator.backward(loss)
    #         optimizer.step(); optimizer.zero_grad()

    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped.save_pretrained(str(COGVIDEO_LORA_PATH))
    print(f"CogVideoX LoRA saved to {COGVIDEO_LORA_PATH}")


if __name__ == "__main__":
    train()
