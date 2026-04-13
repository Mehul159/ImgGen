"""
Phase 5 — Style LoRA Training (separate adapter).

Trains a style-specific LoRA adapter on the WikiArt dataset.
No pivotal tuning — just pure PEFT LoRA on attention + FF layers.
Lighter rank (32) and shorter schedule than the subject LoRA.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from configs.default import (
    SDXL_PATH, SDXL_HUB_ID, resolve_model,
    STYLE_DIR, STYLE_LORA_PATH,
    STYLE_TRIGGER, STYLE_LORA_RANK, STYLE_LORA_ALPHA,
    STYLE_LR, STYLE_STEPS, SUBJECT_BATCH_SIZE, SUBJECT_GRAD_ACCUM,
)


class StyleDataset(Dataset):
    def __init__(self, data_dir: Path):
        print(f"Loading style dataset from {data_dir}…")
        self.ds = load_from_disk(str(data_dir))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"].convert("RGB").resize((1024, 1024), Image.LANCZOS)
        caption = f"{STYLE_TRIGGER}, {sample.get('text', 'a painting')}"
        return {"image": img, "caption": caption}


def load_pipeline():
    src = resolve_model(SDXL_PATH, SDXL_HUB_ID)
    print(f"Loading SDXL from {src}…")
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        src, torch_dtype=torch.bfloat16,
    )
    return pipe, "sdxl"


def train():
    STYLE_LORA_PATH.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=SUBJECT_GRAD_ACCUM,
    )

    pipe, model_type = load_pipeline()
    transformer = pipe.transformer if model_type == "flux" else pipe.unet

    lora_cfg = LoraConfig(
        r=STYLE_LORA_RANK,
        lora_alpha=STYLE_LORA_ALPHA,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=0.0,
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_cfg)
    transformer.print_trainable_parameters()
    if hasattr(pipe.unet, "enable_gradient_checkpointing"):
        pipe.unet.enable_gradient_checkpointing()

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    text_encoder.requires_grad_(False)

    dataset = StyleDataset(STYLE_DIR)

    def collate_fn(batch):
        return {"image": [b["image"] for b in batch], "caption": [b["caption"] for b in batch]}

    loader = DataLoader(dataset, batch_size=SUBJECT_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        transformer.parameters(), lr=STYLE_LR, weight_decay=1e-2,
    )

    transformer, optimizer, loader = accelerator.prepare(
        transformer, optimizer, loader,
    )

    vae = pipe.vae.to(accelerator.device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    text_encoder = text_encoder.to(accelerator.device)

    print(f"\nStarting style LoRA training: {STYLE_STEPS} steps")

    global_step = 0
    while global_step < STYLE_STEPS:
        for batch in loader:
            with accelerator.accumulate(transformer):
                images = [img for img in batch["image"]]
                pixel_values = torch.stack([
                    pipe.image_processor.preprocess(img)
                    if hasattr(pipe, "image_processor")
                    else torch.tensor([])
                    for img in images
                ]).to(accelerator.device, dtype=torch.bfloat16)

                latents = (
                    vae.encode(pixel_values).latent_dist.sample()
                    * vae.config.scaling_factor
                )

                input_ids = tokenizer(
                    batch["caption"],
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)

                encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],), device=accelerator.device,
                )
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                pred = transformer(
                    noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % 100 == 0:
                print(f"  Step {global_step}/{STYLE_STEPS} | loss: {loss.item():.4f}")
            if global_step >= STYLE_STEPS:
                break

    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped.save_pretrained(str(STYLE_LORA_PATH))
    print(f"\nStyle LoRA saved to {STYLE_LORA_PATH}")


if __name__ == "__main__":
    train()
