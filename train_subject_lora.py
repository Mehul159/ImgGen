"""
Phase 4 — DreamBooth + LoRA Training (Subject LoRA).

Trains a subject-specific LoRA adapter using PEFT on FLUX.1-dev
(or SDXL as fallback). Includes pivotal tuning: a new trigger
token embedding is trained jointly with LoRA weights.

Requires >= 16GB VRAM. Use gradient checkpointing + CPU offload
for tighter budgets.
"""

import sys, os, json

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from configs.default import (
    FLUX_PATH, SDXL_PATH, PROCESSED_DIR, SUBJECT_LORA_PATH,
    TRIGGER_TOKEN, SUBJECT_LORA_RANK, SUBJECT_LORA_ALPHA,
    SUBJECT_LR, SUBJECT_STEPS, SUBJECT_BATCH_SIZE, SUBJECT_GRAD_ACCUM,
)


class DreamBoothDataset(Dataset):
    def __init__(self, data_dir: Path):
        meta_path = data_dir / "metadata.jsonl"
        with open(meta_path) as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["file_name"]).convert("RGB")
        return {"image": img, "caption": s["text"]}


def load_pipeline():
    """Load FLUX.1-dev if available, else fall back to SDXL."""
    if FLUX_PATH.exists() and any(FLUX_PATH.iterdir()):
        print(f"Loading FLUX.1-dev from {FLUX_PATH}…")
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            str(FLUX_PATH), torch_dtype=torch.bfloat16,
        )
        return pipe, "flux"
    elif SDXL_PATH.exists() and any(SDXL_PATH.iterdir()):
        print(f"Loading SDXL from {SDXL_PATH}…")
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            str(SDXL_PATH), torch_dtype=torch.bfloat16,
        )
        return pipe, "sdxl"
    else:
        raise FileNotFoundError(
            "No base model found. Run scripts/download_models.py first."
        )


def setup_lora(transformer):
    """Wrap transformer with PEFT LoRA config."""
    lora_cfg = LoraConfig(
        r=SUBJECT_LORA_RANK,
        lora_alpha=SUBJECT_LORA_ALPHA,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "ff.net.0.proj", "ff.net.2",
        ],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(transformer, lora_cfg)
    model.print_trainable_parameters()
    return model


def setup_pivotal_tuning(pipe):
    """Add trigger token to tokenizer and unfreeze its embedding."""
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    num_added = tokenizer.add_tokens([TRIGGER_TOKEN])
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(TRIGGER_TOKEN)
    print(f"Added {num_added} token(s), trigger ID = {token_id}")

    for name, param in text_encoder.named_parameters():
        param.requires_grad = (
            name == "text_model.embeddings.token_embedding.weight"
        )

    return tokenizer, text_encoder


def train():
    SUBJECT_LORA_PATH.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=SUBJECT_GRAD_ACCUM,
    )

    pipe, model_type = load_pipeline()

    if model_type == "flux":
        transformer = pipe.transformer
    else:
        transformer = pipe.unet

    transformer = setup_lora(transformer)
    transformer.gradient_checkpointing_enable()
    tokenizer, text_encoder = setup_pivotal_tuning(pipe)

    dataset = DreamBoothDataset(PROCESSED_DIR)
    loader = DataLoader(dataset, batch_size=SUBJECT_BATCH_SIZE, shuffle=True)

    trainable_params = (
        list(transformer.parameters())
        + list(text_encoder.get_input_embeddings().parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=SUBJECT_LR, weight_decay=1e-2)

    transformer, text_encoder, optimizer, loader = accelerator.prepare(
        transformer, text_encoder, optimizer, loader,
    )

    vae = pipe.vae.to(accelerator.device, dtype=torch.bfloat16)
    vae.requires_grad_(False)

    print(f"\nStarting training: {SUBJECT_STEPS} steps, batch={SUBJECT_BATCH_SIZE}, "
          f"grad_accum={SUBJECT_GRAD_ACCUM}")

    global_step = 0
    while global_step < SUBJECT_STEPS:
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
                    0, 1000, (latents.shape[0],), device=accelerator.device
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
                print(f"  Step {global_step}/{SUBJECT_STEPS} | loss: {loss.item():.4f}")
            if global_step >= SUBJECT_STEPS:
                break

    unwrapped = accelerator.unwrap_model(transformer)
    unwrapped.save_pretrained(str(SUBJECT_LORA_PATH))
    print(f"\nSubject LoRA saved to {SUBJECT_LORA_PATH}")


if __name__ == "__main__":
    train()
