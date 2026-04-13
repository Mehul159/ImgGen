"""
Phase 8 — Hot-Swap Multi-LoRA Inference Engine.

Reusable class that:
  - Pre-loads all LoRA adapters from a registry
  - Hot-swaps adapters per request via set_adapters()
  - Applies speed optimisations: xformers, tomesd token merging
  - Supports fuse_lora() for baked-in export
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path
from configs.default import (
    SDXL_PATH, SDXL_HUB_ID, resolve_model,
    SUBJECT_LORA_PATH, STYLE_LORA_PATH,
    OUTPUT_DIR, ENABLE_CPU_OFFLOAD, DEFAULT_STEPS, DEFAULT_GUIDANCE, DEFAULT_SEED,
)


class LoRAEngine:
    REGISTRY: dict[str, Path] = {
        "subject": SUBJECT_LORA_PATH,
        "style": STYLE_LORA_PATH,
    }

    def __init__(self, model_path: str | Path | None = None):
        if model_path:
            chosen = str(model_path)
        else:
            chosen = resolve_model(SDXL_PATH, SDXL_HUB_ID)

        print(f"Loading pipeline from {chosen}…")
        from diffusers import StableDiffusionXLPipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            chosen, torch_dtype=torch.float16,
        )

        if ENABLE_CPU_OFFLOAD:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")

        self._apply_speed_optimisations()
        self._load_adapters()

    def _apply_speed_optimisations(self):
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("  xformers enabled")
        except Exception:
            print("  [info] xformers not available, using default attention")

        try:
            import tomesd
            tomesd.apply_patch(self.pipe, ratio=0.5)
            print("  tomesd token merging enabled (ratio=0.5)")
        except Exception:
            print("  [info] tomesd not available")

    def _load_adapters(self):
        for name, path in self.REGISTRY.items():
            if path.exists() and any(path.iterdir()):
                self.pipe.load_lora_weights(str(path), adapter_name=name)
                print(f"  Loaded adapter: {name}")
            else:
                print(f"  [skip] Adapter not found: {name} ({path})")

    def generate(
        self,
        prompt: str,
        adapters: dict[str, float] | None = None,
        negative_prompt: str = "ugly, blurry, low quality, deformed",
        num_steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        seed: int = DEFAULT_SEED,
        width: int = 1024,
        height: int = 1024,
    ):
        if adapters:
            names = list(adapters.keys())
            weights = list(adapters.values())
            self.pipe.set_adapters(names, adapter_weights=weights)

        kwargs = dict(
            prompt=prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
            generator=torch.manual_seed(seed),
            width=width,
            height=height,
        )

        if hasattr(self.pipe, "encode_prompt"):
            kwargs["negative_prompt"] = negative_prompt

        return self.pipe(**kwargs).images[0]

    def fuse_and_export(self, output_path: str | Path):
        """Bake LoRA weights into the base model for deployment."""
        self.pipe.fuse_lora(lora_scale=1.0)
        self.pipe.save_pretrained(str(output_path))
        print(f"Fused model saved to {output_path}")

    def list_adapters(self):
        """Return names of currently loaded adapters."""
        return list(self.REGISTRY.keys())


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = LoRAEngine()

    img = engine.generate(
        prompt=f"a photo of sks person on Mars, in the style of artx",
        adapters={"subject": 1.0, "style": 0.6},
    )
    out_path = OUTPUT_DIR / "multi_lora_result.png"
    img.save(str(out_path))
    print(f"Saved: {out_path}")
