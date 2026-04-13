# DreamBooth LoRA Studio

Personalized image and video generation system using DreamBooth + LoRA fine-tuning on FLUX.1-dev / SDXL, with ControlNet, IP-Adapter, AnimateDiff, and CogVideoX integration.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 16 GB (SDXL inference) | 24+ GB (FLUX training, CogVideoX) |
| RAM | 32 GB | 64 GB |
| Disk | 100 GB | 500 GB (all models + datasets) |
| CUDA | 11.8+ | 12.1+ |

> **Note:** GTX 1650 (4 GB) can only run lightweight smoke tests. Use a cloud GPU (A100, H100, RTX 4090) for training and full inference.

## Project Structure

```
ImgGen/
├── configs/
│   └── default.py              # Central configuration
├── scripts/
│   ├── download_datasets.py    # Phase 1: dataset download
│   └── download_models.py      # Phase 2: model download
├── preprocess.py               # Phase 3: BLIP-2 captioning + ControlNet maps
├── train_subject_lora.py       # Phase 4: DreamBooth + LoRA (subject)
├── train_style_lora.py         # Phase 5: Style LoRA
├── controlnet_inference.py     # Phase 6: Multi-ControlNet inference
├── ip_adapter_inference.py     # Phase 7: IP-Adapter guided generation
├── lora_engine.py              # Phase 8: Hot-swap multi-LoRA engine
├── video_cogvideox.py          # Phase 9a: CogVideoX fine-tuning
├── video_animatediff.py        # Phase 9b: AnimateDiff generation
├── avatar_stream.py            # Phase 10: Real-time webcam avatar
├── app.py                      # Phase 11: Gradio web UI
├── requirements.txt
├── data/                       # Downloaded datasets
├── models/                     # Downloaded base models
├── lora_weights/               # Trained LoRA adapters
└── outputs/                    # Generated images/videos
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For PyTorch with CUDA, install from https://pytorch.org/get-started/locally/ first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download Datasets

```bash
python scripts/download_datasets.py
```

### 3. Download Models

FLUX.1-dev requires accepting the license at https://huggingface.co/black-forest-labs/FLUX.1-dev

```bash
huggingface-cli login
python scripts/download_models.py
```

### 4. Preprocess Data

```bash
python preprocess.py
```

### 5. Train LoRA Adapters

```bash
python train_subject_lora.py   # Subject LoRA (1000 steps)
python train_style_lora.py     # Style LoRA (500 steps)
```

### 6. Run Inference

```bash
python controlnet_inference.py   # ControlNet smoke test
python lora_engine.py            # Multi-LoRA smoke test
python ip_adapter_inference.py   # IP-Adapter test
```

### 7. Launch Gradio UI

```bash
python app.py
```

### 8. Video Generation (Optional)

```bash
python video_animatediff.py      # AnimateDiff GIF
python video_cogvideox.py        # CogVideoX (24GB+ VRAM)
```

### 9. Real-Time Avatar (Optional)

```bash
python avatar_stream.py          # Requires webcam
```

## Configuration

All settings are centralized in `configs/default.py`:

- **Paths**: dataset, model, and output directories
- **Training**: LoRA rank, learning rate, steps, batch size
- **Inference**: default steps, guidance scale, seed
- **Hardware**: dtype, CPU offload toggles

## Key Features

- **DreamBooth + LoRA**: Identity-preserving fine-tuning with trigger token
- **Pivotal Tuning**: New token embedding trained jointly with LoRA weights
- **Multi-LoRA Hot-Swap**: Load/swap adapters at runtime with per-adapter weights
- **ControlNet**: Pose, canny edge, and depth conditioning
- **IP-Adapter**: CLIP image embedding injection for image-guided generation
- **AnimateDiff**: Motion LoRA for video generation
- **CogVideoX**: DiT-based video fine-tuning with LoRA
- **StreamDiffusion**: Real-time webcam-to-avatar pipeline
- **Speed Optimizations**: xformers, tomesd token merging, CPU offload

## Execution Order

```
Phase 0  → pip install -r requirements.txt
Phase 1  → python scripts/download_datasets.py
Phase 2  → python scripts/download_models.py
Phase 3  → python preprocess.py
Phase 4  → python train_subject_lora.py
Phase 5  → python train_style_lora.py
Phase 6  → python controlnet_inference.py
Phase 7  → python ip_adapter_inference.py
Phase 8  → python lora_engine.py
Phase 9  → python video_animatediff.py
Phase 10 → python avatar_stream.py
Phase 11 → python app.py
```

## License

Models have their own licenses. Check each Hugging Face model card before commercial use.
