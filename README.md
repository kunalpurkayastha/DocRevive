
# DocRevive: A Unified Pipeline for Document Text Restoration

DocRevive is a unified pipeline for restoring document pages damaged by physical occlusions like stamps, ink, whitener fluid, dust, scribbles, and more. It combines **diffusion-based text editing** with **generative inpainting** to recover both the textual content and visual appearance of occluded regions.

---

## Installation

**Requirements:** Python 3.10, CUDA 12.x, conda

```bash
conda create -n docrevive python=3.10
conda activate docrevive
pip install -r requirements.txt
```

---

## Weights

All model weights live in the `weights/` directory. They are excluded from git but uploaded as a single archive which can be downloaded from [HERE](https://drive.google.com/file/d/1BMHt-oqKLHIrsNTCknppY7D4ZYU9R--r/view?usp=sharing)

```bash
# Restore weights from archive
unzip weights.zip
```

| Path | Model |
|------|-------|
| `weights/model.pth` | text editing diffusion model |
| `weights/vision_model.pth` | text editing vision backbone (ABINet) |
| `weights/text_encoder.pth` | text editing text encoder |
| `weights/style_encoder.pth` | text editing style encoder |
| `weights/vitstr_base_patch16_224.pth` | ViTSTR backbone |
| `weights/vgg19.pth` | VGG-19 perceptual loss |
| `weights/yolo_occlusion.pt` | YOLO occlusion detector |
| `weights/db_resnet50_trained.pth` | text detector (not necessary) |
| `weights/fast_base_trained.pth` | text detector (docTR) |
| `weights/parseq_trained.pth` | text recogniser (docTR) |
| `weights/ocr_model.pth` | OCR model |
| `weights/checkpoint/spm.pt` | text inpainting spatial prior model |
| `weights/checkpoint/rm_gen.pth` | text inpainting refinement model |
| `weights/roberta/` | Fine-tuned RoBERTa |
| `weights/qwen3/` | Qwen3 language model (not necessary) |
| `weights/sd/` | Stable Diffusion v1-5 |


---

## Usage

### Quick start

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n cvpr python main.py \
    --input_dir samples/inputs \
    --output_dir samples/outputs \
    --no_llm \
    --gpu_ids 0,1,2,3 \
    --debug
```

### Single image

```bash
conda run -n cvpr python main.py \
    --input /path/to/document.jpg \
    --output_dir ./outputs \
    --no_llm
```

### Multi-GPU batch

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n cvpr python main.py \
    --input_dir /path/to/images/ \
    --output_dir ./outputs \
    --gpu_ids 0,1,2,3
```

---

Override any field programmatically:

```python
from config import Config
from pipeline import DocRevivePipeline

cfg = Config(llm_mode="roberta", debug=True)
pipeline = DocRevivePipeline(cfg)
result = pipeline.process("document.jpg")
```

---

## Dataset

Evaluation uses the **OPRB (Occluded Pages Restoration Benchmark)**:
[https://huggingface.co/datasets/kpurkayastha/OPRB](https://huggingface.co/datasets/kpurkayastha/OPRB)


