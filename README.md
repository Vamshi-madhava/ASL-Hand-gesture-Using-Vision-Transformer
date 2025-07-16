# ASL Alphabet Image Classifier (Custom Vision Transformer from Scratch)

A custom-built Vision Transformer (ViT) model trained from scratch on American Sign Language (ASL) alphabet images using PyTorch on Google Colab. 
Demonstrates use of attention-based architectures with RoPE, RMSNorm, and mixed precision training.

---

## Dataset

* **Dataset Source:** [Kaggle – ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* **Structure:** 87,000+ images across 29 classes (A–Z, `nothing`, `space`, `delete`)
* **Image Resolution Used:** `100x100` (configurable)

---

## Features

* Custom Vision Transformer with 12 layers, 768-dim embeddings, and 12 heads
* Rotary Positional Embedding (RoPE), inspired by LLaMA/Gemma
* RMSNorm instead of LayerNorm for normalization
* Mixed precision (`float16`) training using `torch.cuda.amp`
* Early stopping and cosine annealing learning rate scheduling
* Optimized `DataLoader` with prefetching and pinning
* Fully compatible with A100/T4/L4 GPUs on Colab

---

## Setup (Colab)

> No local installation needed. This project is designed for Google Colab.

1. Download the ASL dataset from Kaggle.
2. Upload `archive.zip` to your **Google Drive > MyDrive**.
3. Open the Colab notebook (`ASL_ViT.ipynb`) and run all cells.

---

## Model Architecture

```
Input: 100x100x3 RGB image
├── Patch Embedding via Conv2D (patch size: 16x16)
├── + [CLS] Token
├── 12 × Transformer Blocks:
│   ├── Multi-head Self-Attention with RoPE
│   ├── RMSNorm + Residual
│   ├── MLP with GELU + Residual
└── Linear Head → 29 Class Softmax
```

---

## Training Performance

* **GPU:** NVIDIA L4 (Colab)
* **Batch Size:** 512
* **Peak Validation Accuracy:** 99.85%
* **Model Size:** \~84M parameters
* **Model Checkpoint:** `best_vit_model.pth` is saved to Google Drive during training

---

## Results

| Metric              | Value               |
| ------------------- | ------------------- |
| Train Accuracy      | 99.99%             |
| Validation Accuracy | 99.85%              |
| Epochs Trained      | 10 (early stopping) |

Model shows strong generalization with minimal overfitting.

---

## Inference

* Load the trained model checkpoint (`best_vit_model.pth`) from Drive
* Upload test images of ASL gestures
* Run inference using the notebook pipeline

---

## Limitations

* Only supports single-frame static ASL alphabet classification
* The model has not been tested on real-time webcam input or natural backgrounds

---

## License

MIT License – open-source for learning and research.
