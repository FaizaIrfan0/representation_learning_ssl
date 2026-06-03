# Representation Learning: SimCLR vs MAE

AIMS DTU Research Internship 2026 — Comparative Analysis of Self-Supervised Learning Paradigms

## Objective

This project trains two self-supervised learning models on the ROD (Real-Time Obstacle Detection) dataset and compares the quality of the visual representations they learn — without using any labels during training.

The two models compared are:
- **SimCLR** — learns by comparing augmented versions of the same image
- **MAE (Masked Autoencoder)** — learns by reconstructing randomly masked image patches

## Dataset

ROD Dataset — 24,326 urban street images across 25 obstacle categories (cars, people, traffic signs, etc.)  
Source: https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset

## Results

|           Metric          |  SimCLR  |    MAE     |
|---------------------------|----------|------------|
| Linear Probe Accuracy     | 81.66%   | 41.19%.    |
| Silhouette Score          | 0.0490   | -0.3254    |
| Semantic Consistency      | 0.8900   | 0.6212     |
| Training Time (40 epochs) | ~4 hours | ~1.5 hours |

SimCLR produces significantly better class-discriminative representations on this dataset. MAE is 2.5x more compute-efficient but requires more epochs to build comparable semantic structure.

## Project Structure
representation_learning_ssl/
├── ssl_ntbk.ipynb  
├── README.md  

## How to Run

1. Open the notebook in Google Colab
2. Download the ROD dataset from Kaggle and upload to Google Drive as `archive.zip`
3. Run all cells in order — dataset loading, SimCLR training, MAE training, evaluation

Hardware used: NVIDIA T4 GPU (Google Colab free tier)

## Model Architecture

**SimCLR**
- Encoder: ResNet-18 (512-dim embeddings)
- Projection Head: 2-layer MLP (512 → 512 → 128)
- Loss: NT-Xent contrastive loss

**MAE**
- Patch size: 16x16, producing 196 patches per image
- Encoder: ViT with 4 transformer layers, 8 attention heads
- Decoder: 2-layer transformer decoder
- Mask ratio: 75%
- Loss: MSE on masked patches only

## Evaluation Methods

- UMAP visualization of embedding space
- Linear probe accuracy (frozen encoder + logistic regression)
- Silhouette score for cluster quality
- Semantic consistency via cosine similarity of augmented pairs

## References

1. Chen et al. (2020). A Simple Framework for Contrastive Learning. ICML. https://arxiv.org/abs/2002.05709
2. He et al. (2021). Masked Autoencoders Are Scalable Vision Learners. CVPR. https://arxiv.org/abs/2111.06377
3. He et al. (2016). Deep Residual Learning for Image Recognition. https://arxiv.org/abs/1512.03385
4. Dosovitskiy et al. (2020). An Image is Worth 16x16 Words. https://arxiv.org/abs/2010.11929
5. McInnes et al. (2018). UMAP. https://arxiv.org/abs/1802.03426
6. ROD Dataset. https://www.kaggle.com/datasets/abtinzandi/obstacle-detection-dataset
