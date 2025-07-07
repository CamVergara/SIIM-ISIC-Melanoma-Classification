
# Melanoma Skin-Cancer Detection  
*Neural-Network Pipeline in Python – AUC 0.8693 *  

![Kaggle Badge](https://img.shields.io/badge/Kaggle-Top_7%25-blue)  

## 📑 Overview
This repository contains the full, reproducible codebase for a convolutional-neural-network (CNN) that detects melanoma lesions in dermoscopic images.  
The model achieved an **AUC of 0.8693**.

## ✨ Key Features
- **End-to-end pipeline**: data loading, preprocessing, augmentation, model training, inference, and submission file generation.  
- **Transfer learning** with EfficientNet-B3 for faster convergence on limited medical data.  
- **Cross-validation** (5-fold) with stratified sampling to handle class imbalance.  
- **Early stopping & LR scheduling** to prevent overfitting.  
- **Compliant by design**: code structure facilitates GDPR-ready audit trails.

## 📂 Dataset
- **Source**: [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/c/siim-isic-melanoma-classification)  
- **Images**: 33 126 dermoscopic JPEGs (train) + 10 982 (test)  
- **Labels**: Binary (1 = melanoma, 0 = benign)

Implemented in **PyTorch ≥ 2.0** with mixed-precision training (`torch.cuda.amp`) for speed.

## 📊 Results
| Fold | AUC |
|------|-----|
| 1    | 0.866 |
| 2    | 0.871 |
| 3    | 0.870 |
| 4    | 0.869 |
| 5    | 0.871 |
| **CV Mean** | **0.8693** |

The final ensemble beat the public-leaderboard baseline by **+4.7 pp**.

## 🚀 Quick Start
```bash
# 1. Clone repo & install deps
git clone https://github.com/your-handle/melanoma-cnn.git
cd melanoma-cnn
pip install -r requirements.txt

# 2. Download Kaggle dataset (API key required)
kaggle competitions download -c siim-isic-melanoma-classification -p data/

# 3. Train (single GPU)
python train.py --config configs/effnet_b3.yaml

# 4. Generate submission
python predict.py --checkpoint runs/best_model.pth
