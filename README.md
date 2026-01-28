# Skin Lesion Segmentation using DFF-UNet

## 📌 Project Overview
This project implements a Deep Feature Fusion U-Net (DFF-UNet) model for accurate segmentation of skin lesion regions from dermoscopic images. The system is designed as part of a medical image analysis pipeline to assist early melanoma detection.

## 🎯 Objective
To develop a deep learning–based segmentation model that precisely identifies lesion boundaries in ISIC 2018 dermoscopic images.

## 🧠 Model Architecture
The project uses a **DFF-UNet** architecture that enhances segmentation performance by fusing deep and shallow feature maps across encoder–decoder layers.

## ⚙️ Project Workflow
1. Preprocessing ISIC 2018 dermoscopic images  
2. Training the DFF-UNet segmentation model  
3. Generating predicted segmentation masks  
4. Evaluating performance using segmentation metrics  
5. Visualizing predicted masks alongside ground truth masks

## 📊 Evaluation Metrics (Phase 1)

- **Dice Similarity Coefficient (DSC)** – Measures overlap between predicted and ground truth masks  
- **Intersection over Union (IoU / Jaccard Index)** – Measures region similarity  
- **Pixel Accuracy** – Percentage of correctly classified pixels  
- **Precision** – Correctly predicted lesion pixels over total predicted lesion pixels  
- **Recall (Sensitivity)** – Correctly detected lesion pixels over actual lesion pixels  
- **F1-Score** – Harmonic mean of Precision and Recall  

## 🖼 Results
Sample segmentation outputs are available in the `results/` folder showing:  
**Original Image | Ground Truth Mask | Predicted Mask**

## 📁 Dataset
ISIC 2018 Skin Lesion Analysis Dataset (Dermoscopic Images)

## 🚫 Model Weights
The trained model weights (.pth file) are not included in this repository due to GitHub file size limits.  
**Trained weights can be downloaded here:**  https://drive.google.com/drive/folders/16ehyyXVXXR2PtXLYUhbmiZR-SEhG7mVh?usp=drive_link

## 💻 Technologies Used
Python, PyTorch, NumPy, OpenCV, Matplotlib
