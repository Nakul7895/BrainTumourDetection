# 🧠 Brain Tumor Detection with Explainable AI

A deep learning–based application for automated brain tumor detection from MRI images, enhanced with explainable artificial intelligence (Grad-CAM) to provide visual interpretation of model decisions. The system is built using EfficientNet-B0 and deployed through a professional, hospital-grade Streamlit interface.

---

## 📌 Project Overview

Brain tumor detection from MRI scans is a critical task in medical diagnostics. Manual analysis requires expert radiologists and can be time-consuming. This project aims to assist medical professionals by providing an AI-powered decision support system that not only predicts the presence of a tumor but also explains *why* the model made that decision.

The system classifies MRI images into:
- **Tumor Detected**
- **No Tumor Detected**

and highlights the most influential regions using Grad-CAM.

---

## 🎯 Objectives

- Automate brain tumor detection using deep learning
- Achieve high classification accuracy using transfer learning
- Provide explainable AI visualizations for transparency
- Develop a clean, professional, medical-grade user interface
- Support academic and research use cases

---

## ✨ Key Features

- 🧠 EfficientNet-B0 based CNN classifier  
- 🔍 Grad-CAM based explainable AI visualization  
- 📊 Confidence score for predictions  
- 🎨 Dark, professional hospital-style UI  
- ⚡ Fast inference on CPU  
- 📁 Simple image upload and analysis  

---

## 🏗️ System Architecture

MRI Image → Preprocessing → EfficientNet-B0 → Prediction → Grad-CAM → Streamlit UI

---

## 🧪 Model Details

### Architecture
- EfficientNet-B0
- Pretrained on ImageNet
- Fine-tuned for binary classification

### Training
- Transfer Learning
- Loss Function: Cross Entropy Loss
- Optimizer: Adam
- Metric: Accuracy

### Explainability
- Grad-CAM highlights spatial regions influencing predictions

---

## 📁 Project Structure

brain_tumor_project/
│
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── classification/
│       └── brain_tumor_model.pth
├── utils/
│   └── gradcam.py
├── requirements.txt
└── README.md

---

## 📊 Dataset

Brain MRI images categorized as:
- yes → Tumor present
- no → No tumor

Used strictly for academic and research purposes.

---

## 🚀 Installation & Execution

1. Create virtual environment
2. Install dependencies
3. Run Streamlit app

---

## 📈 Results

- Accuracy achieved: ~85–90%
- Stable performance on unseen data
- Effective visual explanations using Grad-CAM

---

## ⚠️ Limitations

- Not clinically validated
- Works on 2D MRI images only
- Not a replacement for professional diagnosis

---

## 🔮 Future Enhancements

- Multi-class tumor detection
- 3D MRI analysis
- FastAPI + React frontend
- Automated diagnostic reports

---

## 🎓 Academic Declaration

This project is developed solely for educational and research purposes and is not intended for clinical use.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- Streamlit
- NumPy
- PIL

---

## 📜 License

Academic use only. Commercial use requires permission.

---

⭐ If you found this project useful, consider giving it a star!
