import streamlit as st
import time
from PIL import Image
import numpy as np
import torch
import cv2
import os
import sys
from torchvision import models, transforms

# Allow imports from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.gradcam import GradCAM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Brain Tumor Detection | Explainable AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- DARK AESTHETIC CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5e7eb;
}

.main-header {
    background: linear-gradient(90deg, #0f172a, #020617);
    padding: 1.8rem 2.5rem;
    border-radius: 0.75rem;
    margin-bottom: 2.5rem;
    border: 1px solid #1e293b;
}

.main-header h1 {
    margin: 0;
    font-size: 2.1rem;
    font-weight: 600;
    color: #f8fafc;
}

.main-header p {
    margin-top: 0.5rem;
    color: #94a3b8;
    font-size: 0.95rem;
}

.custom-card {
    background: #020617;
    border-radius: 0.75rem;
    padding: 1.6rem;
    border: 1px solid #1e293b;
    margin-bottom: 1.2rem;
}

.upload-card {
    border: 2px dashed #334155;
    background-color: #020617;
    padding: 2rem;
    border-radius: 0.75rem;
    text-align: center;
    color: #94a3b8;
}

.result-positive {
    background-color: #190f0f;
    border: 1px solid #7f1d1d;
    padding: 1.4rem;
    border-radius: 0.75rem;
}

.result-negative {
    background-color: #0f172a;
    border: 1px solid #14532d;
    padding: 1.4rem;
    border-radius: 0.75rem;
}

.result-waiting {
    background-color: #020617;
    border: 2px dashed #334155;
    padding: 3rem;
    border-radius: 0.75rem;
    text-align: center;
    color: #64748b;
}

.info-box {
    background-color: #020617;
    border: 1px solid #1e3a8a;
    padding: 1rem;
    border-radius: 0.5rem;
    font-size: 0.85rem;
    color: #bfdbfe;
}

.footer {
    text-align: center;
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 3rem;
    border-top: 1px solid #1e293b;
    padding-top: 2rem;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: #ffffff;
    font-weight: 600;
    padding: 0.75rem;
    border-radius: 0.5rem;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #1e40af);
}


#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-header">
    <h1>🧠 Brain Tumor Detection with Explainable AI</h1>
    <p>Clinical-grade MRI analysis using deep learning and visual explanations</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_and_tools():
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features, 2
    )
    model.load_state_dict(
        torch.load(
            "models/classification/brain_tumor_model.pth",
            map_location="cpu"
        )
    )
    model.eval()

    gradcam = GradCAM(model, model.features[-1])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return model, gradcam, transform


model, gradcam, transform = load_model_and_tools()

# ---------------- REAL PREDICTION ----------------
def predict_tumor(image):
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        confidence, cls = torch.max(probs, 1)

    return {
        "has_tumor": cls.item() == 1,
        "confidence": int(confidence.item() * 100),
        "model_version": "EfficientNet-B0 (PyTorch)"
    }

# ---------------- REAL GRAD-CAM ----------------
def generate_gradcam_overlay(image):
    tensor = transform(image).unsqueeze(0)
    output = model(tensor)
    class_idx = torch.argmax(output).item()

    cam = gradcam.generate(tensor, class_idx)
    cam = cv2.resize(cam, image.size)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    image_np = np.array(image)
    overlay = cv2.addWeighted(
        image_np, 0.6,
        heatmap, 0.4,
        0
    )

    return Image.fromarray(overlay)

# ---------------- MAIN LAYOUT ----------------
col1, col2 = st.columns(2)

# LEFT: UPLOAD
with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 📤 MRI Image Upload")

    uploaded_file = st.file_uploader(
        "Upload MRI scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.uploaded_image = image
        st.image(image, caption="Uploaded MRI Scan", width=420)
        st.success("✓ Image uploaded successfully")

    else:
        st.markdown("""
        <div class="upload-card">
            <p>Upload a brain MRI scan to begin analysis</p>
            <p style="font-size:0.8rem;">Supported formats: JPG, PNG</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        if st.button(" Analyze MRI Scan"):
            with st.spinner("Analyzing MRI scan using deep learning..."):
                time.sleep(1)
                st.session_state.prediction = predict_tumor(image)
                st.session_state.analyzed = True
                st.rerun()

# RIGHT: RESULT
with col2:
    if st.session_state.analyzed and st.session_state.prediction:
        pred = st.session_state.prediction

        result_class = "result-positive" if pred["has_tumor"] else "result-negative"
        status_text = "Tumor Detected" if pred["has_tumor"] else "No Tumor Detected"

        st.markdown(f"""
        <div class="{result_class}">
            <h3>{status_text}</h3>
            <p><strong>Confidence:</strong> {pred["confidence"]}%</p>
            <p><strong>Model:</strong> {pred["model_version"]}</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(pred["confidence"] / 100)

        st.markdown("""
        <div class="info-box">
            ℹ️ This AI-assisted result is for research and educational use only.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="result-waiting">
            Upload an MRI scan to view diagnostic results
        </div>
        """, unsafe_allow_html=True)

# ---------------- EXPLAINABILITY ----------------
if st.session_state.analyzed and st.session_state.uploaded_image:
    st.markdown("---")
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("### 🔍 AI Explainability (Grad-CAM)")

    overlay = generate_gradcam_overlay(st.session_state.uploaded_image)

    c1, c2 = st.columns(2)
    with c1:
        st.image(
            st.session_state.uploaded_image,
            caption="Original MRI",
            width=420
        )
    with c2:
        st.image(
            overlay,
            caption="Grad-CAM Highlighted Regions",
            width=420
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    <strong>EfficientNet-B0</strong> • Grad-CAM • PyTorch + Streamlit<br/>
    For research and educational purposes only. Not for clinical diagnosis.
</div>
""", unsafe_allow_html=True)
