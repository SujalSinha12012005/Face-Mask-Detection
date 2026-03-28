import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import onnxruntime as ort

# -----------------------------------------------
# Page config
# -----------------------------------------------
st.set_page_config(
    page_title='Face Mask Detector',
    page_icon='😷',
    layout='centered',
    initial_sidebar_state='expanded'
)

# -----------------------------------------------
# Custom CSS
# -----------------------------------------------
st.markdown("""
<style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: gray;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 0.8rem;
    }
    .with-mask    { background-color: #d4edda; color: #155724; }
    .without-mask { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# Load face detector (OpenCV SSD)
# -----------------------------------------------
@st.cache_resource
def load_face_detector():
    prototxt = os.path.join("face_detector", "deploy.prototxt")
    weights  = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNet(prototxt, weights)
    return net

# -----------------------------------------------
# Load ONNX mask classifier directly
# -----------------------------------------------
@st.cache_resource
def load_mask_model():
    session = ort.InferenceSession("mask_detector.onnx")
    return session

# -----------------------------------------------
# Preprocess face for MobileNetV2
# -----------------------------------------------
def preprocess_face(face_img):
    face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = face.astype("float32")
    face = face / 127.5 - 1.0          # MobileNetV2 normalization
    face = np.expand_dims(face, axis=0)
    return face

# -----------------------------------------------
# Main detection function
# -----------------------------------------------
def detect_masks(image_path, conf_threshold=0.5):
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # Step 1: Detect faces
    net = load_face_detector()
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # Step 2: Load mask model
    session = load_mask_model()
    input_name = session.get_inputs()[0].name

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY)     = (min(w - 1, endX), min(h - 1, endY))

            face_roi = image[startY:endY, startX:endX]
            if face_roi.size == 0:
                continue

            # Step 3: Run through ONNX mask model
            face_input = preprocess_face(face_roi)
            preds = session.run(None, {input_name: face_input})[0][0]

            mask_prob         = float(preds[0])
            without_mask_prob = float(preds[1])

            label = "Mask" if mask_prob > without_mask_prob else "No Mask"
            prob  = max(mask_prob, without_mask_prob)
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # Draw on image
            display_label = "{}: {:.2f}%".format(label, prob * 100)
            cv2.putText(image, display_label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            results.append((label, prob))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb, results

# -----------------------------------------------
# Sidebar
# -----------------------------------------------
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.markdown("---")
st.sidebar.success("✅ Using trained mask model!")
conf_threshold = st.sidebar.slider(
    "Face detection confidence",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="How confident the face detector needs to be"
)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with **OpenCV** + **ONNX** + **Streamlit**")

# -----------------------------------------------
# Main UI
# -----------------------------------------------
st.markdown('<div class="main-title">😷 Face Mask Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a photo to detect mask usage</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📷 Image Detection", "ℹ️ About"])

# -----------------------------------------------
# Tab 1 — Image Detection
# -----------------------------------------------
with tab1:
    image_file = st.file_uploader(
        "Upload an image",
        type=['jpg', 'jpeg', 'png'],
        help="Supports JPG, JPEG, PNG"
    )

    if image_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📸 Original Image**")
            st.image(image_file, use_container_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(image_file.read())
            tmp_path = tmp.name

        if st.button('🔍 Detect Mask', use_container_width=True, type="primary"):
            with st.spinner("Analysing image..."):
                result_img, detections = detect_masks(tmp_path, conf_threshold)

            with col2:
                st.markdown("**🎯 Detection Result**")
                st.image(result_img, use_container_width=True)

            st.markdown("### 📋 Detection Summary")

            if not detections:
                st.info("No faces detected. Try lowering the confidence threshold.")
            else:
                for i, (label, prob) in enumerate(detections):
                    if label == "Mask":
                        css_class = "with-mask"
                        icon      = "✅"
                        display   = "Mask On"
                    else:
                        css_class = "without-mask"
                        icon      = "❌"
                        display   = "No Mask"

                    st.markdown(
                        f'<div class="result-box {css_class}">'
                        f'{icon} Face {i+1}: {display} — {prob*100:.1f}% confident'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.success("✅ Detection complete!")

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# -----------------------------------------------
# Tab 2 — About
# -----------------------------------------------
with tab2:
    st.markdown("""
    ### About this app
    This app uses a trained **MobileNetV2 mask classifier** 
    converted to ONNX format — no TensorFlow needed!

    ### How it works
    | Step | What happens |
    |---|---|
    | 1 | OpenCV SSD finds all faces in image |
    | 2 | Each face is cropped out |
    | 3 | Face passed to MobileNetV2 mask model |
    | 4 | Model outputs Mask or No Mask |

    ### Detection labels
    | Label | Meaning |
    |---|---|
    | ✅ Mask On | Person wearing mask correctly |
    | ❌ No Mask | Person not wearing a mask |

    ### Confidence score
    | Score | Meaning |
    |---|---|
    | > 90% | Very confident |
    | 70–90% | Good |
    | 50–70% | Uncertain |
    | < 50% | Likely wrong |
    """)
