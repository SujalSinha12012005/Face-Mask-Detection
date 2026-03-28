# 😷 Face Mask Detection

A real-time face mask detection system that identifies whether a person is wearing a face mask or not, built using OpenCV, MobileNetV2, and Streamlit — with zero TensorFlow dependency at runtime.

---

## 📌 Project Overview

This project addresses the problem of **public safety monitoring** by automatically detecting face mask compliance in images. It uses a pre-trained **MobileNetV2** deep learning model converted to **ONNX format** for fast and lightweight inference, paired with an **OpenCV SSD face detector** to locate faces in images.

The entire application is served through a clean and interactive **Streamlit** web interface, making it easy to use for anyone without any technical background.

---

## 🎯 Problem Statement

In public spaces such as hospitals, metro stations, malls, and schools, ensuring that individuals are wearing face masks is critical for health and safety. Manual monitoring is inefficient and error-prone. This project automates that process using AI.

---

## ✨ Features

- ✅ Detects faces in any uploaded image
- ✅ Classifies each face as **Mask On** or **No Mask**
- ✅ Shows **confidence score** for each detection
- ✅ Color coded results — green for mask, red for no mask
- ✅ Adjustable confidence threshold via sidebar slider
- ✅ Works on **Python 3.14** — no TensorFlow required
- ✅ Fast inference using ONNX Runtime
- ✅ Clean and responsive Streamlit UI

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Face Detection | OpenCV DNN (SSD ResNet) |
| Mask Classification | MobileNetV2 (ONNX Runtime) |
| Web Interface | Streamlit |
| Image Processing | OpenCV, Pillow |
| Model Format | ONNX (Open Neural Network Exchange) |
| Language | Python 3.14 |

---

## 📁 Project Structure

```
Face-Mask-Detection/
    ├── app.py                  # Main Streamlit application
    ├── mask_detector.onnx      # Trained MobileNetV2 mask model (ONNX)
    ├── mask_detector.model     # Original trained model (backup)
    ├── requirements.txt        # Python dependencies
    ├── face_detector/          # OpenCV face detection model files
    │     ├── deploy.prototxt
    │     └── res10_300x300_ssd_iter_140000.caffemodel
    ├── images/                 # Sample test images
    └── .venv/                  # Virtual environment
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/yourusername/Face-Mask-Detection.git
cd Face-Mask-Detection
```

### Step 2 — Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install streamlit opencv-python onnxruntime pillow
```

### Step 4 — Run the application
```bash
streamlit run app.py
```

### Step 5 — Open in browser
```
http://localhost:8501
```

---

## 🚀 How to Use

1. Open the app in your browser
2. Click **"Upload an image"**
3. Select any photo (JPG, JPEG, PNG)
4. Click **"🔍 Detect Mask"**
5. View detection results with bounding boxes and confidence scores

---

## 🧠 How It Works

```
Input Image
     ↓
OpenCV SSD Face Detector
(finds all faces in image)
     ↓
Each face cropped & resized to 224×224
     ↓
MobileNetV2 Mask Classifier (ONNX)
(predicts mask probability)
     ↓
Result: Mask On ✅ or No Mask ❌
+ Confidence Score %
```

---

## 📊 Detection Labels

| Label | Color | Meaning |
|---|---|---|
| ✅ Mask On | 🟢 Green | Person wearing mask correctly |
| ❌ No Mask | 🔴 Red | Person not wearing a mask |

---

## 📦 Dependencies

```
streamlit
opencv-python
onnxruntime
pillow
numpy
```

---

## 🔧 Model Details

| Property | Value |
|---|---|
| Base Model | MobileNetV2 |
| Input Size | 224 × 224 × 3 |
| Output Classes | 2 (Mask, No Mask) |
| Model Format | ONNX (opset 13) |
| Face Detector | OpenCV SSD ResNet |
| Face Detector Input | 300 × 300 |

---

## 🙋 Author

**Sujal Kumar Sinha**

Built with ❤️ using Python, OpenCV, ONNX Runtime and Streamlit.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).