# 🚦 TraffiQ - Smart Traffic Video Analysis

![TraffiQ Logo](https://img.icons8.com/color/96/traffic-jam.png)

**TraffiQ** is an AI-powered traffic video analysis tool that detects vehicles, estimates their speed, and identifies helmet violations — all through a user-friendly web interface built with Streamlit and powered by YOLOv8.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-HuggingFace-blue?logo=huggingface&style=for-the-badge)](https://huggingface.co/spaces/Sofiaakhtar/TraffiQ)

---

## 🔍 Features

- 🚗 **Vehicle Detection**: Detects cars, bikes, trucks, buses, and autos using YOLOv8.
- 🕒 **Speed Estimation**: Calculates vehicle speed using distance moved per frame.
- 🪖 **Helmet Violation Detection**: Flags people riding bikes without helmets.
- 📹 **Streamlit Web UI**: Upload videos, run analysis, and download annotated results.

---

## 🌐 Try It Online

👉 **Live Demo**: [https://huggingface.co/spaces/Sofiaakhtar/TraffiQ](https://huggingface.co/spaces/Sofiaakhtar/TraffiQ)  
No setup required — just upload a traffic video and get results instantly!

---

## 🧠 How It Works

1. **Upload a video** through the Streamlit interface.
2. **YOLOv8** detects and classifies vehicles/persons per frame.
3. **Speed is estimated** by tracking object position over time.
4. **Helmet violations** are checked using IoU between detected heads and helmets.
5. **Annotated output** is generated with bounding boxes and labels.
6. **Download** the result directly from the app.

---

## 🛠️ Technologies Used

- Python
- Streamlit
- OpenCV
- YOLOv8 (Ultralytics)
- Hugging Face Spaces

---

## 📦 Local Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/traffiq.git
cd traffiq

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py

