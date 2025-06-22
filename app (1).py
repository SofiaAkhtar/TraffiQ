import streamlit as st
import tempfile
import cv2
import math
from ultralytics import YOLO

# --- Constants ---
FPS = 30
PIXELS_PER_METER = 10
model = YOLO("yolov8n.pt")  # Load your model

# --- Function to Estimate Speed ---
def estimate_speed(p1, p2, fps):
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    meters = dist / PIXELS_PER_METER
    speed = meters * fps * 3.6
    return round(speed, 2)

# --- Video Processing Function ---
def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or FPS
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    track_data = {}

    vehicle_classes = ["car", "motorbike", "bus", "truck", "auto"]
    vehicle_colors = {
        "car": (255, 0, 255),
        "motorbike": (255, 0, 0),
        "bus": (0, 0, 255),
        "truck": (0, 125, 255),
        "auto": (255, 125, 0)
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame, conf=0.5, verbose=False)[0]

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if class_name in vehicle_classes:
                color = vehicle_colors.get(class_name, (255, 255, 255))
                obj_id = f"{class_name}_{i}"
                prev_center = track_data.get(obj_id, (center, 0))[0]
                speed = estimate_speed(prev_center, center, fps) if frame_count > 1 else 0
                track_data[obj_id] = (center, speed)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f"{class_name} {speed} km/h"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            elif class_name == "person":
                person_box = box.xyxy[0]
                has_helmet = False

                for other in results.boxes:
                    other_cls = int(other.cls[0])
                    other_name = model.names[other_cls]
                    if other_name == "helmet":
                        helmet_box = other.xyxy[0]
                        iou = (
                            max(0, min(person_box[2], helmet_box[2]) - max(person_box[0], helmet_box[0])) *
                            max(0, min(person_box[3], helmet_box[3]) - max(person_box[1], helmet_box[1]))
                        )
                        if iou > 0:
                            has_helmet = True
                            break

                if not has_helmet:
                    cv2.putText(frame, "no Helmet", (int(person_box[0]), int(person_box[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.rectangle(frame, (int(person_box[0]), int(person_box[1])),
                                  (int(person_box[2]), int(person_box[3])), (0, 0, 255), 3)

        out.write(frame)

    cap.release()
    out.release()

# --- Streamlit UI ---
# --- Page Configuration ---
st.set_page_config(
    page_title="TraffiQ - Smart Traffic Analysis",
    page_icon="https://img.icons8.com/color/96/traffic-jam.png",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1a1a1a;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            font-size: 1.2rem;
            color: #4b5563;
            margin-bottom: 2rem;
        }
        .section {
            background-color: white;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0px 0px 8px rgba(0,0,0,0.05);
            margin-bottom: 2rem;
        }
        .upload-label {
            font-size: 1.1rem;
            color: #374151;
            font-weight: 600;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://img.icons8.com/color/96/traffic-jam.png", width=60)
with col2:
    st.markdown('<div class="main-title">TraffiQ - Smart Traffic Video Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detect vehicles, estimate speed, and monitor helmet usage from uploaded videos.</div>', unsafe_allow_html=True)

# --- Upload Section ---
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="upload-label">Upload a Traffic Video</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
        tmp_input.write(uploaded_file.read())
        input_path = tmp_input.name

    output_path = input_path.replace(".mp4", "_output.mp4")

    with st.spinner("🔍 Processing video... Please wait."):

        process_video(input_path, output_path)

    st.success("✅ Analysis complete! Preview and download your results below.")

    # Enable video preview
    # st.video(output_path)

    with open(output_path, "rb") as file:
        st.download_button(
            label="📥 Download Annotated Video",
            data=file,
            file_name="traffiq_output.mp4",
            mime="video/mp4"
        )

st.markdown('</div>', unsafe_allow_html=True)
