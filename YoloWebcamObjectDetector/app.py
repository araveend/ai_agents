import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import Counter
import pandas as pd
st.set_page_config(page_title="YOLOv8 Real-Time Object Detection", layout="centered")

# --- Model Loading ---
# IMPORTANT: Ensure "yolo11x.pt" exists in your project directory
# or change it to a standard YOLOv8 model like "yolov8n.pt" which will download automatically.
try:
    model = YOLO("yolov8n.pt")
    st.success("YOLOv8 model loaded successfully!")
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.warning("Please ensure 'yolov8n.pt' is in the same directory, or try a standard model like 'yolov8n.pt'.")
    st.stop() # Stop the app if the model can't be loaded

# --- Streamlit Page Configuration ---

st.title("🎥 YOLOv8 Real-Time Object Detection")
st.write("This application uses your webcam to perform live object detection with YOLOv8.")

# --- Sidebar Controls ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.01)
# Note: selected_device parameter in webrtc_streamer is often buggy or
# not consistently supported across browsers/OS. Defaulting to True for video.
# selected_device = st.sidebar.selectbox("Camera Source", options=[0, 1, 2], index=0)

# --- Video Processor Class ---
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, confidence):
        self.confidence = confidence
        self.object_names = []  # List to store names of objects detected in the current frame
        self.prev_time = time.time()
        self.fps = 0.0
        self.detection_log = [] # Stores history of detections

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.prev_time
        self.prev_time = current_time
        self.fps = 1 / elapsed if elapsed > 0 else 0

        # Perform detection
        results = model.predict(img, conf=self.confidence, verbose=False)[0]
        self.object_names = []  # Reset for each new frame

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            self.object_names.append(model.names[cls]) # Add detected object name to the list

            # Draw bounding box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        # Log detections (you might want to log only unique objects or objects present for a duration)
        if self.object_names: # Log only if objects were detected
            self.detection_log.append({"timestamp": current_time, "objects": self.object_names})

        # Show FPS on frame
        cv2.putText(img, f"FPS: {self.fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- WebRTC Streamer Integration ---
st.subheader("Webcam Feed")
ctx = webrtc_streamer(
    key="yolo-object-detection",
    video_processor_factory=lambda: YOLOVideoProcessor(confidence=confidence_threshold),
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False}, # Ensure video is enabled
    video_html_attrs={"autoPlay": True, "controls": False, "style": {"width": "100%"}}, # Auto-play and style
)

# --- Real-time Detected Object Display & Information ---
st.write("---")
st.subheader("Detection Information")

if ctx.state.playing:
    # Access the video_processor instance directly
    if ctx.video_processor:
        st.markdown(f"### 🟢 Camera Active | 🔁 FPS: **{ctx.video_processor.fps:.2f}**")

        # Get unique detected objects for the current frame
        unique_detected_objects = sorted(list(set(ctx.video_processor.object_names)))
        if unique_detected_objects:
            st.markdown(f"### ✨ **Currently Detected:** `{', '.join(unique_detected_objects)}`")
        else:
            st.markdown("### 🔍 No objects detected in the current frame.")
    else:
        st.markdown("### 🟡 Camera Ready – Waiting for video stream...")
else:
    st.markdown("### 🔴 Camera Not Started. Click 'Start' above to begin detection.")

# --- Live Detection Table ---
if ctx.video_processor and ctx.video_processor.object_names:
    st.markdown("---")
    st.markdown("### 📋 Detected Objects Table (Current Frame)")
    counts = Counter(ctx.video_processor.object_names)
    df = pd.DataFrame(counts.items(), columns=["Object", "Count"])
    st.dataframe(df, use_container_width=True, hide_index=True)
elif ctx.state.playing:
     st.markdown("---")
     st.info("No objects detected yet in the current frame. Adjust confidence or wait for objects to appear.")

# --- Detection History Log ---
if ctx.video_processor and ctx.video_processor.detection_log:
    st.markdown("---")
    st.markdown("### 🗂️ Recent Detection Log")
    # Display up to the last 10 entries for better readability
    recent = ctx.video_processor.detection_log[-10:]
    # Reverse to show most recent at the top
    for entry in reversed(recent):
        if "objects" in entry and isinstance(entry["objects"], list) and entry["objects"]:
            object_list = ", ".join(entry["objects"])
            st.write(f"🕒 {time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}: {object_list}")
        else:
            st.write(f"🕒 {time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}: No objects detected in this log entry.")
elif ctx.state.playing:
    st.markdown("---")
    st.info("Detection log will appear here as objects are detected.")
else:
    st.markdown("---")
    st.info("Start the camera to begin logging detections.")

st.markdown("---")
st.markdown("This app is powered by Streamlit, `streamlit-webrtc`, and YOLOv8.")