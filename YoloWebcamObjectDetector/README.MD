YOLOv8 Real-Time Object Detection with Streamlit
This project provides a Streamlit web application for real-time object detection using the YOLOv8 deep learning model. It leverages your webcam feed to perform live object recognition and displays the results directly in your browser.

🚀 Features
Real-Time Detection: Processes video frames from your webcam in real time.

YOLOv8 Integration: Utilizes a pre-trained YOLOv8 model for accurate and efficient object detection.

Customizable Confidence: Adjust the detection confidence threshold using a sidebar slider.

Live Metrics: Displays Frames Per Second (FPS) of the processing.

Detected Objects Table: Shows a summary of currently detected objects and their counts.

Recent Detections Log: Keeps a log of recently detected objects with timestamps.

Pure Python: Built entirely with Python libraries, requiring no complex frontend development.

🛠️ Requirements
Before running the application, ensure you have the following installed:

Python 3.x

Streamlit (streamlit)

Streamlit WebRTC (streamlit-webrtc)

OpenCV Python (opencv-python)

NumPy (numpy)

Ultralytics (ultralytics) - for YOLOv8 model

Pandas (pandas)

AV (av) - dependency for streamlit-webrtc

You can install all necessary dependencies using pip:

pip install streamlit streamlit-webrtc opencv-python numpy ultralytics pandas av

📦 Setup and Usage
Save the Code:
Save the provided Python code into a file named, for example, app.py.

YOLOv8 Model:
The code loads a YOLOv8 model. By default, it's set to "yolo11x.pt".

If "yolo11x.pt" is a custom model you have, ensure it's in the same directory as your app.py file.

If you intend to use a standard pre-trained YOLOv8 model (like yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, or yolov8x.pt), the ultralytics library will automatically download it the first time you run the app. You might want to change model = YOLO("yolo11x.pt") to model = YOLO("yolov8x.pt") if "yolo11x.pt" isn't a valid model file.

Run the Streamlit App:
Open your terminal or command prompt, navigate to the directory where you saved app.py, and run:

streamlit run app.py

What happens when you run it:
A new tab will open in your default web browser, displaying the Streamlit application.

The application will prompt you for webcam access. Allow it to proceed.

Once access is granted, you will see a live video feed from your webcam.

The YOLOv8 model will start detecting objects in real-time, displaying bounding boxes and labels on the video feed.

The sidebar allows you to adjust the confidence threshold for detections.

Below the video feed, you'll see live FPS, a table of detected objects, and a log of recent detections.

🧠 How It Works
This application integrates several powerful components:

Streamlit: Provides the interactive web interface, handling user input (sliders, select boxes) and displaying outputs (video, tables, logs).

streamlit-webrtc: Facilitates real-time video streaming from the user's webcam to the Streamlit backend. It wraps WebRTC technology, enabling direct peer-to-peer media communication.

YOLOVideoProcessor Class: This is a custom class that inherits from streamlit-webrtc's VideoProcessorBase.

For each incoming video frame from the webcam, the recv method of this class is called.

Inside recv, the frame is passed to the YOLOv8 model (loaded via ultralytics).

The YOLOv8 model performs object detection, returning bounding box coordinates, confidence scores, and class IDs.

OpenCV (cv2): Is used to draw the bounding boxes and labels directly onto the video frame. It also calculates and displays the Frames Per Second (FPS).

The processed frame is then sent back to Streamlit for display in the browser, creating the real-time effect.

Data Display: The application dynamically updates tables and logs in the Streamlit UI by accessing the object_names, fps, and detection_log attributes of the YOLOVideoProcessor instance.

⚠️ Notes
Model Performance: The performance (speed and accuracy) will depend on your system's hardware, webcam resolution, and the specific YOLOv8 model variant (n, s, m, l, x) you choose. Larger models (e.g., yolov8x.pt) are more accurate but slower.

Network Connection: As streamlit-webrtc uses WebRTC, a stable internet connection is beneficial, especially for establishing the initial peer connection (STUN server).

