🧠 Face & Eye Detection using Haar Cascade Classifier (OpenCV – Python)

This project demonstrates real-time face and eye detection using Haar Cascade Classifiers in OpenCV. Haar Cascades are fast, efficient, and widely used for object detection tasks—especially in classical computer vision systems.

🚀 Overview

The project loads pre-trained Haar Cascade XML classifiers to detect faces and eyes in images. Detected regions are highlighted using bounding boxes. This implementation follows a simple and modular structure, making it easy to extend or integrate into other AI/ML projects.

📌 Features

👤 Face Detection using haarcascade_frontalface_default.xml

👀 Eye Detection using haarcascade_eye.xml

🖼️ Works on static images (can be extended to webcam video stream)

💾 Automatically saves output images with detected features

🔧 Clean and modular detection functions

📊 Built with Python, OpenCV, NumPy, and Matplotlib

📂 Project Structure
├── haarcascade_frontalface_default.xml
├── haarcascade_eye.xml
├── andrew.jpg
├── face.jpg
├── eyes.jpg
├── face+eyes.jpg
└── face_detection.py

🧩 How It Works
1️⃣ Load Required Libraries

OpenCV
 for image processing

NumPy
 for array operations

Matplotlib
 for visualization

2️⃣ Load Haar Cascade Models

These XML files contain pre-trained feature classifiers.

3️⃣ Detect Faces

A function identifies all face regions and draws rectangles around them.

4️⃣ Detect Eyes

A similar function highlights eyes using the eye cascade classifier.

5️⃣ Visualize Results

The script displays and saves:

face.jpg → face detection only

eyes.jpg → eye detection only

face+eyes.jpg → combined detection

🖥️ Demo Code Snippet
import cv2
import numpy as np
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def adjusted_detect_face(img):
    face_img = img.copy()
    face_rect = face_cascade.detectMultiScale(face_img, 1.2, 5)
    for (x, y, w, h) in face_rect:
        cv2.rectangle(face_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return face_img

def detect_eyes(img):
    eye_img = img.copy()
    eye_rect = eye_cascade.detectMultiScale(eye_img, 1.2, 5)
    for (x, y, w, h) in eye_rect:
        cv2.rectangle(eye_img, (x, y), (x + w, y + h), (255, 255, 255), 10)
    return eye_img

🛠️ Requirements

Install dependencies with:

pip install opencv-python numpy matplotlib

▶️ Running the Project
python face_detection.py


Make sure the image and XML files are in the same directory.

📦 Future Improvements

Add real-time detection using webcam (OpenCV Video Capture
)

Improve accuracy using Deep Learning (DNN Face Detector, MTCNN
, YOLO
)

Build a simple UI using Tkinter
 or Streamlit

Create REST API using FastAPI
 for remote detection

🤝 Contributions

Feel free to submit issues, fork the repo, and send pull requests!

📄 License

This project is licensed under the MIT License.
