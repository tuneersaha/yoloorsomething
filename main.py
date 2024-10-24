from flask import Flask, render_template, Response, request, jsonify
import cv2
import base64
from ultralytics import YOLO
import math
import numpy as np

app = Flask(__name__)

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Class names for the YOLO model
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Receive the image data from the client
    data = request.json['image']
    img_data = base64.b64decode(data.split(',')[1])
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run YOLO object detection
    results = model(img, stream=True)
    
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            confidence = math.ceil((box.conf[0] * 100)) / 100 
            detections.append({
                "class": classNames[cls],
                "confidence": confidence
            })

    # Send detected class names and confidence back to client
    return jsonify(detections)

if __name__ == "__main__":
    app.run(debug=True)
