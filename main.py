from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolo-Weights/yolov10n.pt') 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        results = model(img)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])  
                confidence = box.conf[0].item()  
                class_name = model.names[cls] if model.names else f'Class {cls}'

                detection = {
                    "class": class_name,
                    "confidence": confidence,
                    "x": int(box.xyxy[0][0]),
                    "y": int(box.xyxy[0][1]),
                    "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
                    "height": int(box.xyxy[0][3] - box.xyxy[0][1])
                }
                detections.append(detection)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', img)
        processed_img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"detections": detections, "image": processed_img_str})

    except Exception as e:
        print("Error processing frame:", e)
        return jsonify({"error": "Error processing frame"}), 500

if __name__ == "__main__":
    app.run(debug=True)
q