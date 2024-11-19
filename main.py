from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolo-Weights/yolov10n.pt')  # Load the YOLOv10 model

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

        # Run YOLOv10 inference
        results = model(img)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])  # Class index
                confidence = box.conf[0].item()  # Confidence score

                # Get class names dynamically
                class_name = model.names[cls] if model.names else f'Class {cls}'

                # Create detection dictionary
                detection = {
                    "class": class_name,
                    "confidence": confidence,
                    "x": int(box.xyxy[0][0]),
                    "y": int(box.xyxy[0][1]),
                    "width": int(box.xyxy[0][2] - box.xyxy[0][0]),
                    "height": int(box.xyxy[0][3] - box.xyxy[0][1])
                }
                detections.append(detection)

                # Draw bounding box on the image
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red color for bounding box

                # Prepare text label with class name and confidence
                label = f'{class_name} {confidence:.2f}'
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Ensure label is directly above the box
                label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10 + label_size[1]
                
                # Draw a filled rectangle as a background for the text
                cv2.rectangle(img, (x1, label_y - label_size[1] - 5), 
                              (x1 + label_size[0], label_y + 5), (0, 0, 255), -1)
                
                # Draw text on top of the background rectangle
                cv2.putText(img, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', img)
        processed_img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({"detections": detections, "image": processed_img_str})

    except Exception as e:
        print("Error processing frame:", e)
        return jsonify({"error": "Error processing frame"}), 500

if __name__ == "__main__":
    app.run(debug=True)