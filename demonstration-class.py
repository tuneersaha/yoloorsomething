from ultralytics import YOLO

model = YOLO("yolo-Weights/yolov10n.pt")

class_names = model.names
print("Class names:", class_names)
