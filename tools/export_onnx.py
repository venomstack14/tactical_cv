"""One-time, online step to get CPU weights:  pip install ultralytics && python -m tools.export_onnx   ->  yolov8n.onnx
Afterwards the system runs fully offline with:  python main.py clip.mp4 --calib c.json --weights yolov8n.onnx --detect-every 3"""
from ultralytics import YOLO
print(YOLO("yolov8n.pt").export(format="onnx", imgsz=640, simplify=True, opset=12))
