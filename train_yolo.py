import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # Fix Windows OMP crash

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="yolo_data.yaml",
    epochs=20,
    imgsz=320,
    batch=4,
    workers=0,
    device="cpu"
)
