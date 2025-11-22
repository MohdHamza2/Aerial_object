from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

results = model("test.jpg", save=True)

print("✔ Detection complete")
