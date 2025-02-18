from ultralytics import YOLO

# Load model YOLOv8 yang akan dilatih
model = YOLO("models/yolov8m.pt")  # Bisa diganti "yolov8m.pt" untuk akurasi lebih tinggi

# Latih model dengan dataset table detection
model.train(
    data="datasets/tables_detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device="cpu"  # Gunakan "cpu" jika tidak memiliki GPU
)

print("Training selesai! Model tersimpan di runs/detect/train/weights/best.pt")
