import cv2 # type: ignore
from ultralytics import YOLO # type: ignore

# Load model YOLOv8
# model = YOLO("yolov8m.pt")

# Load model yang sudah dilatih
model = YOLO(r"C:/Users/Administrator/Desktop/Skripsi/runs/detect/train10/weights/best.pt")

# Latih model dengan dataset table detection
# model.train(data="datasets/data.yaml", epochs=50, imgsz=640)

# Gunakan video atau kamera USB
video_source = "00.mp4"  # Ganti dengan "0" untuk kamera USB
cap = cv2.VideoCapture(video_source if isinstance(video_source, str) else int(video_source))

# Periksa apakah video/kamera dapat dibuka
if not cap.isOpened():
    print("Error: Tidak dapat membuka video/kamera.")
    exit()

# Ambil ukuran asli video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Buat jendela OpenCV agar tidak auto-resize
cv2.namedWindow("Deteksi Meja Kelas - YOLOv8", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Deteksi Meja Kelas - YOLOv8", width, height)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Hentikan jika video selesai atau kamera berhenti

    # Deteksi objek menggunakan YOLOv8
    results = model(frame)

    # Gambar bounding box di frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Kelas objek

            # Hanya tampilkan jika confidence > 0.5
            # if conf > 0.3:
            label = f"{model.names[cls]} {conf:.2f}"
            color = (0, 255, 0)  # Warna hijau untuk bounding box

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Tampilkan hasil deteksi dengan ukuran asli
    cv2.imshow("Deteksi Meja Kelas - YOLOv8", frame)

    # Keluar dengan menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
