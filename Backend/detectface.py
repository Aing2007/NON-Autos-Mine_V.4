import cv2
from ultralytics import YOLO

# =========================
# โหลดโมเดล
# =========================
model = YOLO("detectface.pt")

# =========================
# เปิดวิดีโอ input
# =========================
video_path = "/Users/sutinan/Documents/NON-Autos-Mine_V.4/Video/VDO3.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("ไม่สามารถเปิดวิดีโอได้")

# =========================
# ตั้งค่า Video Writer (output)
# =========================
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# =========================
# Loop อ่าน frame ทีละเฟรม
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # ทำ Detection
    # =========================
    results = model(frame)

    # =========================
    # วาดกรอบ (bounding box) — กรองด้วย confidence threshold
    # =========================
    conf_threshold = 0.6
    annotated_frame = frame.copy()

    if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
        try:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
        except Exception:
            boxes = []
            confs = []
            classes = []

        for i, conf in enumerate(confs):
            if conf >= conf_threshold:
                x1, y1, x2, y2 = boxes[i].astype(int)
                cls_id = int(classes[i]) if len(classes) > i else -1
                try:
                    class_name = model.names[cls_id]
                except Exception:
                    class_name = str(cls_id)

                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # =========================
    # เขียนลงไฟล์ output
    # =========================
    out.write(annotated_frame)

    # (optional) แสดงผล realtime
    cv2.imshow("Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
        break

# =========================
# ปิดทุกอย่าง
# =========================
cap.release()
out.release()
cv2.destroyAllWindows()

print("เสร็จแล้ว! บันทึกเป็น output.mp4")