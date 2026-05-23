import cv2
from ultralytics import YOLO

# =========================
# โหลดโมเดล
# =========================
model = YOLO("detectface.pt")

# =========================
# เปิดวิดีโอ input
# =========================
video_path = "/Users/sutinan/Documents/NON-Autos Mine_NEWV/Video/VDO3.mp4"
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
    # วาดกรอบ (bounding box)
    # =========================
    annotated_frame = results[0].plot()

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