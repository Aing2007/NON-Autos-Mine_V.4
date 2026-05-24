import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.websockets import WebSocket
from ultralytics import YOLO

# ======================================================
# FastAPI App
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# Load 3 YOLO Models
# ======================================================
print("=" * 50)
print("Loading models...")

model_human = YOLO("yolo11n.pt")
model_face  = YOLO("detectface.pt")
model_pose  = YOLO("yolo11n-pose.pt")

print("✅ All 3 models loaded")
print("=" * 50)

# ======================================================
# Confidence Thresholds — แก้ได้ที่นี่ที่เดียว
#
#   "human"    : ค่าต่ำ → จับได้ไวแต่ false positive เยอะ
#   "face"     : ค่าสูง → face ที่ชัวร์เท่านั้น
#   "pose"     : threshold ของ bounding box คน
#   "keypoint" : ส่งให้ frontend ใช้กรองจุดที่ไม่ชัวร์ออก
# ======================================================
CONF = { #ปรับความมั่นใจขั้นต่ำสุดที่ต้องการสำหรับแต่ละโมเดล
    "human":    0.80,
    "face":     0.75,
    "pose":     0.60,
    "keypoint": 0.30,
}

# ThreadPoolExecutor: 3 workers = 3 โมเดลพร้อมกัน
executor = ThreadPoolExecutor(max_workers=3)


# ======================================================
# Model inference functions
# ======================================================
def infer_human(frame: np.ndarray) -> list:
    results = model_human.track(
        frame,
        persist=True,
        conf=CONF["human"],   # ← ใช้ค่าจาก CONF
        classes=[0],
        verbose=False,
        tracker="bytetrack.yaml"
    )
    boxes = []
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF["human"]:   # filter ซ้ำ (กันกรณี YOLO ยังส่งมา)
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id[0]) if box.id is not None else None
            boxes.append({
                "x": float(x1), "y": float(y1),
                "w": float(x2 - x1), "h": float(y2 - y1),
                "conf": round(conf, 3),
                "id": track_id,
                "label": "human",
                "type": "human"
            })
    return boxes


def infer_face(frame: np.ndarray) -> list:
    results = model_face.predict(
        frame,
        conf=CONF["face"],    # ← ใช้ค่าจาก CONF
        verbose=False
    )
    boxes = []
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF["face"]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "x": float(x1), "y": float(y1),
                "w": float(x2 - x1), "h": float(y2 - y1),
                "conf": round(conf, 3),
                "id": None,
                "label": "face",
                "type": "face"
            })
    return boxes


def infer_pose(frame: np.ndarray) -> list:
    results = model_pose.predict(
        frame,
        conf=CONF["pose"],    # ← ใช้ค่าจาก CONF
        verbose=False
    )
    boxes = []
    result = results[0]
    if result.boxes is not None:
        for i, box in enumerate(result.boxes):
            conf = float(box.conf[0])
            if conf < CONF["pose"]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            keypoints = []
            if result.keypoints is not None and i < len(result.keypoints):
                kp = result.keypoints[i]
                if kp.data is not None:
                    for k in kp.data[0].tolist():   # (x, y, kp_conf)
                        keypoints.append({
                            "x":    float(k[0]),
                            "y":    float(k[1]),
                            "conf": round(float(k[2]), 3),
                        })

            boxes.append({
                "x": float(x1), "y": float(y1),
                "w": float(x2 - x1), "h": float(y2 - y1),
                "conf": round(conf, 3),
                "id": None,
                "label": "pose",
                "type": "pose",
                "keypoints": keypoints
            })
    return boxes


# ======================================================
# Serve Static Files
# ======================================================
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return FileResponse("index.html")


# ======================================================
# WebSocket Endpoint
# ======================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client = websocket.client
    client_info = f"{client.host}:{client.port}" if client else "unknown"

    await websocket.accept()
    print(f"\n🟢 [CONNECTED]  Client: {client_info}")

    frame_count  = 0
    start_time   = time.time()
    last_log_time = start_time
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()
            frame_count += 1

            np_arr = np.frombuffer(data, np.uint8)
            frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"  ⚠️  Frame {frame_count}: decode failed")
                continue

            h, w = frame.shape[:2]
            t0 = time.perf_counter()

            boxes_human, boxes_face, boxes_pose = await asyncio.gather(
                loop.run_in_executor(executor, infer_human, frame),
                loop.run_in_executor(executor, infer_face,  frame),
                loop.run_in_executor(executor, infer_pose,  frame),
            )

            inference_ms = (time.perf_counter() - t0) * 1000

            now = time.time()
            if now - last_log_time >= 1.0:
                elapsed = now - start_time
                fps = frame_count / elapsed
                print(
                    f"  📦 Frame:{frame_count:>5} | FPS:{fps:>5.1f} | "
                    f"Res:{w}x{h} | Infer:{inference_ms:>5.1f}ms | "
                    f"👤{len(boxes_human)} 😊{len(boxes_face)} 🏃{len(boxes_pose)}"
                )
                last_log_time = now

            response = {
                "w": w, "h": h,
                "inference_ms": round(inference_ms, 1),
                # ส่ง threshold ที่ใช้ไปด้วย ให้ frontend แสดงได้
                "conf_used": CONF,
                "humans":     boxes_human,
                "faces":      boxes_face,
                "poses":      boxes_pose,
                "detections": boxes_human + boxes_face + boxes_pose,
            }

            await websocket.send_text(json.dumps(response))

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n🔴 [DISCONNECTED]  Client: {client_info}")
        print(f"   Reason  : {e}")
        print(f"   Duration: {elapsed:.1f}s | Frames: {frame_count}")
        print("-" * 50)