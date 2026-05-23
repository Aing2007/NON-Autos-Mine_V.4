import json
import time
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

# ======================================================
# CORS
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# Load YOLO Model
# ======================================================
print("=" * 50)
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")
print("✅ YOLO loaded successfully")
print("=" * 50)

# ======================================================
# Serve Static Files
# ======================================================
app.mount("/static", StaticFiles(directory="."), name="static")

# ======================================================
# Home Route
# ======================================================
@app.get("/")
async def root():
    return FileResponse("game1.html")

# ======================================================
# WebSocket Endpoint
# ======================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client = websocket.client
    client_info = f"{client.host}:{client.port}" if client else "unknown"

    await websocket.accept()
    print(f"\n🟢 [CONNECTED]  Client: {client_info}")

    frame_count = 0
    start_time = time.time()
    last_log_time = start_time

    try:
        while True:

            # ==========================================
            # Receive image bytes from browser
            # ==========================================
            data = await websocket.receive_bytes()
            frame_count += 1

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"  ⚠️  Frame {frame_count}: decode failed (empty frame)")
                continue

            h, w = frame.shape[:2]

            # ==========================================
            # YOLO Human Detection
            # ==========================================
            t0 = time.perf_counter()

            results = model.track(
                frame,
                persist=True,
                conf=0.3,
                classes=[0],
                verbose=False,
                tracker="bytetrack.yaml"
            )

            inference_ms = (time.perf_counter() - t0) * 1000

            boxes = []
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else None

                    boxes.append({
                        "x": float(x1),
                        "y": float(y1),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                        "conf": conf,
                        "id": track_id,
                        "label": "human"
                    })

            # ==========================================
            # Log ทุก 1 วินาที
            # ==========================================
            now = time.time()
            if now - last_log_time >= 1.0:
                elapsed = now - start_time
                fps = frame_count / elapsed
                person_ids = [b["id"] for b in boxes]
                print(
                    f"  📦 Frame: {frame_count:>5} | "
                    f"FPS: {fps:>5.1f} | "
                    f"Res: {w}x{h} | "
                    f"Inference: {inference_ms:>5.1f}ms | "
                    f"People: {len(boxes)} {person_ids}"
                )
                last_log_time = now

            # ==========================================
            # Send JSON back to browser
            # ==========================================
            response = {
                "w": w,
                "h": h,
                "boxes": boxes
            }

            await websocket.send_text(json.dumps(response))

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n🔴 [DISCONNECTED]  Client: {client_info}")
        print(f"   Reason  : {e}")
        print(f"   Duration: {elapsed:.1f}s | Frames received: {frame_count}")
        print("-" * 50)