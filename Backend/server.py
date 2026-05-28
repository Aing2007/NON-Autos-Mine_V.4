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
# Confidence Thresholds
# ======================================================
CONF = {
    "human":    0.80,
    "face":     0.75,
    "pose":     0.60,
    "keypoint": 0.30,
}

executor = ThreadPoolExecutor(max_workers=3)


# ======================================================
# Helper Functions สำหรับวิเคราะห์จับคู่พิกัด (Spatial Math)
# ======================================================
def is_center_inside(center: tuple, box: dict) -> bool:
    """ตรวจสอบว่าจุดศูนย์กลาง (ของใบหน้า) ตกอยู่ในกล่อง (ของร่างกาย) หรือไม่"""
    cx, cy = center
    bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
    return (bx <= cx <= bx + bw) and (by <= cy <= by + bh)

def compute_iou(box1: dict, box2: dict) -> float:
    """คำนวณการทับซ้อน (Intersection over Union) เพื่อจับคู่กล่องที่ใกล้เคียงกันที่สุด"""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


# ======================================================
# Model inference functions
# ======================================================
def infer_human(frame: np.ndarray) -> list:
    results = model_human.track(
        frame,
        persist=True,
        conf=CONF["human"],
        classes=[0],
        verbose=False,
        tracker="bytetrack.yaml"
    )
    boxes = []
    result = results[0]
    if result.boxes is not None:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < CONF["human"]:
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
        conf=CONF["face"],
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
            
            # ดึงชื่อเด็กจากคลาสที่โมเดลตรวจจับได้จริง (ดึงจากเดตาเซ็ตที่เทรนไว้)
            cls_id = int(box.cls[0])
            child_name = model_face.names.get(cls_id, f"Face_{cls_id}")
            
            boxes.append({
                "x": float(x1), "y": float(y1),
                "w": float(x2 - x1), "h": float(y2 - y1),
                "conf": round(conf, 3),
                "id": None,
                "label": child_name, # เปลี่ยนจาก "face" เป็นชื่อเด็กจริง
                "type": "face"
            })
    return boxes


def infer_pose(frame: np.ndarray) -> list:
    results = model_pose.predict(
        frame,
        conf=CONF["pose"],
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
                    for k in kp.data[0].tolist():
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

    # 💾 ระบบจดจำอัตลักษณ์จำลอง (Track Memory Session)
    # ช่วยให้จำได้ว่า Track ID นี้คือใคร แม้เด็กจะหันหลังชั่วขณะแล้วสแกนหน้าไม่เจอก็ตาม
    track_to_child_memory = {}

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

            # ดึงข้อมูลจาก 3 โมเดลแบบขนาน
            boxes_human, boxes_face, boxes_pose = await asyncio.gather(
                loop.run_in_executor(executor, infer_human, frame),
                loop.run_in_executor(executor, infer_face,  frame),
                loop.run_in_executor(executor, infer_pose,  frame),
            )

            # ==================================================
            # 🧠 ALGORITHM: ขบวนการผูกข้อมูลพิกัดซ้อนทับ (Data Binding)
            # ==================================================
            
            # ขั้นตอนที่ 1: ตรวจสอบพิกัดใบหน้าว่าอยู่ในการตีกรอบของโครงร่างมนุษย์คนไหน
            for face in boxes_face:
                face_center = (face["x"] + face["w"]/2, face["y"] + face["h"]/2)
                for human in boxes_human:
                    if human["id"] is not None and is_center_inside(face_center, human):
                        # ผูกชื่อเด็กตัวจริงเข้ากับระบบจดจำ Track ID ของกล้อง
                        track_to_child_memory[human["id"]] = face["label"]
                        break

            # ขั้นตอนที่ 2: จับคู่โครงกระดูก (Pose) เข้ากับโครงร่างมนุษย์โดยใช้ค่า IoU พื้นที่ทับซ้อน
            pose_to_track_map = {}
            for pose in boxes_pose:
                best_iou = 0.0
                matched_track_id = None
                for human in boxes_human:
                    iou = compute_iou(pose, human)
                    if iou > best_iou and iou > 0.4:  # ต้องมีพื้นที่ซ้อนทับกันเกิน 40%
                        best_iou = iou
                        matched_track_id = human["id"]
                if matched_track_id is not None:
                    pose_to_track_map[id(pose)] = matched_track_id

            # ขั้นตอนที่ 3: จัดกลุ่มข้อมูลแบบเบ็ดเสร็จรายบุคคล (Structured Output)
            tracked_children = []
            for human in boxes_human:
                tid = human["id"]
                if tid is None:
                    continue

                # ค้นหาชื่อเด็กจากความจำ หากเฟรมนี้มองไม่เห็นหน้า จะดึงค่าเก่าในอดีตมาใช้แทนทันที
                child_name = track_to_child_memory.get(tid, f"Unknown_ID_{tid}")

                # ค้นหาจุด Keypoints ข้อต่อที่ถูกผูกไว้กับมนุษย์คนนี้
                associated_kps = []
                for pose in boxes_pose:
                    if pose_to_track_map.get(id(pose)) == tid:
                        associated_kps = pose["keypoints"]
                        break

                # คำนวณจุดตัดเส้นทแยงมุม (จุดกึ่งกลางลำตัวจริง 100%)
                center_x = human["x"] + (human["w"] / 2)
                center_y = human["y"] + (human["h"] / 2)

                tracked_children.append({
                    "name": child_name,
                    "track_id": tid,
                    "bbox": {
                        "x": round(human["x"], 1),
                        "y": round(human["y"], 1),
                        "w": round(human["w"], 1),
                        "h": round(human["h"], 1)
                    },
                    "center": [round(center_x, 1), round(center_y, 1)],  # จุดตัดเส้นทแยงมุมตามโจทย์
                    "keypoints": associated_kps,
                    "conf": human["conf"]
                })

            # ==================================================

            inference_ms = (time.perf_counter() - t0) * 1000

            now = time.time()
            if now - last_log_time >= 1.0:
                elapsed = now - start_time
                fps = frame_count / elapsed
                print(
                    f"  📦 Frame:{frame_count:>5} | FPS:{fps:>5.1f} | "
                    f"Infer:{inference_ms:>5.1f}ms | "
                    f"🧒 จับคู่สำเร็จแล้ว: {len(tracked_children)} คน"
                )
                last_log_time = now

            response = {
                "w": w, "h": h,
                "inference_ms": round(inference_ms, 1),
                "conf_used": CONF,
                "humans":     boxes_human,
                "faces":      boxes_face,
                "poses":      boxes_pose,
                "detections": boxes_human + boxes_face + boxes_pose,
                
                # 🔥 ส่งชุดข้อมูลที่ผูกเสร็จแล้วไปให้หน้าเว็บประเมินต่อได้สบายๆ
                "tracked_children": tracked_children 
            }

            await websocket.send_text(json.dumps(response))

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n🔴 [DISCONNECTED]  Client: {client_info}")
        print(f"   Reason  : {e}")
        print(f"   Duration: {elapsed:.1f}s | Frames: {frame_count}")
        print("-" * 50)