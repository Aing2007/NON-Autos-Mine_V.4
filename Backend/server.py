import json
import time
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import math

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
# Helper Functions
# ======================================================
def is_center_inside(center: tuple, box: dict) -> bool:
    """ตรวจสอบว่าจุดศูนย์กลาง (ของใบหน้า) ตกอยู่ในกล่อง (ของร่างกาย)หรือไม่"""
    cx, cy = center
    bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
    return (bx <= cx <= bx + bw) and (by <= cy <= by + bh)

def compute_iou(box1: dict, box2: dict) -> float:
    """คำนวณการทับซ้อน (Intersection over Union)"""
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
            cls_id = int(box.cls[0])
            child_name = model_face.names.get(cls_id, f"Face_{cls_id}")
            
            boxes.append({
                "x": float(x1), "y": float(y1),
                "w": float(x2 - x1), "h": float(y2 - y1),
                "conf": round(conf, 3),
                "id": None,
                "label": child_name,
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
# 🧠 NEW PURE BODY-CENTER SCORING ENGINE
# ======================================================
class ScoringEngine:
    def __init__(self):
        # เก็บพิกัดกึ่งกลางลำตัวย้อนหลัง 3 วินาทีของเด็กแต่ละคน
        self.center_history = defaultdict(list)
        # ตัวแปรประคองสถานะคะแนนล่าสุดของแต่ละ Track ID (เริ่มต้นที่คะแนนเต็ม 100)
        self.current_scores = defaultdict(lambda: {"attention": 100, "fine_motor": 100, "gross_motor": 100})
        # ตัวแปรบันทึกรอบเวลาอัปเดต (ทุก 1 วินาทีเพื่อให้กราฟตอบสนองไวขึ้น)
        self.last_score_time = defaultdict(float)
        
    def compute_pure_center_score(self, center_x, center_y, track_id, current_time):
        """
        [อัลกอริทึมใหม่เพียว 100%]: วัดเฉพาะการเคลื่อนไหวจากจุดกึ่งกลางลำตัวเท่านั้น
        ยิ่งขยับตัวน้อย (ระยะขยับสั้น) -> สมาธิดีเยี่ยม คะแนนยิ่งมากเข้าใกล้ 100
        """
        self.center_history[track_id].append({
            "time": current_time,
            "x": center_x,
            "y": center_y
        })
        
        # ถือถังข้อมูลย้อนหลังไว้ 3 วินาที
        cutoff_time = current_time - 3.0
        self.center_history[track_id] = [
            h for h in self.center_history[track_id] 
            if h["time"] >= cutoff_time
        ]
        
        if len(self.center_history[track_id]) < 2:
            return 100  # เฟรมแรกให้คะแนนเต็มไว้ก่อน
        
        # คำนวณระยะทางรวมที่จุดศูนย์กลางขยับย้ายที่ในรอบ 3 วินาที (Euclidean Distance)
        total_movement = 0.0
        for i in range(1, len(self.center_history[track_id])):
            prev = self.center_history[track_id][i-1]
            curr = self.center_history[track_id][i]
            dist = math.sqrt((curr["x"] - prev["x"])**2 + (curr["y"] - prev["y"])**2)
            total_movement += dist
            
        # 🎯 กฎเกณฑ์การแปลงพิกัดความนิ่งเป็นคะแนน (ยิ่งขยับน้อย คะแนนยิ่งมาก):
        # - ขยับรวม < 5 พิกเซล (อยู่นิ่งสนิท/ตัวสั่นเล็กน้อย)    -> ได้ 100 คะแนนเต็ม
        # - ขยับรวม 5 ถึง 25 พิกเซล (ขยับตัวเล็กน้อย/เอียงตัว)   -> ได้ 85 คะแนน
        # - ขยับรวม 25 ถึง 60 พิกเซล (เริ่มอยู่ไม่นิ่ง/ลุกขยับ)   -> ได้ 60 คะแนน
        # - ขยับรวมเกิน 60 พิกเซลขึ้นไป (วิ่งเล่น/ขยับรุนแรงมาก) -> ได้ 35 คะแนน
        if total_movement < 5.0:
            score = 100
        elif total_movement < 25.0:
            score = 85
        elif total_movement < 60.0:
            score = 60
        else:
            score = 35
            
        return score
    
    def should_update_score(self, track_id, current_time):
        """ตรวจสอบรอบการอัปเดตทุกๆ 1 วินาที เพื่อให้แสดงผลลัพธ์ลื่นไหลขึ้น"""
        if track_id not in self.last_score_time:
            self.last_score_time[track_id] = current_time
            return True
        
        if current_time - self.last_score_time[track_id] >= 1.0:
            self.last_score_time[track_id] = current_time
            return True
        return False


# ======================================================
# WebSocket Endpoint
# ======================================================
scoring_engine = ScoringEngine()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client = websocket.client
    client_info = f"{client.host}:{client.port}" if client else "unknown"

    await websocket.accept()
    print(f"\n🟢 [CONNECTED]  Client: {client_info}")

    # Receive parameters from client
    init_msg = await websocket.receive_text()
    params = json.loads(init_msg) if init_msg else {}
    play_method = params.get("playMethod", "standing")  
    objective = params.get("objective", "attention")      
    game_name = params.get("gameName", "Unknown")

    print(f"  📋 PlayMethod: {play_method} | Objective: {objective} | Game: {game_name}")

    track_to_child_memory = {}
    frame_count = 0
    start_time = time.time()
    last_log_time = start_time
    loop = asyncio.get_event_loop()

    try:
        while True:
            data = await websocket.receive_bytes()
            frame_count += 1

            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"  ⚠️  Frame {frame_count}: decode failed")
                continue

            h, w = frame.shape[:2]
            t0 = time.perf_counter()
            current_time = time.time()

            # ดึงข้อมูลจาก 3 โมเดลแบบขนาน
            boxes_human, boxes_face, boxes_pose = await asyncio.gather(
                loop.run_in_executor(executor, infer_human, frame),
                loop.run_in_executor(executor, infer_face, frame),
                loop.run_in_executor(executor, infer_pose, frame),
            )

            # ==================================================
            # Binding Phase
            # ==================================================
            
            # ขั้นตอนที่ 1: ผูกชื่อเด็กกับ Track ID
            for face in boxes_face:
                face_center = (face["x"] + face["w"]/2, face["y"] + face["h"]/2)
                for human in boxes_human:
                    if human["id"] is not None and is_center_inside(face_center, human):
                        track_to_child_memory[human["id"]] = face["label"]
                        break

            # ขั้นตอนที่ 2: จับคู่โครงกระดูกกับร่างกาย
            pose_to_track_map = {}
            for pose in boxes_pose:
                best_iou = 0.0
                matched_track_id = None
                for human in boxes_human:
                    iou = compute_iou(pose, human)
                    if iou > best_iou and iou > 0.4:
                        best_iou = iou
                        matched_track_id = human["id"]
                if matched_track_id is not None:
                    pose_to_track_map[id(pose)] = matched_track_id

            # ขั้นตอนที่ 3: สร้างข้อมูลรายบุคคลแบบเต็ม + คำนวณคะแนนตามจุดกึ่งกลางลำตัวเพียวๆ
            tracked_children = []
            
            for human in boxes_human:
                tid = human["id"]
                if tid is None:
                    continue

                child_name = track_to_child_memory.get(tid, f"Unknown_ID_{tid}")
                center_x = human["x"] + (human["w"] / 2)
                center_y = human["y"] + (human["h"] / 2)

                # หา keypoints ที่ผูกไว้
                associated_kps = []
                for pose in boxes_pose:
                    if pose_to_track_map.get(id(pose)) == tid:
                        associated_kps = pose["keypoints"]
                        break

                # ==================================================
                # 🎯 คำนวณคะแนนโดยอ้างอิงจุดกึ่งกลางลำตัวเพียวอย่างเดียว (แก้ไขใหม่)
                # ==================================================
                if scoring_engine.should_update_score(tid, current_time):
                    # เรียกใช้อัลกอริทึมใหม่เพียวๆ
                    calculated_score = scoring_engine.compute_pure_center_score(
                        center_x, center_y, tid, current_time
                    )
                    # จ่ายค่าคะแนนเดียวกันให้ทั้ง 3 ออบเจกต์เพื่อไม่ให้หน้าบ้านเกิด Error ไม่ว่าจะรันโหมดใดอยู่ก็ตาม
                    scoring_engine.current_scores[tid] = {
                        "attention": calculated_score,
                        "fine_motor": calculated_score,
                        "gross_motor": calculated_score
                    }

                # ดึงสถานะคะแนนล่าสุดส่งออกไปในทุกเฟรม (แก้บั๊กคะแนนเป็น 0)
                scores = scoring_engine.current_scores[tid]

                # Filter display based on play_method
                display_pose_lower = True
                if play_method == "sitting":
                    display_pose_lower = False

                tracked_children.append({
                    "name": child_name,
                    "track_id": tid,
                    "bbox": {
                        "x": round(human["x"], 1),
                        "y": round(human["y"], 1),
                        "w": round(human["w"], 1),
                        "h": round(human["h"], 1)
                    },
                    "center": [round(center_x, 1), round(center_y, 1)],
                    "keypoints": associated_kps,
                    "conf": human["conf"],
                    "scores": scores,
                    "display_lower_body": display_pose_lower
                })

            inference_ms = (time.perf_counter() - t0) * 1000

            now = time.time()
            if now - last_log_time >= 1.0:
                elapsed = now - start_time
                fps = frame_count / elapsed
                print(
                    f"  📦 Frame:{frame_count:>5} | FPS:{fps:>5.1f} | "
                    f"Infer:{inference_ms:>5.1f}ms | "
                    f"🧒 Tracked: {len(tracked_children)} children"
                )
                last_log_time = now

            # ==================================================
            # Response
            # ==================================================
            response = {
                "w": w, "h": h,
                "inference_ms": round(inference_ms, 1),
                "conf_used": CONF,
                "humans": boxes_human,
                "faces": boxes_face,
                "poses": boxes_pose,
                "detections": boxes_human + boxes_face + boxes_pose,
                "tracked_children": tracked_children,
                "params": {
                    "playMethod": play_method,
                    "objective": objective,
                    "gameName": game_name
                }
            }

            await websocket.send_text(json.dumps(response))

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n🔴 [DISCONNECTED]  Client: {client_info}")
        print(f"   Reason  : {e}")
        print(f"   Duration: {elapsed:.1f}s | Frames: {frame_count}")
        print("-" * 50)