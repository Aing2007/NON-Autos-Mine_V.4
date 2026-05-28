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
# Keypoint Indices (COCO 17-point format)
# ======================================================
KEYPOINT_IDX = {
    "nose": 0,
    "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# Joints for angle calculation (parent-child-grandchild)
ANGLE_CHAINS = {
    "fine_motor": [
        (5, 7, 9),   # left shoulder-elbow-wrist
        (6, 8, 10),  # right shoulder-elbow-wrist
    ],
    "gross_motor": [
        (5, 11, 13),  # left shoulder-hip-knee
        (6, 12, 14),  # right shoulder-hip-knee
        (11, 13, 15), # left hip-knee-ankle
        (12, 14, 16), # right hip-knee-ankle
    ]
}

# ======================================================
# Scoring Config
# ======================================================
SCORE_CONFIG = {
    "attention": {
        "window_sec": 3,
        "speed_thresholds": [
            {"min": 0,   "max": 5,   "score": 100},
            {"min": 5,   "max": 20,  "score": 80},
            {"min": 20,  "max": 999, "score": 50},
        ]
    },
    "fine_motor": {
        "keypoints": [5, 6, 7, 8, 9, 10],
        "min_confidence": 0.4,
    },
    "gross_motor": {
        "keypoints": [11, 12, 13, 14, 15, 16],
        "min_confidence": 0.4,
    }
}

# ======================================================
# Helper Functions
# ======================================================
def is_center_inside(center: tuple, box: dict) -> bool:
    """ตรวจสอบว่าจุดศูนย์กลาง (ของใบหน้า) ตกอยู่ในกล่อง (ของร่างกาย) หรือไม่"""
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

def calculate_angle(kp_a, kp_b, kp_c):
    """
    คำนวณมุมระหว่าง 3 keypoints (A-B-C) ที่ B เป็นจุดยอด
    Returns: angle in degrees (0-180) or None if any point is invalid
    """
    if kp_a is None or kp_b is None or kp_c is None:
        return None
    if kp_a["conf"] < SCORE_CONFIG["fine_motor"]["min_confidence"] or \
       kp_b["conf"] < SCORE_CONFIG["fine_motor"]["min_confidence"] or \
       kp_c["conf"] < SCORE_CONFIG["fine_motor"]["min_confidence"]:
        return None
    
    # Vector BA and BC
    ba = np.array([kp_a["x"] - kp_b["x"], kp_a["y"] - kp_b["y"]])
    bc = np.array([kp_c["x"] - kp_b["x"], kp_c["y"] - kp_b["y"]])
    
    # Magnitude
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    
    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return None
    
    # Cosine similarity
    cos_angle = np.dot(ba, bc) / (mag_ba * mag_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

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
# Scoring Engine
# ======================================================
class ScoringEngine:
    def __init__(self):
        # Historical data per track_id: {track_id: [(timestamp, center_x, center_y, keypoints), ...]}
        self.speed_history = defaultdict(list)
        self.angle_history = defaultdict(list)
        self.last_score_time = defaultdict(float)
        self.current_scores = defaultdict(lambda: {"attention": 100, "fine_motor": 0, "gross_motor": 0})
    def compute_attention_score(self, center_x, center_y, track_id, current_time):
        """
        วัดอัตราเร็วของจุดศูนย์กลาง
        ถ้า 3 วิที่ผ่านมา มีความเร็วน้อย -> คะแนนสูง
        """
        self.speed_history[track_id].append({
            "time": current_time,
            "x": center_x,
            "y": center_y
        })
        
        # ลบข้อมูลเก่าเกิน 3 วิ
        cutoff_time = current_time - SCORE_CONFIG["attention"]["window_sec"]
        self.speed_history[track_id] = [
            h for h in self.speed_history[track_id] 
            if h["time"] >= cutoff_time
        ]
        
        # หากมีข้อมูลไม่ถึง 2 จุด ยังไม่สามารถคำนวณได้
        if len(self.speed_history[track_id]) < 2:
            return None
        
        # คำนวณระยะห่างทั้งหมด ใน 3 วิ
        total_distance = 0
        for i in range(1, len(self.speed_history[track_id])):
            prev = self.speed_history[track_id][i-1]
            curr = self.speed_history[track_id][i]
            dist = math.sqrt(
                (curr["x"] - prev["x"])**2 + 
                (curr["y"] - prev["y"])**2
            )
            total_distance += dist
        
        # Thresholding
        score = 50  # default lowest
        for threshold in SCORE_CONFIG["attention"]["speed_thresholds"]:
            if threshold["min"] <= total_distance < threshold["max"]:
                score = threshold["score"]
                break
        
        return score
    
    def compute_motor_skill_score(self, keypoints, track_id, current_time, motor_type):
        """
        วัดอัตราการขยับข้อต่อ โดยดูความเปลี่ยนแปลงของมุม
        motor_type: "fine_motor" หรือ "gross_motor"
        """
        if not keypoints or len(keypoints) < 17:
            return None
        
        # บันทึกข้อมูล
        self.angle_history[track_id].append({
            "time": current_time,
            "keypoints": keypoints,
            "motor_type": motor_type
        })
        
        # ลบข้อมูลเก่าเกิน 3 วิ
        cutoff_time = current_time - SCORE_CONFIG["attention"]["window_sec"]
        self.angle_history[track_id] = [
            h for h in self.angle_history[track_id] 
            if h["time"] >= cutoff_time
        ]
        
        # ต้องมีข้อมูลอย่างน้อย 2 frame
        history_for_type = [
            h for h in self.angle_history[track_id]
            if h["motor_type"] == motor_type
        ]
        if len(history_for_type) < 2:
            return None
        
        # คำนวณความเปลี่ยนแปลงของมุมในแต่ละ joint
        angle_changes = []
        
        for chain in ANGLE_CHAINS[motor_type]:
            idx_a, idx_b, idx_c = chain
            
            # หา angle ในแต่ละ frame
            angles = []
            for hist in history_for_type:
                kp = hist["keypoints"]
                angle = calculate_angle(kp[idx_a], kp[idx_b], kp[idx_c])
                if angle is not None:
                    angles.append(angle)
            
            # คำนวณเฉลี่ยของการเปลี่ยนแปลง
            if len(angles) > 1:
                for i in range(1, len(angles)):
                    change = abs(angles[i] - angles[i-1])
                    angle_changes.append(change)
        
        # ถ้าไม่มี angle change ให้คะแนน 0
        if not angle_changes:
            return 0
        
        # คำนวณเฉลี่ยการเปลี่ยนแปลง (degree per frame)
        avg_change = sum(angle_changes) / len(angle_changes)
        
        # Scaling: 0 degree → 0 point, 30+ degree → 100 point
        score = min(100, int((avg_change / 30.0) * 100))
        
        return score
    
    def should_update_score(self, track_id, current_time):
        """ตรวจสอบว่าควร update score ทุก 3 วิหรือไม่"""
        if track_id not in self.last_score_time:
            self.last_score_time[track_id] = current_time
            return True
        
        elapsed = current_time - self.last_score_time[track_id]
        if elapsed >= SCORE_CONFIG["attention"]["window_sec"]:
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
    play_method = params.get("playMethod", "standing")  # sitting or standing
    objective = params.get("objective", "attention")      # attention, fine_motor, gross_motor
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

            # ขั้นตอนที่ 3: สร้างข้อมูลรายบุคคลแบบเต็ม + คำนวณคะแนน
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
                # 🎯 คำนวณคะแนนตามแต่ละประเภท
                # ==================================================
                scores = {}
                
                # ==================================================
                # 🎯 [แก้ไขบั๊ก]: คำนวณคะแนนตามแต่ละประเภท (ประคองค่าคะแนนล่าสุด)
                # ==================================================
                
                # ตรวจสอบว่าครบรอบ 3 วินาทีที่ต้องประมวลผลหรือไม่
                if scoring_engine.should_update_score(tid, current_time):
                    
                    # 1. คำนวณ Attention Score
                    att_score = scoring_engine.compute_attention_score(center_x, center_y, tid, current_time)
                    if att_score is not None:
                        scoring_engine.current_scores[tid]["attention"] = att_score
                    
                    # 2. คำนวณ Fine Motor Score
                    fm_score = scoring_engine.compute_motor_skill_score(associated_kps, tid, current_time, "fine_motor")
                    if fm_score is not None:
                        scoring_engine.current_scores[tid]["fine_motor"] = fm_score
                    
                    # 3. คำนวณ Gross Motor Score
                    gm_score = scoring_engine.compute_motor_skill_score(associated_kps, tid, current_time, "gross_motor")
                    if gm_score is not None:
                        scoring_engine.current_scores[tid]["gross_motor"] = gm_score

                # ดึงค่าคะแนนล่าสุดที่มีการบันทึกไว้ส่งออกไป (ไม่เป็น 0 ในเฟรมย่อยอีกต่อไป)
                scores = scoring_engine.current_scores[tid]

                # Filter display based on play_method
                display_human = True
                display_pose_lower = True
                
                if play_method == "sitting":
                    # สำหรับ sitting ไม่ต้องแสดง lower body keypoints
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