from flask import Flask, request, jsonify
from queue import PriorityQueue
import os
import torch
from ultralytics import YOLO
import math
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import time
import json

# ==== 경로탐색용 변수 ====
GRID_SIZE = 300
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

start_x, start_z = 20, 50
destination_x, destination_z = 250, 280
start = (start_x, start_z)
destination = (destination_x, destination_z)

INITIAL_YAW = 90.0

# ==== 조준/탐지용 모델 및 변수 ====
IMAGE_WIDTH = 1921
IMAGE_HEIGHT = 1080
FOV_HORIZONTAL = 47.81061
FOV_VERTICAL = 28

MODEL_PATH = "turret_final/best_dnn_model.h5"
XGB_PATH = "best_xgb_model.pkl"
SCALER_PATH = "scaler.pkl"
POLY_PATH = "poly_transformer.pkl"

# 모델 로딩
model_dnn = load_model(MODEL_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
poly = joblib.load(POLY_PATH)

model_yolo = YOLO('best_8s.pt')

# ==== Flask ====
app = Flask(__name__)

# ==== 통합 전역 변수 ====
mode = "move"     # "move", "aim"
tank_detected = False
enemy_pos = {}
yolo_results = []
last_bullet_info = {}
FIND_MODE = True

# ===== A* 알고리즘 및 경로 함수 =====
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1
    D2 = math.sqrt(2)
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def get_neighbors(pos):
    neighbors = []
    for dx, dz in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        x, z = pos[0] + dx, pos[1] + dz
        if 0 <= x < GRID_SIZE and 0 <= z < GRID_SIZE:
            # 대각선 이동시 모서리 못뚫게
            if dx != 0 and dz != 0:
                if maze[pos[1]][x] == 1 or maze[z][pos[0]] == 1:
                    continue
            if maze[z][x] == 0:
                neighbors.append((x, z))
    return neighbors

def a_star(start, goal):
    open_set = PriorityQueue()
    open_set.put((0, Node(start)))
    closed = set()
    while not open_set.empty():
        _, current = open_set.get()
        if current.position == goal:
            path = []
            while current:
                path.append(current.position)
                current = current.parent
            return path[::-1]
        closed.add(current.position)
        for nbr in get_neighbors(current.position):
            if nbr in closed: continue
            node = Node(nbr, current)
            dx = abs(nbr[0] - current.position[0])
            dz = abs(nbr[1] - current.position[1])
            step_cost = math.sqrt(2) if dx != 0 and dz != 0 else 1
            node.g = current.g + step_cost
            node.h = heuristic(nbr, goal)
            node.f = node.g + node.h
            open_set.put((node.f, node))
    return [start]

def calculate_angle(current, next_pos):
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]
    return (math.degrees(math.atan2(dz, dx)) + 360) % 360

# ==== Turret(조준/발사)용 함수 ====
def get_angles_from_yolo_bbox(bbox):
    x_center = (bbox["x1"] + bbox["x2"]) / 2
    y_center = (bbox["y1"] + bbox["y2"]) / 2
    x_norm = x_center / IMAGE_WIDTH
    y_norm = y_center / IMAGE_HEIGHT
    h_angle = (x_norm - 0.5) * FOV_HORIZONTAL
    v_angle = (0.5 - y_norm) * FOV_VERTICAL
    return h_angle, v_angle

def find_lidar_cluster_center_adaptive(lidar_points, h_angle, v_angle, bbox_width_ratio, bbox_height_ratio):
    h_angle_tol = bbox_width_ratio * FOV_HORIZONTAL
    v_angle_tol = bbox_height_ratio * FOV_VERTICAL
    candidates = [
        p for p in lidar_points
        if p["isDetected"]
        and abs((p["angle"] - h_angle + 180) % 360 - 180) < h_angle_tol
        and abs(p.get("verticalAngle", 0) - v_angle) < v_angle_tol
    ]
    if not candidates:
        return None
    avg_x = sum(p["position"]["x"] for p in candidates) / len(candidates)
    avg_y = sum(p["position"]["y"] for p in candidates) / len(candidates)
    avg_z = sum(p["position"]["z"] for p in candidates) / len(candidates)
    avg_dist = sum(p["distance"] for p in candidates) / len(candidates)
    return {
        "position": {"x": avg_x, "y": avg_y, "z": avg_z},
        "distance": avg_dist
    }

def match_yolo_to_lidar(bboxes, lidar_points):
    results = []
    for bbox in bboxes:
        h_angle, v_angle = get_angles_from_yolo_bbox(bbox)
        bbox_width_ratio = (bbox["x2"] - bbox["x1"]) / IMAGE_WIDTH
        bbox_height_ratio = (bbox["y2"] - bbox["y1"]) / IMAGE_HEIGHT
        cluster = find_lidar_cluster_center_adaptive(
            lidar_points, h_angle, v_angle, bbox_width_ratio, bbox_height_ratio
        )
        if cluster:
            results.append({
                "bbox": bbox,
                "matched_lidar_pos": cluster["position"],
                "distance": cluster["distance"]
            })
    return results

def get_yaw_angle(player_pos, enemy_pos):
    dx = enemy_pos['x'] - player_pos['x']
    dz = enemy_pos['z'] - player_pos['z']
    angle_rad = math.atan2(dx, dz)
    angle_deg = math.degrees(angle_rad)
    angle_deg = (angle_deg + 360) % 360
    return round(angle_deg, 2)

def find_angle_for_distance_dy_dnn(distance, dy):
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)
    y_pred = model_dnn.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))
    return float(y_pred_angle)

def find_angle_for_distance_dy_xgb(distance, dy):
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)
    y_pred = xgb_model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))
    return float(y_pred_angle)

# ==== Flask API ====
@app.route('/init', methods=['GET'])
def init():
    global mode, tank_detected, enemy_pos, last_bullet_info, FIND_MODE
    mode = "move"
    tank_detected = False
    enemy_pos = {}
    last_bullet_info = {}
    FIND_MODE = True
    return jsonify({
        "startMode": "start",
        "blStartX": start_x, "blStartY": 10, "blStartZ": start_z,
        "rdStartX": 160, "rdStartY": 10, "rdStartZ": 260,
        "trackingMode": True, "detectMode": True, "logMode": True,
        "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000
    })

@app.route('/info', methods=['POST'])
def info():
    # 장애물 맵핑, 라이다 처리 필요시 여기서!
    # (원래 0602.py에서 장애물 info 처리)
    return jsonify({"status": "success"})

@app.route('/detect', methods=['POST'])
def detect():
    global tank_detected, enemy_pos, mode, yolo_results
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400
    image_path = 'temp_image.jpg'
    image.save(image_path)

    # 라이다 데이터는 form-data로 전달 받는다고 가정
    lidar_points = request.form.get('lidar_points')
    if lidar_points:
        lidar_points = json.loads(lidar_points)
    else:
        lidar_points = []

    results = model_yolo(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    # YOLO 클래스 매핑
    target_classes = {1: "car1", 2: "car2", 3: "tank", 4: "human"}
    filtered_results = []
    current_bboxes = []
    for box in detections:
        if box[4] >= 0.85:
            class_id = int(box[5])
            if class_id == 3:
                current_bboxes.append({'x1': float(box[0]), 'y1': float(box[1]), 'x2': float(box[2]), 'y2': float(box[3])})
            if class_id in target_classes:
                filtered_results.append({
                    'className': target_classes[class_id],
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4]),
                    'color': '#00FF00',
                    'filled': False,
                    'updateBoxWhileMoving': True
                })

    yolo_results = match_yolo_to_lidar(current_bboxes, lidar_points)
    if yolo_results:
        tank_detected = True
        mode = "aim"
        enemy_pos.update(yolo_results[0]['matched_lidar_pos'])
    else:
        tank_detected = False
        mode = "move"
        enemy_pos.clear()
    return jsonify(filtered_results)

@app.route('/get_action', methods=['POST'])
def get_action():
    global mode, enemy_pos, tank_detected, last_bullet_info, FIND_MODE

    data = request.get_json(force=True)
    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    if mode == "move" or not tank_detected:
        # 경로 따라 자율주행 (0602.py 방식)
        current_grid = (int(pos_x), int(pos_z))
        path = a_star(current_grid, destination)
        if len(path) > 1:
            next_grid = path[1]
        else:
            next_grid = current_grid
        target_angle = calculate_angle(current_grid, next_grid)
        diff = (target_angle - turret_x + 360) % 360
        if diff > 180:
            diff -= 360
        forward = {'command': "W", 'weight': 0.45}
        turn = {'command': 'A' if diff > 0 else 'D', 'weight': min(abs(diff)/30, 1.0)}
        cmd = {
            'moveWS': forward,
            'moveAD': turn,
            'turretQE': {'command': '', 'weight': 0.0},
            'turretRF': {'command': '', 'weight': 0.0},
            'fire': False
        }
        # 적탱크 감지 시 정지
        if tank_detected:
            cmd['moveWS'] = {'command': 'STOP', 'weight': 1.0}
            cmd['moveAD'] = {'command': '', 'weight': 0.0}
            mode = "aim"
        return jsonify(cmd)

    elif mode == "aim" and tank_detected and enemy_pos:
        # Turret_Control.py의 조준/발사 방식
        player_pos = {"x": pos_x, "y": pos_y, "z": pos_z}
        # 적 좌표
        target_yaw = get_yaw_angle(player_pos, enemy_pos)
        distance = math.sqrt(
            (pos_x - enemy_pos['x'])**2 +
            (pos_y - enemy_pos['y'])**2 +
            (pos_z - enemy_pos['z'])**2
        )
        dy = pos_y - enemy_pos['y']
        target_pitch_dnn = find_angle_for_distance_dy_dnn(distance, dy)
        target_pitch_xgb = find_angle_for_distance_dy_xgb(distance, dy)
        target_pitch = (target_pitch_dnn + target_pitch_xgb) / 2

        yaw_diff = target_yaw - turret_x
        pitch_diff = target_pitch - turret_y
        # -180 ~ 180 범위로 조정
        if yaw_diff > 180:
            yaw_diff -= 360
        elif yaw_diff < -180:
            yaw_diff += 360

        def calc_weight(diff, min_w=0.01):
            return min(max(abs(diff) / 30, min_w), 1.0)

        if yaw_diff > 0.1:
            turretQE_cmd = "E"
        elif yaw_diff < -0.1:
            turretQE_cmd = "Q"
        else:
            turretQE_cmd = ""

        if pitch_diff > 0.1:
            turretRF_cmd = "R"
        elif pitch_diff < -0.1:
            turretRF_cmd = "F"
        else:
            turretRF_cmd = ""

        aim_ready = abs(yaw_diff) <= 0.1 and abs(pitch_diff) <= 0.1

        cmd = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": turretQE_cmd, "weight": calc_weight(yaw_diff)},
            "turretRF": {"command": turretRF_cmd, "weight": calc_weight(pitch_diff, 0.1)},
            "fire": aim_ready
        }

        # 발사 완료시 상태 복구
        if aim_ready:
            tank_detected = False
            mode = "move"
            enemy_pos.clear()

        return jsonify(cmd)

    # 예외상황 기본 멈춤
    return jsonify({
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False
    })

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global last_bullet_info
    last_bullet_info = request.get_json()
    print("💥 탄 정보 갱신:", last_bullet_info)
    return jsonify({"status": "ok"})

@app.route('/collision', methods=['POST'])
def collision():
    data = request.get_json()
    position = data.get('position', {})
    print(f"💥 Collision Detected - Position: ({position.get('x')}, {position.get('y')}, {position.get('z')})")
    return jsonify({'status': 'success'})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        destination = (int(x), int(z))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
