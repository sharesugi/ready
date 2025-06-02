from flask import Flask, request, jsonify
from ultralytics import YOLO
import random
import pandas as pd
import requests
import time
import math

app = Flask(__name__)
model = YOLO('/root/jupyter_home/YOLOv8_Object_Detection/datasets/runs/detect/yolv8m_tankv2_augmentated2/weights/best.pt')

# === 카메라 및 화면 정보 (예시값, 환경에 맞게 수정 가능) ===
CAMERA_H_FOV = 90  # 수평 시야각
CAMERA_V_FOV = 60  # 수직 시야각
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 450

# 위치 추적 변수
current_position = None
last_position = None
position_history = []

# 자율 주행 상태
current_action = "W"
action_ticks_remaining = 0
wall_avoid_mode = False
fire = False

# Turret 관련 전역 상태
turret_qe_action = random.choice(["Q", "E"])
turret_qe_ticks = random.randint(10, 30)

turret_rf_action = random.choice(["R", "F"])
turret_rf_ticks = random.randint(10, 30)

# 전역 장애물 리스트
obstacles = []

# 최근 LiDAR 데이터를 저장할 변수
latest_lidar_data = []

# 시뮬레이터 장애물 업데이트 URL
SIMULATOR_URL = "http://127.0.0.1:5000/update_obstacle"  # 실제 시뮬레이터 주소로 교체

def calculate_move():
    global current_position, last_position, current_action, action_ticks_remaining, wall_avoid_mode

    if not current_position:
        return "W", 1.0

    x, z = current_position
    near_wall = x <= 10 or x >= 290 or z <= 10 or z >= 290

    if near_wall and not wall_avoid_mode:
        wall_avoid_mode = True
        current_action = random.choice(["A", "D"])
        action_ticks_remaining = random.randint(5, 10)
        print(f"🧱 벽 감지: 회전 → {current_action} for {action_ticks_remaining} ticks")

    if wall_avoid_mode:
        action_ticks_remaining -= 1
        if action_ticks_remaining <= 0:
            wall_avoid_mode = False
            current_action = "W"
            action_ticks_remaining = random.randint(20, 40)
        return current_action, 0.8

    if action_ticks_remaining <= 0:
        if current_action == "W":
            current_action = "D"
            action_ticks_remaining = random.randint(5, 15)
        else:
            current_action = "W"
            action_ticks_remaining = random.randint(20, 50)
        print(f"🔄 방향 전환: {current_action} for {action_ticks_remaining} ticks")

    action_ticks_remaining -= 1
    return current_action, 1.0

def calculate_turret_action():
    global turret_qe_action, turret_qe_ticks
    global turret_rf_action, turret_rf_ticks

    if turret_qe_ticks <= 0:
        turret_qe_action = random.choice(["Q", "E"])
        turret_qe_ticks = random.randint(10, 30)
        print(f"🎯 터렛 Q/E 방향 변경: {turret_qe_action} for {turret_qe_ticks} ticks")

    if turret_rf_ticks <= 0:
        turret_rf_action = random.choice(["R", "F"])
        turret_rf_ticks = random.randint(10, 30)
        print(f"🎯 터렛 R/F 방향 변경: {turret_rf_action} for {turret_rf_ticks} ticks")

    turret_qe_ticks -= 1
    turret_rf_ticks -= 1

    return turret_qe_action, turret_rf_action

# 내 전차, 쐈을 때 포 각도, 포탄 떨어진 위치
turret_info = []

@app.route('/get_action', methods=['POST'])
def get_action():
    global current_position, last_position, turret_info, fire
    # {'position': {'x': 60.0, 'y': 8.0, 'z': 40.0}, 'turret': {'x': 84.62, 'y': -0.37}}
    data = request.get_json(force=True)
    position = data.get("position", {})
    turret = data.get("turret", {})
    time = 0

    x = position.get("x", 0)
    y = position.get("y", 0)
    z = position.get("z", 0)

    if current_position:
        last_position = current_position
    current_position = (x, z)
    position_history.append(current_position)

    move, weight = calculate_move()
    turret_qe, turret_rf = calculate_turret_action()

    moveWS = {"command": move if move in ["W", "S", "STOP"] else "W", "weight": weight if move in ["W", "S", "STOP"] else 0.0}
    moveAD = {"command": move if move in ["A", "D"] else "", "weight": weight if move in ["A", "D"] else 0.0}

    if time % 10 == 0:
        if not fire:
            fire = True
            my_tank_info = [position.get("x", 0), position.get("y", 0), position.get("z", 0), turret.get('x', 0), turret.get('y', 0)]
            print(position.get("x", 0))
            print(position.get("y", 0))
            print(position.get("z", 0))
            print(turret.get('x', 0))
            print(turret.get('y', 0))
            turret_info = my_tank_info[:]
            print(turret_info)

    command = {
        "moveWS": moveWS,
        "moveAD": moveAD,
        "turretQE": {"command": turret_qe, "weight": 0.5},
        "turretRF": {"command": turret_rf, "weight": 0.1},
        "fire": fire
    }
    if fire:
        fire = False

    print(f"🚗 Position: {current_position}, Sent Command: {command}")
    return jsonify(command)

# === 픽셀 → 각도 변환 함수 ===
def pixel_to_angle(pixel_x, pixel_y):
    h_angle = ((pixel_x / IMAGE_WIDTH) - 0.5) * CAMERA_H_FOV
    v_angle = ((0.5 - (pixel_y / IMAGE_HEIGHT))) * CAMERA_V_FOV
    return h_angle, v_angle

# === LiDAR 포인트 중 가장 각도 유사한 포인트 찾기 ===
def find_matched_lidar_point(h_angle, v_angle, lidar_data):
    min_error = float('inf')
    best_point = None
    for pt in lidar_data:
        if not pt.get('isDetected', False):
            continue
        error = abs(pt['angle'] - h_angle) + abs(pt['verticalAngle'] - v_angle)
        if error < min_error:
            min_error = error
            best_point = pt
    return best_point

# # === /detect 엔드포인트 ===
# @app.route("/detect", methods=["POST"])
# def detect():
#     image = request.files.get('image')
#     if not image:
#         return jsonify({"error": "No image received"}), 400

#     image_path = 'temp.jpg'
#     image.save(image_path)

#     results = model(image_path)
#     detections = results[0].boxes.data.cpu().numpy()

#     # print(f"results : {results}")

#     target_classes = {0: "car1", 1: "car2", 2: "human", 3: "tank"}
#     filtered_results = []
#     for box in detections:
#         class_id = int(box[5])
#         if class_id in target_classes:
#             filtered_results.append({
#                 'className': target_classes[class_id],
#                 'bbox': [float(coord) for coord in box[:4]],
#                 'confidence': float(box[4]),
#                 'color': '#00FF00',
#                 'filled': True,
#                 'updateBoxWhileMoving': True
#             })

#     lidar_data = latest_lidar_data

#     # print(lidar_data)
#     detections = []
#     for obj in filtered_results:
#         if obj["className"] != "tank":
#             continue

#         # 바운딩 박스 → 중심 좌표
#         bbox = obj["bbox"]
#         cx = (bbox[0] + bbox[2]) / 2
#         cy = (bbox[1] + bbox[3]) / 2

#         # 중심 좌표 → 시야각
#         h_angle, v_angle = pixel_to_angle(cx, cy)

#         # LiDAR 중 가장 가까운 각도의 포인트 찾기
#         matched = find_matched_lidar_point(h_angle, v_angle, lidar_data)

#         if matched:
#             tank_info = {
#                 "class": obj["className"],
#                 "confidence": obj["confidence"],
#                 "bbox": obj["bbox"],
#                 "world_position": matched["position"],
#                 "distance": matched["distance"]
#             }
#             detections.append(tank_info)
#     print(detections)

#     return jsonify(filtered_results)

# @app.route('/info', methods=['POST'])
# def info():
#     global latest_lidar_data
#     data = request.get_json()

#     # print(data["lidarPoints"])

#     if data and "lidarPoints" in data:
#         latest_lidar_data = data["lidarPoints"]  # LiDAR 데이터 갱신
#         print(f"[INFO] LiDAR data updated: {len(latest_lidar_data)} points")
#     return jsonify({"status": "ok"})

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    # print(f"results : {results}")

    target_classes = {0: "car1", 1: "car2", 2: "human", 3: "tank"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': True,
                'updateBoxWhileMoving': True
            })
            print(filtered_results[-1])

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    # 여기서 lidar 실시간으로 받을 수 있음 (log mode)
    data = request.get_json()
    # print(data)
    return jsonify({"status": "success", "control": ""})

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    print("update_obstacle 호줄됨 !!!!!!!!!!!!!!!!!!!!!")
    # {'obstacles': [{'x_min': 102.19522857666016, 'x_max': 105.19522857666016, 'z_min': 114.58087158203125, 'z_max': 120.58087158203125}]}
    data = request.get_json()
    print(data)
    obstacles = data.get("obstacles")
    print(obstacles)
    print(f"🪨 장애물 업데이트: {len(obstacles)}개")
    return jsonify({'status': 'success'})

bullet_data = []

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global obstacles, turret_info
    # {'x': 83.64369, 'y': 7.860109, 'z': 76.69936, 'hit': 'terrain'}
    data = request.get_json()
    print(data)
    hit_type = data.get("hit")
    bullet_x = data.get("x", 0)
    bullet_y = data.get("y", 0)
    bullet_z = data.get("z", 0)

    turret_info.extend([data.get("x", 0), data.get("y", 0), data.get("z", 0)])
    print(turret_info)
    

    print(f"💥 bullet update: hit={hit_type}, x={bullet_x}, y={bullet_y}, z={bullet_z}")

    if hit_type == "enemy":
        removed = False
        new_obstacles = []
        for obs in obstacles:
            obx_min = obs.get("x_min")
            obx_max = obs.get("x_max")
            obz_min = obs.get("z_min")
            obz_max = obs.get("z_max")
            if obx_min <= bullet_x <= obx_max and obz_min <= bullet_z <= obz_max:
                print(f"[🎯] 적 제거: ({obx_min}, {obx_max}, {obz_min}, {obz_max})")
                removed = True
                continue
            new_obstacles.append(obs)

        if removed:
            obstacles = new_obstacles
            print(f"새로운 장애물 개수 : {obstacles}")
            try:
                res = requests.post(SIMULATOR_URL, json={"obstacles": obstacles})
                print(f"[📡] 장애물 재전송 완료 (status={res.status_code})")
            except Exception as e:
                print(f"[❌] 시뮬레이터 전송 실패: {e}")
        else:
            print("[⚠️] 제거할 적을 찾지 못함")

    return jsonify({'status': 'bullet processed'})

@app.route('/collision', methods=['POST'])
def collision():

    data = request.get_json()
    print(data)
    obj = data.get('objectName')
    pos = data.get('position', {})
    print(f"💥 Collision: {obj} at {pos}")
    return jsonify({'status': 'success'})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",
        "blStartX": 60,
        "blStartY": 10,
        "blStartZ": 40,
        "rdStartX": 59,
        "rdStartY": 0,
        "rdStartZ": 280,
        "trackingMode": False,
        "detactMode": False,
        "logMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Init config:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 Simulation Started")
    return jsonify({"control": ""})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'obstacle_count': len(obstacles),
        'obstacles': obstacles
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
