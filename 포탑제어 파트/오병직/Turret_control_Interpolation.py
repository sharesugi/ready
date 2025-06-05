from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 로드
df = pd.read_csv('/root/jupyter_home/tank_project/ready/포탑제어 파트/Data/data_060211.csv')

# 거리 계산
df['dist'] = np.sqrt(
    (df['x_pos'] - df['x_target'])**2 +
    (df['y_pos'] - df['y_target'])**2 +
    (df['z_pos'] - df['z_target'])**2
)

# ✅ 130m 넘는 거리 제거 (이상치)
df = df[df['dist'] <= 130]

# ✅ 21m보다 가까운 거리 제거 (이상치)
df = df[df['dist'] > 21]

# ✅ 각 y_angle별 중앙값 거리 계산
grouped = df.groupby('y_angle')['dist'].median().reset_index()

# 보간용 배열 생성
angles = np.array(grouped['y_angle'])        # y축 각도
distances = np.array(grouped['dist'])        # 중앙값 거리

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: "person", 2: "car", 7: "truck", 15: "rock"}
    filtered_results = []
    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)

def get_yaw_angle(player_pos, enemy_pos):
    dx = enemy_pos['x'] - player_pos['x']
    dz = enemy_pos['z'] - player_pos['z']

    # atan2는 radian 기준, z를 먼저 넣는 이유는 좌표계 때문
    angle_rad = math.atan2(dx, dz)
    angle_deg = math.degrees(angle_rad)

    # 0~359로 변환
    angle_deg = (angle_deg + 360) % 360

    return round(angle_deg, 2)

def find_angle_for_distance(target_distance, angles, distances):
    # 거리 기준 정렬
    sort_idx = np.argsort(distances)
    sorted_distances = distances[sort_idx]
    sorted_angles = angles[sort_idx]
    
    # 범위 확인
    if not (target_distance < 130):
        return 10
    
    # 선형 보간
    return np.interp(target_distance, sorted_distances, sorted_angles)

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info

    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    # 현재 내 위치
    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    # 현재 터렛 각도 (x: yaw, y: pitch)
    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    # 적 위치
    enemy_x = enemy_pos.get("x", 0)
    enemy_y = enemy_pos.get("y", 0)
    enemy_z = enemy_pos.get("z", 0)

    print(enemy_x, enemy_y, enemy_z)

    player_pos = {"x": pos_x, "y": pos_y, "z": pos_z}
    enemy_pos = {"x": enemy_x, "y": enemy_y, "z": enemy_z}

    # 수평 각도 계산
    target_yaw = get_yaw_angle(player_pos, enemy_pos)

    # 거리 계산
    distance = math.sqrt(
        (pos_x - enemy_x)**2 +
        (pos_y - enemy_y)**2 +
        (pos_z - enemy_z)**2
    )

    if distance >= 130:
        last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

    # y축 (pitch) 각도 보간
    target_pitch = find_angle_for_distance(distance, angles, distances)

    # 현재 터렛 각도와 목표 각도 차이 계산
    yaw_diff = target_yaw - turret_x
    pitch_diff = target_pitch - turret_y

    # 각도 차이 보정 (-180 ~ 180)
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    # 최소 가중치 0.1 설정, 최대 1.0 제한
    def calc_weight(diff):
        w = min(max(abs(diff) / 30, 0.1), 1.0)  # 30도 내외로 가중치 조절 예시
        return w

    yaw_weight = calc_weight(yaw_diff)
    pitch_weight = calc_weight(pitch_diff)

    # 좌우 회전 명령 결정 (Q: CCW, E: CW)
    if yaw_diff > 0.1:  # 목표가 오른쪽
        turretQE_cmd = "E"
    elif yaw_diff < -0.1:  # 목표가 왼쪽
        turretQE_cmd = "Q"
    else:
        turretQE_cmd = ""

    # 상하 포탑 명령 (R: up, F: down)
    if pitch_diff > 0.1:  # 포탑을 위로 올림
        turretRF_cmd = "R"
    elif pitch_diff < -0.1:
        turretRF_cmd = "F"
    else:
        turretRF_cmd = ""

    # 조준 완료 판단 (yaw, pitch 오차가 1도 이내일 때)
    aim_ready = bool(abs(yaw_diff) <= 0.1 and abs(pitch_diff) <= 0.1)
    print(target_yaw, target_pitch)
    print(aim_ready)

    # 이동은 일단 멈춤
    command = {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0},
        "turretRF": {"command": turretRF_cmd, "weight": pitch_weight if turretRF_cmd else 0.0},
        "fire": aim_ready
    }

    print("🔁 Sent Combined Action:", command)
    return jsonify(command)

# 전역 상태 저장
last_bullet_info = {}

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global last_bullet_info
    last_bullet_info = request.get_json()
    print("💥 탄 정보 갱신됨:", last_bullet_info)
    return jsonify({"result": "ok"})

enemy_pos = {}
true_hit_ratio = []

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, true_hit_ratio

    data = request.get_json()
    enemy_position = data.get('enemyPos', {})
    enemy_pos['x'] = enemy_position.get('x', 0)
    enemy_pos['y'] = enemy_position.get('y', 0)
    enemy_pos['z'] = enemy_position.get('z', 0)
    time = data.get("time", 0)
    control = ""

    if time > 30:
        control = 'reset'
        last_bullet_info = {}

    if last_bullet_info:
        if last_bullet_info.get("hit") == "terrain":
            print("🌀 탄이 지형에 명중! 전차를 초기화합니다.")
            control = "reset"
            true_hit_ratio.append(0)
            df = pd.DataFrame(true_hit_ratio, columns=["is_hit"])
            df.to_csv("true_hit_ratio_map1.csv", index=False)
            last_bullet_info = {}

        if last_bullet_info.get("hit") == "enemy":
            print("🌀 탄이 적 전차에 명중! 전차를 초기화합니다.")
            control = "reset"
            true_hit_ratio.append(1)
            df = pd.DataFrame(true_hit_ratio, columns=["is_hit"])
            df.to_csv("true_hit_ratio_map1.csv", index=False)
            last_bullet_info = {}
        else:
            control = "reset"
            last_bullet_info = {}

    return jsonify({
        "status": "success",
        "message": "Data received",
        "control": control,
    })

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})

@app.route('/collision', methods=['POST']) 
def collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    print("🛠️ /init 라우트 진입 확인!")

    blStartX = random.uniform(10, 290)
    blStartY = 20
    blStartZ = random.uniform(10, 290)
    rlStartX = random.uniform(10, 290)
    rlStartY = 20
    rlStartZ = random.uniform(10, 290)

    config = {
        "startMode": "start",
        "blStartX": blStartX,
        "blStartY": blStartY,
        "blStartZ": blStartZ,
        "rdStartX": rlStartX,
        "rdStartY": rlStartY,
        "rdStartZ": rlStartZ,
        "trackingMode": True,
        "detactMode": False,
        "logMode": True,
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
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False, use_reloader=False)
