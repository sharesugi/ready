from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model

# ✅ 파일 경로
MODEL_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/best_dnn_model.h5"
XGB_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/best_xgb_model.pkl"
SCALER_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/scaler.pkl"
POLY_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/poly_transformer.pkl"

# ✅ 모델 및 전처리기 불러오기
model = load_model(MODEL_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
poly = joblib.load(POLY_PATH)

app = Flask(__name__)
model_yolo = YOLO('/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/best_8s.pt')

is_detected = False

@app.route('/detect', methods=['POST'])
def detect():
    global is_detected

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model_yolo(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    print(f'########## detections : {detections} #############')
    if detections.size == 0:
        is_detected = False
    else:
        is_detected = True

    target_classes = {1: "car1", 2: "car2", 3: "tank", 4: "human"}
    filtered_results = []
    for box in detections:
        if box[4] >= 0.85:
            print(f'✨클래스 ID : {int(box[5])} Confidence : {box[4]}')
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

def find_angle_for_distance_dy_dnn(distance, dy):
    # ✅ 예측용 입력 설정 (Distance + dy)
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)

    # ✅ 예측
    y_pred = model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

    return float(y_pred_angle)
    
def find_angle_for_distance_dy_xgb(distance, dy):
    # ✅ 예측용 입력 설정 (Distance + dy)
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)

    # ✅ 예측
    y_pred = xgb_model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

    return float(y_pred_angle)

angle_hist = []
save_time = 0
len_angle_hist = -1

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, is_detected, not_hit_body_data, angle_hist, save_time, len_angle_hist

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

    save_time += 1
    if save_time > 10:
        save_time = 0
        angle_hist.append([round(turret_x, 2), round(turret_y, 2)])
        len_angle_hist += 1

    print(angle_hist)

    patience = 1 # 3 x n초
    if len_angle_hist > 3:
        print(angle_hist[len_angle_hist][:], angle_hist[len_angle_hist - patience][:])
        if angle_hist[len_angle_hist][:] == angle_hist[len_angle_hist - patience][:]:
            angle_hist = []
            len_angle_hist = -1
            last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}
    
    # 적 위치
    enemy_x = enemy_pos.get("x", 0)
    enemy_y = enemy_pos.get("y", 0)
    enemy_z = enemy_pos.get("z", 0)

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

    dy = pos_y - enemy_y

    if distance >= 130:
        last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

    if pos_y < 5 or enemy_y < 5:
        last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

    # y축 (pitch) 각도 보간
    target_pitch_dnn = find_angle_for_distance_dy_dnn(distance, dy)
    target_pitch_xgb = find_angle_for_distance_dy_xgb(distance, dy)
    target_pitch = (target_pitch_dnn + target_pitch_xgb) / 2

    # 현재 터렛 각도와 목표 각도 차이 계산
    yaw_diff = target_yaw - turret_x
    pitch_diff = target_pitch - turret_y

    # 각도 차이 보정 (-180 ~ 180)
    if yaw_diff > 180:
        yaw_diff -= 360
    elif yaw_diff < -180:
        yaw_diff += 360

    # 최소 가중치 0.01 설정, 최대 1.0 제한
    def calc_yaw_weight(diff):
        w = min(max(abs(diff) / 30, 0.01), 1.0)  # 30도 내외로 가중치 조절 예시
        return w
    
    # 최소 가중치 0.1 설정, 최대 1.0 제한
    def calc_pitch_weight(diff):
        w = min(max(abs(diff) / 30, 0.1), 1.0)  # 30도 내외로 가중치 조절 예시
        return w

    yaw_weight = calc_yaw_weight(yaw_diff)
    pitch_weight = calc_pitch_weight(pitch_diff)

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

    print(f'is_detected : {is_detected}')

    # 조준 완료 판단 (yaw, pitch 오차가 1도 이내일 때)
    aim_ready = bool(abs(yaw_diff) <= 0.1 and abs(pitch_diff) <= 0.1 and is_detected)
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
not_hit_body_data = []
time = 0

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, true_hit_ratio, not_hit_body_data, is_detected, time

    data = request.get_json()
    enemy_position = data.get('enemyPos', {})
    enemy_pos['x'] = enemy_position.get('x', 0)
    enemy_pos['y'] = enemy_position.get('y', 0)
    enemy_pos['z'] = enemy_position.get('z', 0)
    time = data.get("time", 0)
    body_x = data.get('playerBodyX', 0)
    body_y = data.get('playerBodyY', 0)
    body_z = data.get('playerBodyZ', 0)
    control = ""

    # not_hit_body_data = [body_x, body_y, body_z]
    print(f'Body : {body_x, body_y, body_z}')

    if time > 30:
        control = 'reset'
        last_bullet_info = {}

    if last_bullet_info:
        if last_bullet_info.get("hit") == "terrain":
            print("🌀 탄이 지형에 명중! 전차를 초기화합니다.")
            is_detected = False
            control = "reset"
            # true_hit_ratio.append(0)
            # df = pd.DataFrame(true_hit_ratio, columns=["is_hit"])
            # df.to_csv("true_hit_ratio_map5_DNN_4500.csv", index=False)

            # not_hit_body_data.append([body_x, body_y, body_z])
            # df = pd.DataFrame(not_hit_body_data, columns=["body_x", "body_y", "body_z"])
            # df.to_csv("not_hit_body_data.csv", index=False)
            last_bullet_info = {}

        if last_bullet_info.get("hit") == "enemy":
            print("🌀 탄이 적 전차에 명중! 전차를 초기화합니다.")
            is_detected = False
            control = "reset"
            # true_hit_ratio.append(1)
            # df = pd.DataFrame(true_hit_ratio, columns=["is_hit"])
            # df.to_csv("true_hit_ratio_map5_DNN_4500.csv", index=False)
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
        "detactMode": True,
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
    app.run(host='0.0.0.0', port=5011, debug=False, use_reloader=False)