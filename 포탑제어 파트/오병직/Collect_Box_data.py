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

SAVE_MODE = False
FIND_MODE = False
is_detected = False

detected_list = []

@app.route('/detect', methods=['POST'])
def detect():
    global is_detected, detected_list, last_bullet_info, SAVE_MODE

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model_yolo(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    if detections.size == 0:
        is_detected = False
    else:
        is_detected = True

    target_classes = {1: "car1", 2: "car2", 3: "tank", 4: "human"}
    filtered_results = []
    for box in detections:
        if box[4] >= 0.90:
            print(f'✨클래스 ID : {int(box[5])} Confidence : {box[4]}')
            class_id = int(box[5])
            if class_id == 3 and SAVE_MODE: 
                detected_list = [box[0], box[1], box[2], box[3]] # x_min, y_min, x_max, y_max
                last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}
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

pos_hist = []
save_time = 0
len_pos_hist = -1
pos_list = []

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, is_detected, pos_hist, save_time, len_pos_hist, FIND_MODE, pos_list, SAVE_MODE, p_speed, e_speed, time

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

    if p_speed < 0.1 and e_speed < 0.1 and time > 5:
        pos_list = [pos_x, pos_y, pos_z, enemy_x, enemy_y, enemy_z, turret_x, turret_y]
        FIND_MODE = True
        SAVE_MODE = True

    print(f'🛹🛹🛹🛹{p_speed}, {e_speed}, {FIND_MODE}, {SAVE_MODE}, {time}')
    
    player_pos = {"x": pos_x, "y": pos_y, "z": pos_z}
    enemy_pos = {"x": enemy_x, "y": enemy_y, "z": enemy_z}

    if FIND_MODE:
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

        # 이동은 일단 멈춤
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0},
            "turretRF": {"command": turretRF_cmd, "weight": pitch_weight if turretRF_cmd else 0.0},
            "fire": False
        }
    else:
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False
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
time = 0
box_data_list = []
p_speed = 10
e_speed = 10

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, is_detected, time, FIND_MODE, detected_list, pos_list, box_data_list, SAVE_MODE, p_speed, e_speed

    data = request.get_json()
    enemy_position = data.get('enemyPos', {})
    enemy_pos['x'] = enemy_position.get('x', 0)
    enemy_pos['y'] = enemy_position.get('y', 0)
    enemy_pos['z'] = enemy_position.get('z', 0)
    time = data.get("time", 0)
    p_speed = data.get('playerSpeed', 0)
    e_speed = data.get('enemySpeed', 0)
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

        if is_detected and SAVE_MODE:
            print("🌀 전차 감지 ! 데이터 저장 ! 초기화 !")
            save = []
            save.extend(pos_list) # 각 전차 좌표, 터렛 각도
            save.extend(detected_list) # 탐지된 객체 박스 좌표 (x_min, y_min, x_max, y_max)
            save.extend([body_x, body_y, body_z])
            if len(save) == 15:
                box_data_list.append(save)
            df = pd.DataFrame(box_data_list, columns=['pos_x', 'pos_y', 'pos_z',
                                               'enemy_x', 'enemy_y', 'enemy_z', 
                                               'turret_x', 'turret_y',
                                               'x_min', 'y_min', 'x_max', 'y_max',
                                               'body_x', 'body_y', 'body_z'])
            df.to_csv('box_data.csv', index=False)
            last_bullet_info = {}
            is_detected = False
            FIND_MODE = False
            SAVE_MODE = False
            control = "reset"
        else:
            control = "reset"
            last_bullet_info = {}

    return jsonify({
        "status": "success",
        "message": "Data received",
        "control": control,
    })

@app.route('/init', methods=['GET'])
def init():
    print("🛠️ /init 라우트 진입 확인!")

    blStartX = random.uniform(10, 290)
    blStartY = 40
    blStartZ = random.uniform(10, 290)
    rlStartX = random.uniform(10, 290)
    rlStartY = 40
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
    app.run(host='0.0.0.0', port=5003, debug=False, use_reloader=False)