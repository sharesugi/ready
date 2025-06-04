# 이거 맵마다 내 전차 y값은 /init에서 수정해주셔야 합니다.
# 포트번호 5004번 입니다. 수정해서 사용하세요!

from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO
import random, math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

time = 0

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, turret_info, time

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

    turretQE_cmd = random.choice(['Q', 'E'])
    turretRF_cmd = random.choice(['R', 'F'])

    yaw_weight = round(random.random(), 1)
    pitch_weight = round(random.random(), 1)
    
    fire = False

    if time > 5 and len(turret_info) == 0:
        fire = True
        print(f'pos_x : {pos_x}')
        print(f'pos_y : {pos_y}')
        print(f'pos_z : {pos_z}')
        print(f'turret_x : {turret_x}')
        print(f'turret_y : {turret_y}')
        turret_info.extend([pos_x, pos_y, pos_z, turret_x, turret_y])

    # 이동은 일단 멈춤
    command = {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0},
        "turretRF": {"command": turretRF_cmd, "weight": pitch_weight if turretRF_cmd else 0.0},
        "fire": fire
    }

    print("🔁 Sent Combined Action:", command)
    return jsonify(command)

# 전역 상태 저장
last_bullet_info = {}
turret_info = []
saved_data = []

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global last_bullet_info, turret_info
    last_bullet_info = request.get_json()
    bullet_x = last_bullet_info.get('x', 0)
    bullet_y = last_bullet_info.get('y', 0)
    bullet_z = last_bullet_info.get('z', 0)
    print(f'bullet_x : {bullet_x}')
    print(f'bullet_y : {bullet_y}')
    print(f'bullet_z : {bullet_z}')
    turret_info.extend([bullet_x, bullet_y, bullet_z])

    print("💥 탄 정보 갱신됨:", last_bullet_info)
    return jsonify({"result": "ok"})

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, turret_info, time

    data = request.get_json()
    body_x = data.get('playerBodyX', 0)
    body_y = data.get('playerBodyY', 0)
    body_z = data.get('playerBodyZ', 0)
    time = data.get("time", 0)
    control = ""

    if time > 10:
        control = 'reset'
        last_bullet_info = {}
        turret_info = []

    if last_bullet_info:
        if last_bullet_info.get("hit") == "terrain":
            print("🌀 탄이 지형에 명중! 전차를 초기화합니다.")
            control = 'reset'
            print(f'body_x : {body_x}')
            print(f'body_y : {body_y}')
            print(f'body_z : {body_z}')
            turret_info.extend([body_x, body_y, body_z])
            if len(turret_info) == 11:
                saved_data.append(turret_info[:])
                print(saved_data)
                df = pd.DataFrame(saved_data, columns=["x_pos", "y_pos", "z_pos",
                                                       "x_angle", "y_angle",
                                                       "bullet_x", "bullet_y", "bullet_z",
                                                       "body_x", "body_y", "body_z"])
                df.to_csv("turret_data_body_added.csv", index=False)
            last_bullet_info = {}
            turret_info = []
        else:
            control = "reset"
            last_bullet_info = {}
            turret_info = []

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
    blStartY = 8
    blStartZ = random.uniform(10, 290)
    rlStartX = random.uniform(10, 290)
    rlStartY = 0
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
