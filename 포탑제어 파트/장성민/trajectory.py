from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('./best_8s.pt')

@app.route('/detect', methods=['POST'])
def detect():
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

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
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    import math

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # =============================
    # [1] 기본 파라미터
    # =============================
    g = 9.80665  # 중력 가속도
    v0 = 57.9947  # 초기 발사 속도
    h0 = 2.6369  # 초기 발사 높이 (지면에서 터렛까지 오프셋)

    # =============================
    # [2] 전차 위치, 터렛 각도
    # =============================
    pos = data.get("playerPos", {})
    pos_x = pos.get("x", 0)
    pos_y = pos.get("y", 0)
    pos_z = pos.get("z", 0)

    yaw_deg = data.get("playerTurretX", 0)  # 좌우 회전 (Yaw)
    pitch_deg = data.get("playerTurretY", 0)  # 상하 기울기 (Pitch)

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # =============================
    # [3] 초기 위치, 속도 벡터
    # =============================
    start_x = pos_x
    start_y = pos_y + h0
    start_z = pos_z

    vx = v0 * math.cos(pitch) * math.sin(yaw)
    vy = v0 * math.sin(pitch)
    vz = v0 * math.cos(pitch) * math.cos(yaw)

    # =============================
    # [4] 궤적 계산
    # =============================
    trajectory = []
    t = 0.0
    while t < 10.0:
        x = start_x + vx * t
        y = start_y + vy * t - 0.5 * g * t ** 2
        z = start_z + vz * t
        if y < 0:
            break
        trajectory.append((x, y, z))
        t += 0.1

    # =============================
    # [5] LiDAR 충돌 탐지
    # =============================
    lidar_points = data.get("lidarPoints", [])
    min_distance = float("inf")
    impact_point = None

    for point in lidar_points:
        p = point.get("position", {})
        px = p.get("x")
        py = p.get("y")
        pz = p.get("z")

        for (tx, ty, tz) in trajectory:
            dist = math.sqrt((tx - px)**2 + (ty - py)**2 + (tz - pz)**2)
            if dist < 1.0:  # 허용 충돌 거리
                if dist < min_distance:
                    min_distance = dist
                    impact_point = {"x": px, "y": py, "z": pz}
                break

    # =============================
    # [6] 결과 출력
    # =============================
    # print(f"📍 전차 위치: x={pos_x:.2f}, y={pos_y:.2f}, z={pos_z:.2f}")
    # print(f"🔭 터렛 각도: yaw={yaw_deg:.2f}°, pitch={pitch_deg:.2f}°")
    if impact_point:
        print(f" 예상 충돌 지점: x={impact_point['x']:.2f}, y={impact_point['y']:.2f}, z={impact_point['z']:.2f}")
    # else:
    #     print("❌ 충돌 예측 실패: 포탄이 LiDAR와 교차하지 않음")

    return jsonify({"status": "success", "control": ""})


@app.route('/get_action', methods=['POST'])
def get_action():
    data = request.get_json(force=True)

    position = data.get("position", {})
    turret = data.get("turret", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)

    turret_x = turret.get("x", 0)
    turret_y = turret.get("y", 0)

    print(f"📨 Position received: x={pos_x}, y={pos_y}, z={pos_z}")
    print(f"🎯 Turret received: x={turret_x}, y={turret_y}")

    if combined_commands:
        command = combined_commands.pop(0)
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

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    data = request.get_json()
    if not data:
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    print(f"💥 실제 타격 지점={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
    return jsonify({"status": "OK", "message": "Bullet impact data received"})

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
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 27.23,
        "rdStartX": 59, #Red Start Position
        "rdStartY": 10,
        "rdStartZ": 280,
        "trackingMode": False,
        "detactMode": True,
        "logMode": True,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
