from flask import Flask, request, jsonify
import os
import torch
import numpy as np
from ultralytics import YOLO
import math

app = Flask(__name__)
model = YOLO('/home/user/jupyter_home/tank/yolo8s_best.pt')

# 카메라 파라미터 (가정: 실제 값은 시뮬레이터 설정에 따라 조정 필요)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 384
FOV_HORIZONTAL = 47.81061  # 수평 FOV (도)
FOV_VERTICAL = 28    # 수직 FOV (도)

# 최신 LiDAR 데이터를 저장하기 위한 전역 변수
latest_lidar_data = None

# 탐지할 전차의 실제 위치 (init 함수에서 가져옴)
# 이 값은 init 함수가 호출될 때 업데이트되거나, 필요에 따라 전역 변수로 관리될 수 있습니다.
# 현재는 초기값으로 설정하며, 실제 시뮬레이션 흐름에 따라 조정이 필요할 수 있습니다.
actual_tank_position = {
    'x': 59,
    'y': 10,
    'z': 280
}

# LiDAR 데이터를 저장하는 엔드포인트 수정
@app.route('/info', methods=['POST'])
def info():
    global latest_lidar_data
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    latest_lidar_data = data  # 최신 LiDAR 데이터 저장
    return jsonify({"status": "success", "control": ""})

# 디버그 로그 포함한 detect 함수
@app.route('/detect', methods=['POST'])
def detect():
    global latest_lidar_data, actual_tank_position
    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    if latest_lidar_data is None:
        print("LiDAR 데이터가 없습니다.")
        return jsonify({"error": "No LiDAR data available"}), 400

    lidar_origin = latest_lidar_data.get('lidarOrigin', {'x': 0, 'y': 0, 'z': 0})
    lidar_rotation = latest_lidar_data.get('lidarRotation', {'x': 0, 'y': 0, 'z': 0})
    lidar_points = latest_lidar_data.get('lidarPoints', [])

    camera_yaw = latest_lidar_data.get('playerTurretX', 0)
    camera_pitch = latest_lidar_data.get('playerBodyY', 0)
    camera_roll = latest_lidar_data.get('playerBodyZ', 0)
    camera_origin = latest_lidar_data.get('turretCameraPos', lidar_origin)
    camera_pos = np.array([camera_origin['x'], camera_origin['y'], camera_origin['z']])

    target_classes = {0: "car1", 1: "car2", 2: "human", 3: "tank"}
    filtered_results = []

    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            print(f'class명: {target_classes[class_id]}, Bound Box의 좌표: {box[0:4]}')

            if class_id == 3:  # tank
                x_min, y_min, x_max, y_max = box[:4]
                bbox_center_x = (x_min + x_max) / 2
                bbox_center_y = (y_min + y_max) / 2
                print(f"[디버그] bbox 중심 좌표: ({bbox_center_x:.1f}, {bbox_center_y:.1f})")

                norm_x = (bbox_center_x / IMAGE_WIDTH) * 2 - 1
                norm_y = (bbox_center_y / IMAGE_HEIGHT) * 2 - 1
                print(f"[디버그] 정규화된 이미지 좌표: ({norm_x:.3f}, {norm_y:.3f})")

                theta_h = norm_x * (FOV_HORIZONTAL / 2) * (math.pi / 180)
                theta_v = -norm_y * (FOV_VERTICAL / 2) * (math.pi / 180)

                # 카메라 기준 방향 벡터
                direction = np.array([
                    math.cos(theta_v) * math.sin(theta_h),
                    math.sin(theta_v),
                    math.cos(theta_v) * math.cos(theta_h)
                ])
                print("[테스트] direction:", direction)
                direction = direction / np.linalg.norm(direction)
                print(f"[디버그] 카메라 기준 광선 방향 벡터: {direction}")

                # 회전행렬 (Yaw(Z) → Pitch(Y) → Roll(X))
                yaw_rad = math.radians(camera_yaw)
                pitch_rad = math.radians(camera_pitch)
                roll_rad = math.radians(camera_roll)

                Rz = np.array([
                    [math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                    [math.sin(yaw_rad),  math.cos(yaw_rad), 0],
                    [0, 0, 1]
                ])
                Ry = np.array([
                    [math.cos(pitch_rad), 0, math.sin(pitch_rad)],
                    [0, 1, 0],
                    [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]
                ])
                Rx = np.array([
                    [1, 0, 0],
                    [0, math.cos(roll_rad), -math.sin(roll_rad)],
                    [0, math.sin(roll_rad), math.cos(roll_rad)]
                ])

                R = Rz @ Ry @ Rx
                world_direction = R @ direction
                world_direction = world_direction / np.linalg.norm(world_direction)

                print(f"[디버그] 월드 기준 광선 방향 벡터: {world_direction}")
                print(f"[디버그] 카메라 위치: {camera_pos}")

                min_dist = float('inf')
                tank_position = None

                for point in lidar_points:
                    if point.get('isDetected', False):
                        pos = point.get('position', {})
                        point_pos = np.array([pos['x'], pos['y'], pos['z']])
                        vec_to_point = point_pos - camera_pos

                        cross_vec = np.cross(world_direction, vec_to_point)
                        perpendicular_dist = np.linalg.norm(cross_vec) / np.linalg.norm(world_direction)
                        dist_along_ray = np.dot(vec_to_point, world_direction)

                        if perpendicular_dist < 1.0 and dist_along_ray > 0:
                            if dist_along_ray < min_dist:
                                min_dist = dist_along_ray
                                tank_position = point_pos
                                
                if tank_position is not None:
                    print(f"탐지된 전차 위치: x={tank_position[0]:.2f}, y={tank_position[1]:.2f}, z={tank_position[2]:.2f}")
    
                    # 실제 위치와의 오차 계산
                    actual_pos_np = np.array([
                        actual_tank_position['x'],
                        actual_tank_position['y'],
                        actual_tank_position['z']
                    ])
                    error = np.linalg.norm(tank_position - actual_pos_np)
                    print(f"**[디버그] 실제 전차 위치: ({actual_pos_np[0]:.2f}, {actual_pos_np[1]:.2f}, {actual_pos_np[2]:.2f})**")
                    print(f"**[디버그] 탐지된 전차 위치와 실제 위치 간 오차: {error:.2f}**")
    
                    # LiDAR 원점과 탐지된 전차 간 거리
                    lidar_origin_np = np.array([
                        lidar_origin['x'],
                        lidar_origin['y'],
                        lidar_origin['z']
                    ])
                    distance_to_lidar = np.linalg.norm(tank_position - lidar_origin_np)
                    print(f"**[디버그] LiDAR 원점과 탐지된 전차 간 거리: {distance_to_lidar:.2f}**")


            filtered_results.append({
                'className': target_classes[class_id],
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

    return jsonify(filtered_results)


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

    # combined_commands가 정의되지 않아 오류가 발생할 수 있으므로, 임시로 빈 리스트로 초기화합니다.
    # 실제 환경에서는 이 변수가 다른 곳에서 채워져야 합니다.
    combined_commands = [] 

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

    print(f"💥 Bullet Impact at X={data.get('x')}, Y={data.get('y')}, Z={data.get('z')}, Target={data.get('hit')}")
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
    global actual_tank_position # 전역 변수 업데이트를 위해 선언
    config = {
        "startMode": "start",  # Options: "start" or "pause"
        "blStartX": 60,  #Blue Start Position
        "blStartY": 10,
        "blStartZ": 200,
        "rdStartX": 60, #Red Start Position
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

    # init 시점에 실제 전차 위치 업데이트
    actual_tank_position['x'] = config['rdStartX']
    actual_tank_position['y'] = config['rdStartY']
    actual_tank_position['z'] = config['rdStartZ']

    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
