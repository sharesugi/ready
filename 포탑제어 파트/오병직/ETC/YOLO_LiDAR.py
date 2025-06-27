from flask import Flask, request, jsonify
import os
import torch
from ultralytics import YOLO

IMAGE_WIDTH = 1921
IMAGE_HEIGHT = 1080

FOV_HORIZONTAL = 47.81061  # 도
FOV_VERTICAL = 28          # 도

app = Flask(__name__)
model = YOLO('/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/best_8s.pt')

def get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_horizontal, fov_vertical):
    # 중심 좌표
    x_center = (bbox["x1"] + bbox["x2"]) / 2
    y_center = (bbox["y1"] + bbox["y2"]) / 2

    # 정규화 (0~1)
    x_norm = x_center / image_width
    y_norm = y_center / image_height

    # 각도 변환 (중앙 기준, 좌/위가 음수, 우/아래가 양수)
    h_angle = (x_norm - 0.5) * fov_horizontal
    v_angle = (0.5 - y_norm) * fov_vertical  # y축은 반대로 계산 (위가 0)
    
    return h_angle, v_angle

def find_lidar_cluster_center(lidar_points, h_angle, v_angle, h_angle_tolerance=5, v_angle_tolerance=3):
    # 1. 각도 기준으로 후보 필터링 (isDetected == True)
    candidates = [
        p for p in lidar_points
        if p["isDetected"]
        and abs((p["angle"] - h_angle + 180) % 360 - 180) < h_angle_tolerance
        and abs(p.get("verticalAngle", 0) - v_angle) < v_angle_tolerance
    ]

    if not candidates:
        return None

    # 2. 중심 좌표 계산
    avg_x = sum(p["position"]["x"] for p in candidates) / len(candidates)
    avg_y = sum(p["position"]["y"] for p in candidates) / len(candidates)
    avg_z = sum(p["position"]["z"] for p in candidates) / len(candidates)

    # 3. 가장 가까운 점 기준으로 거리 추정 (optional)
    dist = sum(p["distance"] for p in candidates) / len(candidates)

    return {
        "position": {"x": avg_x, "y": avg_y, "z": avg_z},
        "distance": dist
    }

def match_yolo_to_lidar(bboxes, lidar_points, image_width, image_height, fov_h, fov_v):
    results = []
    for bbox in bboxes:
        h_angle, v_angle = get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_h, fov_v)
        cluster = find_lidar_cluster_center(lidar_points, h_angle, v_angle)
        if cluster:
            results.append({
                "bbox": bbox,
                "matched_lidar_pos": cluster["position"],
                "distance": cluster["distance"]
            })
    return results

lidar_data = []

@app.route('/detect', methods=['POST'])
def detect():
    global lidar_data

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

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
                    'updateBoxWhileMoving': False
                })

    results = match_yolo_to_lidar(
        bboxes=current_bboxes,
        lidar_points=lidar_data,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fov_h=FOV_HORIZONTAL,
        fov_v=FOV_VERTICAL
    )

    # 결과 확인
    for i, r in enumerate(results):
        print(f"탐지된 전차 {i+1}:")
        print(f"  바운딩 박스: {r['bbox']}")
        print(f"  LiDAR 좌표: {r['matched_lidar_pos']}")
        print(f"  거리: {r['distance']:.2f}m")
        print()

    return jsonify(filtered_results)

@app.route('/info', methods=['POST'])
def info():
    global lidar_data

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    lidar_data = data.get('lidarPoints', [])
    
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
    app.run(host='0.0.0.0', port=5005)
