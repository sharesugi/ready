# Y Position : 1.5
# Channel : 65
# Max Distance : 150
# Lidar Position : Turret
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

# 화면 해상도 (스크린샷 찍었을 때 이미지 크기)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# 카메라 각도
FOV_HORIZONTAL = 47.81061 
FOV_VERTICAL = 28         

# 터렛 각도 예측 모델 및 전처리기기 파일 경로
MODEL_PATH = "./best_dnn_model.h5"
XGB_PATH = "./best_xgb_model.pkl"
SCALER_PATH = "./scaler.pkl"
POLY_PATH = "./poly_transformer.pkl"

# 모델 및 전처리기 불러오기
model = load_model(MODEL_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
poly = joblib.load(POLY_PATH)

app = Flask(__name__)
model_yolo = YOLO('./best_8s.pt')

# 적 전차를 찾는 상태
FIND_MODE = True

# 라이다로 탐지된 전차의 위치와 실제 전차의 위치를 비교하기 위한 리스트
lidar_detect_results = []

# 화면 상에 그려진 바운딩 박스가 라이다의 어느 각도에 있는지를 찾는 함수
def get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_horizontal, fov_vertical):
    # 중심 좌표
    x_center = (bbox["x1"] + bbox["x2"]) / 2
    y_center = (bbox["y1"] + bbox["y2"]) / 2

    # 정규화 (0~1)
    x_norm = x_center / image_width
    y_norm = y_center / image_height

    # 각도 변환 (중앙 기준, 좌/위가 음수, 우/아래가 양수) - 화면상의 위치로 라이다에 매칭시킴
    h_angle = (x_norm - 0.5) * fov_horizontal
    v_angle = (0.5 - y_norm) * fov_vertical  # y축은 반대로 계산 (위가 0)

    return h_angle, v_angle

# 위 함수에서 가져온 각도로 바운딩 박스 안에 찍히는 라이다 값을 모두 가져와 평균값을 return 하는 함수
def find_lidar_cluster_center_adaptive(lidar_points, h_angle, v_angle,
                                       bbox_width_ratio, bbox_height_ratio,
                                       fov_horizontal=47.81061,
                                       fov_vertical=28.0):
    BOX_THRESHOLD = 0.8

    # 바운딩박스 크기에 따라 허용 각도 조정
    h_angle_tol = bbox_width_ratio * fov_horizontal * BOX_THRESHOLD
    v_angle_tol = bbox_height_ratio * fov_vertical * BOX_THRESHOLD

    print(f'🎯 Bbox h_angle : {h_angle_tol}, 🎯Bbox v_angle : {v_angle_tol}')

    # 전체 라이다 데이터에서 박스안에 해당하는 라이다 포인트만 저장
    # candidates 1차 후보 수집
    candidates = [
        p for p in lidar_points
        if p["isDetected"]
        and abs((p["angle"] - h_angle + 180) % 360 - 180) < h_angle_tol
        and abs(p.get("verticalAngle", 0) + v_angle) < v_angle_tol
    ]

    # 필터링 후에도 후보가 없다면 예외 처리
    if not candidates:
        print(f'❌ After z-filter, no candidates remain')
        return None

    # 바운딩 박스 안에 찍힌 라이다 포인트들의 평균 좌표 및 거리
    avg_x = sum(p["position"]["x"] for p in candidates) / len(candidates)
    avg_y = sum(p["position"]["y"] for p in candidates) / len(candidates)
    avg_z = sum(p["position"]["z"] for p in candidates) / len(candidates)
    avg_dist = sum(p["distance"] for p in candidates) / len(candidates)

    return {
        "position": {"x": avg_x, "y": avg_y, "z": avg_z},
        "distance": avg_dist
    }

# 위 두 함수를 사용하여 우리가 필요한 실제 감지된 전차의 좌표를 return 해주는 함수
def match_yolo_to_lidar(bboxes, lidar_points, image_width, image_height, fov_h, fov_v, pitch=0.0, roll=0.0):
    results = []
    for bbox in bboxes:
        h_angle, v_angle = get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_h, fov_v)

        # 바운딩박스 비율 계산
        bbox_width_ratio = (bbox["x2"] - bbox["x1"]) / image_width
        bbox_height_ratio = (bbox["y2"] - bbox["y1"]) / image_height

        print(f'🎯 h_angle : {h_angle} 🎯 v_angle : {v_angle}')

        # LiDAR 클러스터 추정
        cluster = find_lidar_cluster_center_adaptive(
            lidar_points, h_angle, v_angle,
            bbox_width_ratio, bbox_height_ratio,
            fov_horizontal=fov_h,
            fov_vertical=fov_v
        )

        if cluster:
            results.append({
                "bbox": bbox,
                "matched_lidar_pos": cluster["position"],
                "distance": cluster["distance"]
            })
    return results

lidar_data = [] # /info 에서 가져오는 라이다 데이터 저장

@app.route('/detect', methods=['POST'])
def detect():
    global lidar_data, enemy_pos, FIND_MODE, yolo_results

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model_yolo(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {2: "human", 3: "tank"}
    filtered_results = []
    current_bboxes = [] # 인식된 전차의 바운딩 박스 좌표를 저장하기 위한 리스트
    for box in detections:
        if box[4] >= 0.85: # confidence가 0.85 이상인 것만 인식
            class_id = int(box[5])
            if class_id == 3: # 인식된 객체가 전차라면
                FIND_MODE = False # 탐색 중지
                current_bboxes.append({'x1': float(box[0]), 'y1': float(box[1]), 
                                       'x2': float(box[2]), 'y2': float(box[3])})

            if class_id in target_classes:
                filtered_results.append({
                    'className': target_classes[class_id],
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4]),
                    'color': '#00FF00',
                    'filled': False,
                    'updateBoxWhileMoving': True
                })

    # current_bboxes에 저장되어있는 현재 인식된 전차들의 바운딩 박스 좌표로 그 전차의 실제 좌표값 가져오기
    yolo_results = match_yolo_to_lidar(
        bboxes=current_bboxes,
        lidar_points=lidar_data,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fov_h=FOV_HORIZONTAL,
        fov_v=FOV_VERTICAL
    )   

    print(f'🗺️ yolo_results : {yolo_results}')

    # 결과 확인
    for i, r in enumerate(yolo_results):
        enemy_pos['x'] = r['matched_lidar_pos'].get('x', 0)
        enemy_pos['y'] = r['matched_lidar_pos'].get('y', 0)
        enemy_pos['z'] = r['matched_lidar_pos'].get('z', 0)
        print(f"탐지된 전차 {i+1}:")
        print(f"  바운딩 박스: {r['bbox']}")
        print(f"  LiDAR 좌표: {r['matched_lidar_pos']}")
        print(f"  거리: {r['distance']:.2f}m")
        print()

    return jsonify(filtered_results)

# 내 전차의 x, z좌표, 목표 전차의 x, z좌표로 터렛이 바라봐야 하는 x각도 return
# 모델 x 단순 계산
def get_yaw_angle(player_pos, enemy_pos):
    dx = enemy_pos['x'] - player_pos['x']
    dz = enemy_pos['z'] - player_pos['z']

    # atan2는 radian 기준, z를 먼저 넣는 이유는 좌표계 때문
    angle_rad = math.atan2(dx, dz)
    angle_deg = math.degrees(angle_rad)

    # 0~359로 변환
    angle_deg = (angle_deg + 360) % 360

    return round(angle_deg, 2)

# 학습시킨 dnn 모델로 터렛의 y 각도 예측
# 내 전차의 좌표, 적 전차의 좌표를 사용해 거리와 dy를 구하여 입력으로 넣음.
# 출력은 y 각도
def find_angle_for_distance_dy_dnn(distance, dy):
    # ✅ 예측용 입력 설정 (Distance + dy)
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)

    # ✅ 예측
    y_pred = model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

    return float(y_pred_angle)

# 학습시킨 xgb 모델로 터렛의 y 각도 예측
# 내 전차의 좌표, 적 전차의 좌표를 사용해 거리와 dy를 구하여 입력으로 넣음.
# 출력은 y 각도
def find_angle_for_distance_dy_xgb(distance, dy):
    # ✅ 예측용 입력 설정 (Distance + dy)
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)

    # ✅ 예측
    y_pred = xgb_model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

    return float(y_pred_angle)

# 아래 세 변수 모두 사격 불가능한 각도 판별할 때 사용하는 변수
angle_hist = []
save_time = 0
len_angle_hist = -1

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, angle_hist, save_time, len_angle_hist
    global FIND_MODE, start_distance, yolo_results, lidar_detect_results, real_enemy_pos, lidar_rotation

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

    print(f'🗺️ FIND_MODE : {FIND_MODE}')

    if FIND_MODE: # 적 전차를 탐색하는 상태일 때
        # 처음 시작되고 적 전차와 내 전차의 거리가 20 이하 110 이상이면 reset
        if start_distance >= 110 or start_distance <= 20:
            # last_bullet_info에 데이터가 들어가면 reset됨
            last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

        # 적 전차를 탐색하는 상태일 때는 터렛만 반시계방향으로 돌림
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "Q", "weight": 1.0},
            "turretRF": {"command": "turretRF_cmd", "weight": 0.0},
            "fire": False
        }
    else: # 적 전차를 찾았다면 (화면에 적 전차에 대한 바운딩 박스가 그려져 있다면)
        if not yolo_results: # 전차 인식은 됐는데 그에 해당하는 라이다 포인트가 없다면 정지
            command = {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "", "weight": 0.0},
                "turretQE": {"command": "", "weight": 0.0},
                "turretRF": {"command": "turretRF_cmd", "weight": 0.0},
                "fire": False
        }
        else: # 인식도 됐고, 그에 해당하는 라이다 포인트도 있다면
            # 아래 273~284번 줄은 조준 가능한 각도인지 판단하고, 조준불가능한 각도라면 reset하는 코드
            save_time += 1
            if save_time > 10:
                save_time = 0
                angle_hist.append([round(turret_x, 2), round(turret_y, 2)])
                len_angle_hist += 1

            patience = 1 # 3 x n초
            if len_angle_hist > 3:
                if angle_hist[len_angle_hist][:] == angle_hist[len_angle_hist - patience][:]:
                    angle_hist = []
                    len_angle_hist = -1
                    last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}
            
            # 적 위치
            enemy_x = enemy_pos.get("x", 0)
            enemy_y = enemy_pos.get("y", 0)
            enemy_z = enemy_pos.get("z", 0)

            bboxes = yolo_results[0]['bbox']

            player_pos = {"x": pos_x, "y": pos_y, "z": pos_z}
            enemy_pos = {"x": enemy_x, "y": enemy_y, "z": enemy_z}

            # 유클리드 거리 계산 함수
            def euclidean_distance(p1, p2):
                return math.sqrt((p1['x'] - p2['x'])**2 + (p1['y'] - p2['y'])**2 + (p1['z'] - p2['z'])**2)

            # 거리 및 차이 계산
            distance_to_enemy = euclidean_distance(player_pos, real_enemy_pos)
            height_diff = real_enemy_pos['y'] - player_pos['y']  # 부호 유지

            x_diff = real_enemy_pos['x'] - enemy_pos['x']
            y_diff = real_enemy_pos['y'] - enemy_pos['y']
            z_diff = real_enemy_pos['z'] - enemy_pos['z']

            detect_real_diff = euclidean_distance(enemy_pos, real_enemy_pos)

            # 결과 추가
            lidar_detect_results.append({
                "detect_enemy_pos_x": enemy_pos['x'],
                "detect_enemy_pos_y": enemy_pos['y'],
                "detect_enemy_pos_z": enemy_pos['z'],
                "real_enemy_pos_x": real_enemy_pos['x'],
                "real_enemy_pos_y": real_enemy_pos['y'],
                "real_enemy_pos_z": real_enemy_pos['z'],
                "player_pos_x": player_pos['x'],
                "player_pos_y": player_pos['y'],
                "player_pos_z": player_pos['z'],
                "bbox_x1": bboxes['x1'],
                "bbox_y1": bboxes['y1'],
                "bbox_x2": bboxes['x2'],
                "bbox_y2": bboxes['y2'],
                "lidar_yaw": lidar_rotation['y'],
                "lidar_pitch": lidar_rotation['x'],
                "lidar_roll": lidar_rotation['z'],
                "distance_to_enemy": distance_to_enemy,
                "height_diff": height_diff,
                "x_diff": x_diff,
                "y_diff": y_diff,
                "z_diff": z_diff,
                "detect_real_diff": detect_real_diff
            })

            # DataFrame 생성
            df = pd.DataFrame(lidar_detect_results)
            
            file_path = "lidar_detection.csv"
            write_header = not os.path.exists(file_path)

            # 파일에 누적 저장
            df.to_csv(file_path, mode='a', index=False, header=write_header)

            lidar_detect_results = [] # 라이다 리스트 초기화

            # 수평 각도 계산
            target_yaw = get_yaw_angle(player_pos, enemy_pos)

            # 모델 입력을 위한 거리 계산
            distance = math.sqrt(
                (pos_x - enemy_x)**2 +
                (pos_y - enemy_y)**2 +
                (pos_z - enemy_z)**2
            )

            print(f'❌❌❌❌ 거리 오차 {distance - start_distance}')

            # 모델 입력을 위한 dy 계산
            dy = pos_y - enemy_y

            # # 5번 맵 테스트용으로 내 전차랑 적 전차가 맵밖으로 떨어지면 reset
            # if pos_y < 5 or enemy_y < 5:
            #     last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

            # y축 (pitch) 각도 에측 후 앙상블
            target_pitch_dnn = find_angle_for_distance_dy_dnn(distance, dy)
            target_pitch_xgb = find_angle_for_distance_dy_xgb(distance, dy)
            target_pitch = (target_pitch_dnn + target_pitch_xgb) / 2 # 사용할 y 각도

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
                w = min(max(abs(diff) / 15, 0.01), 2.0)  # 30도 내외로 가중치 조절 예시
                return w
            
            # 최소 가중치 0.1 설정, 최대 1.0 제한
            def calc_pitch_weight(diff):
                w = min(max(abs(diff) / 10, 0.01), 3.0)  # 30도 내외로 가중치 조절 예시
                return w

            # 위 두 함수에서 최소 가중치를 낮게 할수록 조준 속도는 낮아지지만 정밀 조준 가능능
            yaw_weight = calc_yaw_weight(yaw_diff)
            pitch_weight = calc_pitch_weight(pitch_diff)

            # 좌우 회전 명령 결정
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
            print(f'🏹target_yaw : {target_yaw}, 🏹target_pitch : {target_pitch}')

            # 이동은 일단 멈춤, 위에서 계산한 각도 오차에 따른 가중치로 조준
            command = {
                "moveWS": {"command": "STOP", "weight": 1.0},
                "moveAD": {"command": "", "weight": 0.0},
                "turretQE": {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0},
                "turretRF": {"command": turretRF_cmd, "weight": pitch_weight if turretRF_cmd else 0.0},
                "fire": aim_ready
            }

    return jsonify(command)

# 전역 상태 저장 (시뮬레이터 reset 시킬 때 사용)
last_bullet_info = {}

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global last_bullet_info
    # 발사한 탄이 지형 / 전차에 떨어 졌는지 저장해주는 변수
    last_bullet_info = request.get_json()
    print("💥 탄 정보 갱신됨:", last_bullet_info)
    return jsonify({"yolo_results": "ok"})

enemy_pos = {} # 적 전차의 위치
true_hit_ratio = [] # 평가를 위해서 사용했던 변수
time = 0 # 시뮬레이터 시간

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, true_hit_ratio, time, lidar_data, FIND_MODE, enemy_pos, real_enemy_pos, lidar_rotation

    data = request.get_json()
    lidar_data = data.get('lidarPoints', [])
    time = data.get("time", 0)
    real_enemy_pos = data.get('enemyPos', 0)
    lidar_rotation = data.get('lidarRotation', 0)
    # body_y = data.get('playerBodyY', 0)
    # body_z = data.get('playerBodyZ', 0)
    control = ""

    # 45초가 지났는데도 탄이 발사되지 않았다면 reset
    # 정확히는 지형 / 전차에 떨어진 탄이 없다면
    if time > 20:
        control = 'reset'
        FIND_MODE = True
        last_bullet_info = {}
        enemy_pos = {}

    # 발사된 탄이 어딘가에 떨어졌을 때
    if last_bullet_info:
        # 지형에 맞았다면
        if last_bullet_info.get("hit") == "terrain":
            print("🌀 탄이 지형에 명중! 전차를 초기화합니다.")
            FIND_MODE = True
            control = "reset"
            # true_hit_ratio.append(0)
            # df = pd.DataFrame(true_hit_ratio, columns=["is_hit"])
            # df.to_csv("true_hit_ratio_map5_YOLO.csv", index=False)
            last_bullet_info = {}
            enemy_pos = {}

        # 적 전차에 맞았다면
        if last_bullet_info.get("hit") == "enemy":
            print("🌀 탄이 적 전차에 명중! 전차를 초기화합니다.")
            FIND_MODE = True
            control = "reset"
            # true_hit_ratio.append(1)
            # df = pd.DataFrame(true_hit_ratio, columns=["is_hit"])
            # df.to_csv("true_hit_ratio_map5_YOLO.csv", index=False)
            last_bullet_info = {}
            enemy_pos = {}
        # 탄이 맞지않고 다양한 이유로 reset을 시킬 때
        else:
            control = "reset"
            FIND_MODE = True
            last_bullet_info = {}
            enemy_pos = {}

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
    global start_distance, FIND_MODE, last_bullet_info, enemy_pos

    FIND_MODE = True
    last_bullet_info = {}
    enemy_pos = {}

    print("🛠️ /init 라우트 진입 확인!")

    # 내 전차, 적 전차 시작 좌표 랜덤값
    blStartX = random.uniform(10, 290)
    blStartY = 10
    blStartZ = random.uniform(10, 290)
    rlStartX = random.uniform(10, 290)
    rlStartY = 10
    rlStartZ = random.uniform(10, 290)

    # 초기 거리 계산 위에서 설정한 조건에 충족하지 않으면 reset 시키기 위해서
    start_distance = math.sqrt(
        (blStartX - rlStartX)**2 +
        (blStartY - rlStartY)**2 +
        (blStartZ - rlStartZ)**2
    )

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
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)