# 0616 병직님 코드에서 buffer 2번 들어간 것 빼고, path[2] 적용

# Y Position : 1.5
# Channel : 65
# Max Distance : 150
# Lidar Position : Turret
from flask import Flask, request, jsonify
import os
from ultralytics import YOLO
import random, math
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from queue import PriorityQueue
import math
import json
import time

# 화면 해상도 (스크린샷 찍었을 때 이미지 크기)
IMAGE_WIDTH = 1921
IMAGE_HEIGHT = 1080

# 카메라 각도도
FOV_HORIZONTAL = 47.81061 
FOV_VERTICAL = 28         

# 터렛 각도 예측 모델 및 전처리기기 파일 경로
MODEL_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/best_dnn_model.h5"
XGB_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/best_xgb_model.pkl"
SCALER_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/scaler.pkl"
POLY_PATH = "/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/turret_final/poly_transformer.pkl"

# 모델 및 전처리기 불러오기
model = load_model(MODEL_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
poly = joblib.load(POLY_PATH)

app = Flask(__name__)
model_yolo = YOLO('/root/jupyter_home/tank_project/ready/포탑제어 파트/오병직/best_8s.pt')

# 적 전차를 찾는 상태
MOVE_MODE = True

# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 20
start_z = 50
start = (start_x, start_z)

# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
destination_x = 260 # 기존에는 destination과 적 전차 위치를 똑같이 줬으나, LiDAR로 물체를 감지할 경우 적 전차도 감지해서 장애물이라 생각하고 목표에 끝까지 도달을 안함. 그래서 이제부터 따로 줌.
destination_z = 46
destination = (destination_x, destination_z)
print(f"🕜️ 초기 destination 설정: {destination}")

INITIAL_YAW = 0.0  # 초기 YAW 값 - 맨 처음 전차의 방향이 0도이기 때문에 0.0 줌. 이를  
current_yaw = INITIAL_YAW  # 현재 차체 방향 추정치 -> playerBodyX로 바꾸면 좋겠으나 실패... playerBodyX의 정보를 받아 오는데 딜레이가 걸린다면 지금처럼 current_yaw값 쓰는게 좋다고 함(by GPT)
previous_position = None  # 이전 위치 (yaw 계산용)
target_reached = False  # 목표 도달 유무 플래그
current_angle = 0.0  # 실제 플레이어의 차체 각도 저장용 (degree) -> playerBodyX 받아오는 방법 사용해 볼 것임.
collision_count = 0  # 충돌 횟수 카운터 추가
total_distance = 0

# 시각화 관련 부분
current_position = None
last_position = None
position_history = []
original_obstacles = []  # 원본 장애물 좌표 저장용 (버퍼 없이)
collision_points = [] # 전역변수에 collision point 추가(충돌 그림에 필요)

astar_how_many_implement = 0 # 0616 희연님 코드 추가

# 충돌 없을 때 파일 저장
with open('collision_points.json', 'w') as f:
    json.dump({
        "collision_count": 0,
        "collision_points": []
    }, f, indent=2)

# 시간 세는 부분
start_time = None
end_time = None

# A* 알고리즘 관련 클래스 및 함수 정의
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    def __lt__(self, other):
        return self.f < other.f

def heuristic(a, b): # Diagonal (Octile) 방식으로 heuristic 변경
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
            # 대각선 이동일 경우 추가 확인
            if dx != 0 and dz != 0:
                if maze[pos[1]][x] == 1 or maze[z][pos[0]] == 1:
                    continue  # 대각선 경로에 인접한 직선 중 하나라도 막혀있으면 skip # 즉 모서리를 못 뚫고 지나가게 수정
            if maze[z][x] == 0: 
                neighbors.append((x, z))
    return neighbors

def a_star(start, goal):
    global astar_how_many_implement # 0616 희연님 코드 추가

    astar_how_many_implement+=1 # 0616 희연님 코드 추가
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

            # 이 부분 추가함.
            dx = abs(nbr[0] - current.position[0])
            dz = abs(nbr[1] - current.position[1])
            step_cost = math.sqrt(2) if dx != 0 and dz != 0 else 1

            node.g = current.g + step_cost
            node.h = heuristic(nbr, goal)
            node.f = node.g + node.h
            open_set.put((node.f, node))
    return [start]

# 현재 위치와 다음 위치 간 각도 계산 함수
def calculate_angle(current, next_pos): # A*알고리즘을 통해서 어디로 갈지 전체 경로를 정했기 때문에 다음 위치로만 가면 됨.
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]
    return (math.degrees(math.atan2(dz, dx)) + 360) % 360

# 전방 장애물 감지 함수_ 기홍님 추가 _0602_ 아침에 깃허브에서 받음
# 함수 설명:이동하기 전에, 지금 위치와 현재 바라보는 방향(yaw)을 기준으로 
# 앞으로 radius만큼 한 칸씩 쭉 살펴봐서, 장애물(maze에서 1로 표시된 곳)이 있으면 미리 감속 -> 회피 시간 늘어남.
def is_obstacle_ahead(pos, yaw, maze, radius=30):
    """
    현재 yaw(도 단위) 방향 기준 전방 radius만큼 검사.
    장애물(maze=1)이 있으면 True 리턴.
    """
    x, z = pos   # 현 좌표
    rad = math.radians(yaw)   # 현 각도 라디안으로 변경
    dx = math.cos(rad)       
    dz = math.sin(rad)

    for step in range(1, radius + 1):
        nx = int(round(x + dx * step))
        nz = int(round(z + dz * step))
        if 0 <= nx < GRID_SIZE and 0 <= nz < GRID_SIZE:
            if maze[nz][nx] == 1:
                print(f"⚠️ 전방 장애물 감지: ({nx},{nz})")
                return True
    return False

# 맵 유효 위치 확인
def is_valid_pos(pos, size=GRID_SIZE): # 내 전차 혹은 A* 경로가 300x300 안에 있는지 확인
    x, z = pos
    return 0 <= x < size and 0 <= z < size

# 이동 거리 구하는 함수(평가용)
def calculate_actual_path():
    global total_distance
    
    if len(position_history) > 1:
        for i in range(len(position_history) -1):
            x1, z1 = position_history[i] # 이전 좌표
            x2, z2 = position_history[i+1] # 현재 좌표
            step_distance = math.sqrt((x2 - x1)**2 + (z2 - z1)**2) # 가장 최근 두 지점의 좌표 추출
            total_distance += step_distance                        # 지금 이동한 거리(step_distance)를 누적 거리(total_distance)에 더함
    return total_distance

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

    # 전체 라이다 데이터에서 박스안에 해당하는 라이다 포인트만 저장
    candidates = [
        p for p in lidar_points
        if p["isDetected"]
        and abs((p["angle"] - h_angle + 180) % 360 - 180) < h_angle_tol
        and abs(p.get("verticalAngle", 0) + v_angle) < v_angle_tol
    ]

    # 박스가 그려진 각도에 라이다 값이 없다면 (여기가 문제. 라이다 데이터를 촘촘하게 받지 않으면 못찾음.)
    if not candidates:
        print(f'❌ There is no candidates')
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
def match_yolo_to_lidar(bboxes, lidar_points, image_width, image_height, fov_h, fov_v):
    results = []
    for bbox in bboxes:
        h_angle, v_angle = get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_h, fov_v)

        # 바운딩박스 비율 계산
        bbox_width_ratio = (bbox["x2"] - bbox["x1"]) / image_width
        bbox_height_ratio = (bbox["y2"] - bbox["y1"]) / image_height

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
    global lidar_data, enemy_pos, MOVE_MODE, yolo_results

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

    # current_bboxes에 저장되어있는 현재 인식된 전차들의 바운딩 박스 좌표로 그 전차의 실제 좌표값 가져오기
    yolo_results = match_yolo_to_lidar(
        bboxes=current_bboxes,
        lidar_points=lidar_data,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fov_h=FOV_HORIZONTAL,
        fov_v=FOV_VERTICAL
    )   

    if yolo_results and yolo_results[0]['distance'] <= 100:
        MOVE_MODE = False

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

# 여기 리스트에 cmd 2개를 넣는다
combined_command_cache = []

# tank_detected = False     # 0616 쓸모 없어진 부분인듯?
# tank_detect_time = None   # 0616 쓸모 없어진 부분인듯?

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, angle_hist, save_time, len_angle_hist, MOVE_MODE, start_distance, yolo_results
    global target_reached, previous_position, current_yaw, current_position, last_position
    global start_time, end_time
    # global tank_detected, tank_detect_time   # 0616 쓸모 없어진 부분인듯?

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

    print(f'🗺️ MOVE_MODE : {MOVE_MODE}')

    if MOVE_MODE: # 적 전차를 탐색하는 상태일 때
        # 이동 시간, 거리 등 평가용
        if start_time is None: # 추가0605
            start_time = time.time()  
            print("🟢 trackingMode 활성화: 시간 기록 시작")  
        
        if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0: # 거리 5 미만이면 도착으로 간주
            target_reached = True  
            end_time = time.time()  # 추가0605
            elapsed = end_time - start_time  
            print(f"⏱️ 도착까지 걸린 시간: {elapsed:.3f}초")
            print(f"이동거리: {calculate_actual_path():.3f}")
            print("✨ 목표 도달: 전차 정지 플래그 설정")
            
        if target_reached:
            stop_cmd = {k: {'command': 'STOP', 'weight': 1.0} for k in ['moveWS', 'moveAD']}
            return jsonify(stop_cmd)

        if previous_position is not None:
            dx = pos_x - previous_position[0]
            dz = pos_z - previous_position[1]
            if math.hypot(dx, dz) > 0.01:
                current_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
        previous_position = (pos_x, pos_z)

        current_grid = (int(pos_x), int(pos_z))
        path = a_star(current_grid, destination)

        #######################################################################
        # 2 좌표 이동한 후. astar(현좌표, 최종목적지) 함수 실행해서 path 새로 뽑기 반복

        if combined_command_cache:
        # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
            cmd = combined_command_cache.pop(0)
            return jsonify(cmd)
        
        if len(path) > 2:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
            next_grid = path[1:3]  # 1~2번째 좌표 참조
        elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
            next_grid = [path[1]]      # 한개씩 참조  
        else: 
            next_grid = [current_grid]   # 0개면 멈춰라! 도착한거니까!

        for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1

            # next_grid[1]의 회전 각도는 current 가 아니라 next_grid[0]에서 계산해야 맞음 
            base_pos = current_grid if i == 0 else next_grid[i - 1]  
        
            if not is_valid_pos(next_grid[i]):  # 가야하는 곳이 맵 외에 있으면 움직이는거 멈춤
                stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
                stop_cmd['fire'] = False
                return jsonify(stop_cmd)

            target_angle = calculate_angle(base_pos, next_grid[i])  # 현재 좌표에서 두번째 좌표로
            diff = (target_angle - current_yaw + 360) % 360   # 현 각도랑 틀어야할 각도 차이 알아내고
            if diff > 180:  # 이거는 정규화 비슷
                diff -= 360

            # 이건 그냥 유클리드 거리. sqrt는 제곱근! 현위치랑 목적좌표까지의 거리
            # 목적지 근처에 가면 천천히 뒷 키 잡으려고, 목적지까지 남은 거리를 계산.
            distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)
    
            # 전방 장애물 감지 _ 기홍님이 새로 추가 0602_ 오늘 아침에 깃허브에서 받음
            # 장애물이 눈 앞에 있으면 뒷 키 잡으려고, 장애물이 눈 앞에 있는지 판단.
            ahead_obstacle = is_obstacle_ahead(base_pos, current_yaw, maze)

            if distance < 50 :   # 앞으로 가는 weight
                w_weight = 0.2
                acceleration = 'S'
            elif ahead_obstacle:
                w_weight = 0.15  # 전방에 장애물 있을 경우 감속
                acceleration = 'S'
            else:
                w_weight = 0.45
                acceleration = 'W'

            abs_diff = abs(diff)
            if 0 < abs_diff < 30 :  
                w_degree = 0.3
            elif 30 <= abs_diff < 60 :    
                w_degree = 0.6
                stop = True
            elif 60 <= abs_diff < 90 : 
                w_degree = 0.75
            else :
                w_degree = 1.0
        
            forward = {'command': acceleration, 'weight': w_weight}
            turn = {'command': 'A' if diff > 0 else 'D', 'weight': w_degree}

            cmd = {
                'moveWS': forward,
                'moveAD': turn
            }

            combined_command_cache.append(cmd)   # 두 좌표에 대한 명령값 2개가 여기 리스트에 저장됨

        # 처음 1회 A* 경로 계산_ 출발할 때 구한 A* 경로 시각화용
        if len(position_history) == 0:
            path = a_star((int(pos_x), int(pos_z)), destination)  # 현 위치에서 최종 목적지까지 다시 계산
            df = pd.DataFrame(path, columns=["x", "z"])
            df.to_csv("a_star_path.csv", index=False)

        # 내 전차의 이동 경로 시각화용
        if current_grid:
            last_position = current_grid
        position_history.append(current_grid)
        
        df = pd.DataFrame(position_history, columns=["x", "z"])
        df.to_csv("tank_path0.csv", index=False)


        # print문 살짝 수정-희연
        print(f"📍 현재 pos=({pos_x:.1f},{pos_z:.1f})") # yaw={current_yaw:.1f} 두번째 좌표로 가는 앵글 ={target_angle:.1f} 차이 ={diff:.1f}")
        print(f"🚀 cmd 2개 {combined_command_cache}")
        cmd = combined_command_cache.pop(0)
        return jsonify(cmd)

    else: # 적 전차를 찾았다면 (화면에 적 전차에 대한 바운딩 박스가 그려져 있다면)
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

        player_pos = {"x": pos_x, "y": pos_y, "z": pos_z}
        enemy_pos = {"x": enemy_x, "y": enemy_y, "z": enemy_z}

        # 수평 각도 계산
        target_yaw = get_yaw_angle(player_pos, enemy_pos)

        # 모델 입력을 위한 거리 계산
        distance = math.sqrt(
            (pos_x - enemy_x)**2 +
            (pos_y - enemy_y)**2 +
            (pos_z - enemy_z)**2
        )

        # 모델 입력을 위한 dy 계산
        dy = pos_y - enemy_y

        # 5번 맵 테스트용으로 내 전차랑 적 전차가 맵밖으로 떨어지면 reset
        if pos_y < 5 or enemy_y < 5:
            last_bullet_info = {'x':None, 'y':None, 'z':None, 'hit':None}

        # y축 (pitch) 각도 예측 후 앙상블
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
            w = min(max(abs(diff) / 30, 0.01), 1.0)  # 30도 내외로 가중치 조절 예시
            return w
        
        # 최소 가중치 0.1 설정, 최대 1.0 제한
        def calc_pitch_weight(diff):
            w = min(max(abs(diff) / 30, 0.1), 1.0)  # 30도 내외로 가중치 조절 예시
            return w

        # 위 두 함수에서 최소 가중치를 낮게 할수록 조준 속도는 낮아지지만 정밀 조준 가능
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

# DBSCAN 대체 방안 함수... 인접한 좌표들의 거리 차이를 통해서 라벨링을 함.
# 단점?_ 값이 자주 튀는 언덕이나 곡선이면 연결된 선의 형태라도 나뉘어질 수 있다... 일단 동작에는 문제 없음
def split_by_distance(lidar_data):   
    x = lidar_data['x'].astype(int)
    z = lidar_data['z'].astype(int)
    
    coords = np.column_stack((x, z))
    dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    
    threshold = 3.0  # 연결 판단 거리
    split_idx = np.where(dist > threshold)[0] + 1
    
    # 그룹 ID 생성
    group_ids = np.zeros(len(x), dtype=int)
    for i, idx in enumerate(split_idx):
        group_ids[idx:] += 1
    
    # 그룹 ID를 데이터프레임에 추가
    lidar_data['line_group'] = group_ids

    # ✅ 그룹별 개수 계산
    group_counts = lidar_data['line_group'].value_counts()

    # ✅ 너무 크거나 너무 작은 그룹 제거 (45 이상 또는 5 이하)
    # bad_groups = group_counts[(group_counts >= 45) ].index  # | (group_counts <= 5)
    # lidar_data = lidar_data[~lidar_data['line_group'].isin(bad_groups)].reset_index(drop=True)

    return lidar_data

def detect_obstacle_and_hill(df):

    hill_groups = set()  # 언덕 그룹 저장용...
    
    for i in df['line_group'].unique():
        group = df[df['line_group'] == i]
        x = group['x'].astype(int)
        z = group['z'].astype(int)

        
        coords = list(zip(x, z))  # 좌표 튜플로 묶음.
        # print("raw  좌표값: ",coords)

        no_dup_coords = list(dict.fromkeys(coords))  # 계산량을 줄이기 위해서 중복은 줄임.  
        # print("중복 제거 좌표값: ", no_dup_coords)

        if len(coords) <= 2:  # 데이터 너무 적으면 언덕 취급
            hill_groups.add(i)
            continue
                    
        if len(coords) > 50:  # 데이터 과다 = 언덕
            hill_groups.add(i)
            continue

        print(f"Group {i}: {len(group)} points")
        
        arr = np.array(no_dup_coords)  # 차이 계산을 위해서 리스트로 풀어줌.
        dx = np.diff(arr[:, 0])        # x 값들만 뽑아서 차이 계산
        dz = np.diff(arr[:, 1])
    
        angles = np.arctan2(dx, dz)
        angle_deg = np.degrees(angles)  # 우리가 아는 각도 값으로 바꿈
    
        angle_diff_deg = np.diff(angle_deg) # 각도의 차이를 알자_ 확실한거는 다 0이면 직선이라는 것!!
        sum_angle = sum(angle_diff_deg)

        if 3 <= len(coords) <= 4:   # 4개에서 3개인데 직선이면...
            if np.all(np.abs(sum_angle) < 1):
                print("⚠️ small wall (데이터 부족하지만 직선)")  # 소형벽
                continue
        elif len(coords) <= 5:
            print("❌ 데이터 부족하고 직선도 아님 → 제외")
            hill_groups.add(i)
            continue

        # 각도가 잘 가다가 갑자기 90도로 꺾일때(차이)를 봐야하니까 angle_diff_deg 가 맞음. 
        # angle_deg면 90도 방향의 직선에서 문제 생김!!!!
        # 90도나 270이 생길 수 있음.
        sharp_turns = np.sum((np.abs(angle_diff_deg) >= 80) & (np.abs(angle_diff_deg) <= 100) |
                             (np.abs(angle_diff_deg) >= 260) & (np.abs(angle_diff_deg) <= 280))   

        loose_turns = np.sum((np.abs(angle_diff_deg) <= 50) & (np.abs(angle_diff_deg) > 0))    # 곡선 판단용...

    
        if sum_angle == 0 and sharp_turns == 0 and loose_turns <= 2:
            print(f"ㅡ ㅣ 장애물")
            
        # 대신 sum_angle이 0은 아님,...   // and abs(sum_angle) == 90   이거 270이 될 수도 있음
        elif sharp_turns == 1  and loose_turns <=1 and (abs(sum_angle) == 90 or abs(sum_angle) == 270):   
            print(f"ㄱ 장애물_loose_turns : {loose_turns}, sum_angle: {sum_angle}")
            
         # 급하게 꺾이는 구간이 3개 이상이고(전차는 꺾임 구간이 2개라서 혹시 몰라서 임시방편으로...) 
        # and 각도가 느슨하게 꺾이는 것이 3번 이상 발생하면 언덕...
        elif sharp_turns > 1 and loose_turns >=3:  
            print("급변하는 언덕")
            hill_groups.add(i)
            
        elif sharp_turns and loose_turns:  # 급하게 꺾이는 구간은 없지만 느슨하게 서서히 꺾일 때
            print("느슨한 언덕")
            hill_groups.add(i)
        else:  
            # 이 부분 추후 수정 필요...
            print(f"분류안함(언덕)_sum_angle: {sum_angle}, sharp_turns: {sharp_turns}, loose_turns: {loose_turns}")
            hill_groups.add(i)
        print()

    # print(f"hill_groups: {hill_groups}")
    return hill_groups

def map_obstacle(only_obstacle_df):
    global maze, original_obstacles  # <- 전역 변수 선언
    
    for i in only_obstacle_df['line_group'].unique():
        obstacle_points = only_obstacle_df[only_obstacle_df['line_group'] == i]
        x_min_raw = int(np.min(obstacle_points['x']))   # x 값의 최소, 최대
        x_max_raw = int(np.max(obstacle_points['x']))
        z_min_raw = int(np.min(obstacle_points['z']))  # z 값의 최소 최대
        z_max_raw = int(np.max(obstacle_points['z']))

        # ✅ 시각화용 원본 좌표 저장
        original_obstacles.append({
            "x_min": x_min_raw,
            "x_max": x_max_raw,
            "z_min": z_min_raw,
            "z_max": z_max_raw
        })

        # 👉 A*용 maze에는 buffer 적용
        buffer = 5
        x_min = max(0, x_min_raw - buffer)
        x_max = min(GRID_SIZE - 1, x_max_raw + buffer)
        z_min = max(0, z_min_raw - buffer)
        z_max = min(GRID_SIZE - 1, z_max_raw + buffer)

        # map에 적용. 따로 일반 함수로 빼놔도 좋을 듯...
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                if maze[z][x] == 0:  # 이미 마킹된 경우는 생략
                    maze[z][x] = 1

enemy_pos = {} # 적 전차의 위치
true_hit_ratio = [] # 평가를 위해서 사용했던 변수
s_time = 0 # 시뮬레이터 시간

@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, true_hit_ratio, s_time, lidar_data, MOVE_MODE, enemy_pos
    global maze, original_obstacles

    maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    data = request.get_json()
    lidar_data = data.get('lidarPoints', [])
    s_time = data.get("s_time", 0)
    # body_x = data.get('playerBodyX', 0)
    # body_y = data.get('playerBodyY', 0)
    # body_z = data.get('playerBodyZ', 0)
    control = ""

    drive_lidar_data = [  
        (pt["position"]["x"], pt["position"]["z"]) # ,pt["position"]["y"])
        for pt in data.get("lidarPoints", [])
        if pt.get("verticalAngle", 0) <= 1.045 and pt.get("isDetected", False) == True
    ]
    if not drive_lidar_data:
        print("라이다 감지되는 것 없음")
        return jsonify({"status": "no lidar points"})

    # 라이다 데이터 -> df로 변환...
    lidar_df = pd.DataFrame(drive_lidar_data, columns=['x', 'z']) 
    split_lidar_df = split_by_distance(lidar_df)  # line_group 이라는 칼럼이 추가된 형태가 됨

    hill_groups = detect_obstacle_and_hill(split_lidar_df)  # 언덕으로 분류된 line_group 값을 알아옴
    if hill_groups:  # 언덕으로 분류된게 있으면
        only_obstacle_df = split_lidar_df[~split_lidar_df['line_group'].isin(hill_groups)]  # 언덕으로 분류된 것 죄다 버리기...
    else:
        only_obstacle_df = split_lidar_df

    if len(only_obstacle_df) == 0:
        print("감지되는 장애물 없음")
        # continue  #  ..?
        # return jsonify({"status": "no obstacles detected"})  # 끝내기.
    else:
        map_obstacle(only_obstacle_df)

    try:
        json_path = os.path.join(os.path.dirname(__file__), "original_obstacles.json")
        with open(json_path, "w") as f:
            json.dump(original_obstacles, f, indent=2)
        print("✅ original_obstacles.json 저장 완료")

        np.save("maze.npy", np.array(maze))
        np.savetxt("maze.csv", np.array(maze), fmt="%d", delimiter=",")
    except Exception as e:
        print(f"❌ 장애물 저장 실패: {e}")

    # 발사된 탄이 어딘가에 떨어졌을 때
    if last_bullet_info:
        # 지형에 맞았다면
        if last_bullet_info.get("hit") == "terrain":
            print("🌀 탄이 지형에 명중! 전차를 초기화합니다.")
            MOVE_MODE = True
            control = ""
            last_bullet_info = {}
            enemy_pos = {}

        # 적 전차에 맞았다면
        if last_bullet_info.get("hit") == "enemy":
            print("🌀 탄이 적 전차에 명중! 전차를 초기화합니다.")
            MOVE_MODE = True
            control = ""
            last_bullet_info = {}
            enemy_pos = {}
        # 탄이 맞지않고 다양한 이유로 reset을 시킬 때
        else:
            control = ""
            MOVE_MODE = True
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
    global collision_points, collision_count

    d = request.get_json(force=True)
    p = d.get('position', {})
    x = p.get('x')
    z = p.get('z')

    if x is not None and z is not None:
        collision_points.append((x, z))
        collision_count += 1  # 충돌 횟수 증가

        # 저장 파일 구조: 충돌 좌표 목록과 총 횟수 포함
        save_data = {
            "collision_count": collision_count,
            "collision_points": collision_points
        }

        with open('collision_points.json', 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"💥 Collision #{collision_count} at ({x}, {z})")

    return jsonify({'status': 'success', 'collision_count': collision_count})

#Endpoint called when the episode starts
@app.route('/init', methods=['GET'])
def init():
    global start_distance, MOVE_MODE, last_bullet_info, enemy_pos
    global current_yaw, previous_position, target_reached
    current_yaw = INITIAL_YAW
    previous_position = None
    target_reached = False

    MOVE_MODE = True
    last_bullet_info = {}
    enemy_pos = {}

    print("🛠️ /init 라우트 진입 확인!")

    config = {
        "startMode": "start",
        "blStartX": start_x, 
        "blStartY": 10, 
        "blStartZ": start_z,
        "rdStartX": 160, 
        "rdStartY": 10, 
        "rdStartZ": 260,
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