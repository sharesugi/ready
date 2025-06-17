# 그림 그리기 코드 적용
# 장애물 뒤에 언덕 있을 경우, 장애물을 인식 못하는 문제를 해결하기 위해서 가장 가까운 포인트에만 Δy 적용
# path[2]가 연산하는데 시간이 오래 걸리는 것 같아서 path[1]으로 바꿈
# 그냥 개수만 받아오던 것에서 x,z 값의 좌표를 통해 하나의 장애물을 하나의 cluster로 묶어서 그 좌표들의 x_min,x_max,z_min,z_max값을 받아옴. 그 값을 기존의 update_obstacle 하던 부분에 넣기!
# LiDAR로부터 감지되는 장애물의 정보를 받아오고자 함. 현재는 그냥 개수만 받아 옴
# 0609 LiDAR 적용을 시작
# 0605_ 시작지점 -> 목적지점 도달 시간 추가_희연
# 0604_휴리스틱 함수 추가
# path 2개 이동후 재계산 추가_ 희연(틀어야할 각도가 클때 멈추는건 뺌. 같이 있으면 성능 안 좋아짐)
# 장애물 근접시 속도 줄이기 추가
# Flask 및 필요한 라이브러리 불러오기
from flask import Flask, request, jsonify
from queue import PriorityQueue
from collections import defaultdict # 가까운 곳에만 Δy 적용할 때 사용함.
from sklearn.cluster import DBSCAN # clustering 작업 - LiDAR에서 장애물 감지시 하나의 장애물을 1도, 2도, ... 의 정보로 받아오므로 걔네를 하나의 군집으로 묶는 역할
import os
import torch
from ultralytics import YOLO
import math
import heapq
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
import time  # 추가0605
import numpy as np

# Flask 앱 초기화 및 YOLO 모델 로드
app = Flask(__name__)
model = YOLO('yolov8n.pt')


# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 40
start_z = 50
start = (start_x, start_z)
# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
destination_x = 250 # 기존에는 destination과 적 전차 위치를 똑같이 줬으나, LiDAR로 물체를 감지할 경우 적 전차도 감지해서 장애물이라 생각하고 목표에 끝까지 도달을 안함. 그래서 이제부터 따로 줌.
destination_z = 50
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

path = a_star(start, destination)  # 현재 A* 결과

# 현재 위치와 다음 위치 간 각도 계산 함수
def calculate_angle(current, next_pos): # A*알고리즘을 통해서 어디로 갈지 전체 경로를 정했기 때문에 다음 위치로만 가면 됨.
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]
    return (math.degrees(math.atan2(dz, dx)) + 360) % 360

# 전방 장애물 감지 함수_ 기홍님 추가 _0602_ 아침에 깃허브에서 받음
# 함수 설명:이동하기 전에, 지금 위치와 현재 바라보는 방향(yaw)을 기준으로 
# 앞으로 radius만큼 한 칸씩 쭉 살펴봐서, 장애물(maze에서 1로 표시된 곳)이 있으면 미리 감지. 
# 그래서 아직 이동하지 않았어도 앞으로 막히는지 미리 확인 가능.
def is_obstacle_ahead(pos, yaw, maze, radius=30):
    """
    현재 yaw(도 단위) 방향 기준 전방 radius만큼 검사.
    장애물(maze=1)이 있으면 True 리턴.
    """
    x, z = pos   # 현좌표
    rad = math.radians(yaw)   # 현각도 라디안으로 변경
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


# 장애물 맵 유효 위치 확인
def is_valid_pos(pos, size=GRID_SIZE): # 장애물이 300x300 안에 있는지 확인
    x, z = pos
    return 0 <= x < size and 0 <= z < size

# Flask API 라우팅 시작
@app.route('/init', methods=['GET'])
def init():
    global current_yaw, previous_position, target_reached
    current_yaw = INITIAL_YAW
    previous_position = None
    target_reached = False

    config = {
        "startMode": "start",
        "blStartX": start_x, "blStartY": 10, "blStartZ": start_z,
        "rdStartX": 160, "rdStartY": 10, "rdStartZ": 260,
        "trackingMode": True, "detectMode": False, "logMode": True,
        "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000
    }
    print("🛠️ /init config:", config)
    return jsonify(config)

def calculate_actual_path():
    global total_distance
    
    if len(position_history) > 1:
        for i in range(len(position_history) -1):
            x1, z1 = position_history[i] # 이전 좌표
            x2, z2 = position_history[i+1] # 현재 좌표
            step_distance = math.sqrt((x2 - x1)**2 + (z2 - z1)**2) # 가장 최근 두 지점의 좌표 추출
            total_distance += step_distance                        # 지금 이동한 거리(step_distance)를 누적 거리(total_distance)에 더함
    return total_distance

    
# 여기 리스트에 cmd 2개를 넣는다
combined_command_cache = []

@app.route('/get_action', methods=['POST'])
def get_action():
    global target_reached, previous_position, current_yaw, current_position, last_position
    global start_time, end_time
    data = request.get_json(force=True)
    pos = data.get('position', {})
    pos_x = float(pos.get('x', 0))
    pos_z = float(pos.get('z', 0))

    # tracking_mode가 True일 때만 시간 측정 시작
    if start_time is None: # 추가0605
        start_time = time.time()  
        print("🟢 trackingMode 활성화: 시간 기록 시작")  
        
    if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0:
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

    if combined_command_cache:  # 명령어가 남아있다면
    # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
        cmd = combined_command_cache.pop(0)
        # print(f"👊두번째 명령어 실행_cmd : {cmd}")
        return jsonify(cmd)
    elif not combined_command_cache: #or combined_command_cache is None:  # 비어있다면 = 명령어 두개 다 실행했다면, 이동 
        # print("combined_command_cache 비어있어서 a_star 실행해요...")
        path = a_star(current_grid, destination)  
    
    if len(path) > 2:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
        next_grid = path[1:3]  # 두번째 좌표 참조
        # print(f"👊👊next_grid가 두개예요: {next_grid}👊👊")
    elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
        next_grid = [path[1]]      # 한개씩 참조  
    else: 
        next_grid = [current_grid]   # 0개면 멈춰라! 도착한거니까!

    for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1
        # print(f"i:{i},  (len(next_grid)) : {len(next_grid)}")
        # next_grid[1]의 회전 각도는 current 가 아니라 next_grid[0]에서 게산해야 맞음 
        base_pos = current_grid if i == 0 else next_grid[i - 1]  

        print(f"next_grid: {next_grid[i]}")
        if not is_valid_pos(next_grid[i]):  # 가야하는 곳이 맵 외에 있으면 움직이는거 멈춤
            stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
            stop_cmd['fire'] = False
            return jsonify(stop_cmd)

        target_angle = calculate_angle(base_pos, next_grid[i])  # 현재 좌표에서 두번째 좌표로
        diff = (target_angle - current_yaw + 360) % 360   # 현 각도랑 틀어야할 각도 차이 알아내고
        if diff > 180:  # 이거는 정규화 비슷
            diff -= 360

        # 이건 그냥 유클리드 거리. sqrt는 제곱근! 현위치랑 목적좌표까지의 거리 
        distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)

        # 전방 장애물 감지 _ 기홍님이 새로 추가 0602_ 오늘 아침에 깃허브에서 받음
        ahead_obstacle = is_obstacle_ahead(base_pos, current_yaw, maze)

        if distance < 25 :   # 앞으로 가는 weight
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

    # 처음 1회 A* 경로 계산_ 기홍님이 새로 추가
    if len(position_history) == 0:
        path = a_star((int(pos_x), int(pos_z)), destination)  # 현 위치에서 최종 목적지까지 다시 계산
        df = pd.DataFrame(path, columns=["x", "z"])
        df.to_csv("a_star_path.csv", index=False)

    
    if current_grid:
        last_position = current_grid
    position_history.append(current_grid)
    
    df = pd.DataFrame(position_history, columns=["x", "z"])
    df.to_csv("tank_path0.csv", index=False)


    # print문 살짝 수정-희연
    print(f"📍 현재 pos=({pos_x:.1f},{pos_z:.1f}) yaw={current_yaw:.1f} 두번째 좌표로 가는 앵글 ={target_angle:.1f} 차이 ={diff:.1f}")
    print(f"🚀 cmd 2개 {combined_command_cache}")
    return jsonify(combined_command_cache.pop(0))



@app.route('/set_destination', methods=['POST'])
def set_destination():
    global destination
    data = request.get_json()
    if not data or 'destination' not in data:
        return jsonify({'status': 'ERROR', 'message': 'Missing destination'}), 400
    try:
        x, y, z = map(float, data['destination'].split(','))
        destination = (int(x), int(z))
        print(f"🎯 destination set to: {destination}")
        return jsonify({'status': 'OK', 'destination': {'x': x, 'y': y, 'z': z}})
    except Exception as e:
        return jsonify({'status': 'ERROR', 'message': str(e)}), 400

@app.route('/start', methods=['GET'])
def start():
    print('start')
    return jsonify({'control': ''})

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


@app.route('/info', methods=['POST'])
def info():
    global maze, original_obstacles

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    lidar_data = data.get("lidarPoints", []) ### 여기서 부터 달라짐 #########################
    if not lidar_data:
        return jsonify({"status": "no lidar points"})

    point_map = defaultdict(lambda: None)

    # 1. 가장 가까운 포인트만 추출 (vertical angle ±10도 필터 포함)
    for point in lidar_data:
        if not point.get("isDetected"):
            continue
        vertical_angle = point.get("verticalAngle", 999)
        if vertical_angle > 5 or vertical_angle < -7 :
            continue

        pos = point.get("position", {})
        x, y, z = pos.get("x"), pos.get("y"), pos.get("z")
        if None in (x, y, z):
            continue

        dist = math.sqrt(x**2 + z**2)
        key = (round(x, 1), round(z, 1))
        prev = point_map[key]

        if prev is None or dist < prev["dist"]:
            point_map[key] = {"x": x, "z": z, "y": y, "dist": dist}

    # 2. Δy 필터 적용
    raw_points = []
    for p in point_map.values():
        x, y, z = p["x"], p["y"], p["z"]

        neighbors = [
            other for other in point_map.values()
            if math.hypot(other["x"] - x, other["z"] - z) < 1.0 and other != p
        ]
        if not neighbors:
            continue

        avg_y = sum(n["y"] for n in neighbors) / len(neighbors)
        dy = abs(y - avg_y)

        if dy > 0.5:
            raw_points.append([x, z])

    if not raw_points:
        # print("⚠️ Δy 필터를 통과한 포인트가 없습니다.")
        return jsonify({"status": "no obstacles"})

    # print(f"✅ Δy 필터 후 남은 포인트 수: {len(raw_points)}")

    # 3. 클러스터링
    points_np = np.array(raw_points)
    clustering = DBSCAN(eps=2.0, min_samples=2).fit(points_np)
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print(f"🧩 감지된 장애물 덩어리 수: {num_clusters}")
    unique_labels = set(labels)
    for label in unique_labels:
        count = sum(labels == label)
        # print(f" - 클러스터 {label}: {count}개 점")

    # 4. 장애물 정보 저장
    # 💡 set(labels)로 반복 (noise 라벨 -1 제외)
    for cluster_id in set(labels):
        if cluster_id == -1:
            continue

        mask = (labels == cluster_id)
        cluster_points = points_np[mask]
    
        if len(cluster_points) == 0:
            # print(f"⚠️ 클러스터 {cluster_id}는 비어 있음")
            continue

        # 👉 buffer 없이 원본 좌표 계산
        x_min_raw = int(np.min(cluster_points[:, 0]))
        x_max_raw = int(np.max(cluster_points[:, 0]))
        z_min_raw = int(np.min(cluster_points[:, 1]))
        z_max_raw = int(np.max(cluster_points[:, 1]))

        # ✅ 시각화용 원본 좌표 저장
        original_obstacles.append({
            "x_min": x_min_raw,
            "x_max": x_max_raw,
            "z_min": z_min_raw,
            "z_max": z_max_raw
        })

        # 👉 A*용 maze에는 buffer 적용
        buffer = 4
        x_min = max(0, x_min_raw - buffer)
        x_max = min(GRID_SIZE - 1, x_max_raw + buffer)
        z_min = max(0, z_min_raw - buffer)
        z_max = min(GRID_SIZE - 1, z_max_raw + buffer)
    
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                maze[z][x] = 1

    # 저장
    try:
        json_path = os.path.join(os.path.dirname(__file__), "original_obstacles.json")
        with open(json_path, "w") as f:
            json.dump(original_obstacles, f, indent=2)
        print("✅ original_obstacles.json 저장 완료")

        np.save("maze.npy", np.array(maze))
        np.savetxt("maze.csv", np.array(maze), fmt="%d", delimiter=",")
    except Exception as e:
        print(f"❌ 장애물 저장 실패: {e}")

    return jsonify({"status": "success", "obstacle_clusters": num_clusters})


# 서버 실행
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5010)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 감지됨 (Ctrl+C)")
    finally:
        print(f"📊 총 충돌 횟수: {collision_count}회")