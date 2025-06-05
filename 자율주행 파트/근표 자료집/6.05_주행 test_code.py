# Flask 및 필요한 라이브러리 불러오기
from flask import Flask, request, jsonify
from queue import PriorityQueue
import os
import torch
from ultralytics import YOLO
import math
import cv2
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
import time

# Flask 앱 초기화 및 YOLO 모델 로드
app = Flask(__name__)
model = YOLO('yolov8n.pt')

# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 260
start_z = 13
start = (start_x, start_z)
# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
destination_x = 150
destination_z = 275
destination = (destination_x, destination_z)
print(f"🕜️ 초기 destination 설정: {destination}")

INITIAL_YAW = 90.0  # 초기 YAW 값 - 맨 처음 전차의 방향이 0도이기 때문에 0.0 줌. 이를  
current_yaw = INITIAL_YAW  # 현재 차체 방향 추정치 -> playerBodyX로 바꾸면 좋겠으나 실패... playerBodyX의 정보를 받아 오는데 딜레이가 걸린다면 지금처럼 current_yaw값 쓰는게 좋다고 함(by GPT)
previous_position = None  # 이전 위치 (yaw 계산용)
target_reached = False  # 목표 도달 유무 플래그
current_angle = 0.0  # 실제 플레이어의 차체 각도 저장용 (degree) -> playerBodyX 받아오는 방법 사용해 볼 것임.

# 시간 세는 부분
start_time = None
end_time = None


# 이동 경로 그림 그릴 때 필요함.
current_position = None
last_position = None
position_history = []

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

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
            node.g = current.g + 1
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
        "rdStartX": destination_x, "rdStartY": 10, "rdStartZ": destination_z,
        "trackingMode": False, "detectMode": False, "logMode": False,
        "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000
    }
    print("🛠️ /init config:", config)
    return jsonify(config)


@app.route('/get_action', methods=['POST'])
def get_action():
    global target_reached, previous_position, current_yaw, current_position, last_position
    global turn_state, turn_timestamp
    global start_time, end_time

    data = request.get_json(force=True)
    pos = data.get('position', {})
    pos_x = float(pos.get('x', 0))
    pos_z = float(pos.get('z', 0))
    
    # tracking_mode가 True일 때만 시간 측정 시작
    if start_time is None: # 추가0605
        start_time = time.time()   # 추가0605
        print("🟢 trackingMode 활성화: 시간 기록 시작")  # 추가0605
        
    if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0:
        target_reached = True
        end_time = time.time()  # 추가0605
        elapsed = end_time - start_time  # 추가0605
        print(f"⏱️ 도착까지 걸린 시간: {elapsed:.3f}초")# 추가0605        
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
    next_grid = path[1] if len(path) > 1 else current_grid

    if not is_valid_pos(next_grid):
        stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
        stop_cmd['fire'] = False
        return jsonify(stop_cmd)

    target_angle = calculate_angle(current_grid, next_grid)
    diff = (target_angle - current_yaw + 360) % 360
    if diff > 180:
        diff -= 360

    distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)

    if distance < 60:
        w_weight = 0.2
    else:
        w_weight = 0.7

    if 0 < abs(diff) < 30:
        w_degree = 0.3
    elif 30 <= abs(diff) < 60:
        w_degree = 0.6
    elif 60 <= abs(diff) < 90:
        w_degree = 0.75
    else:
        w_degree = 1.0

    # 타임스탬프 변수 초기화
    if 'turn_state' not in globals():
        turn_state = 'IDLE'
        turn_timestamp = time.time()

    cmd = {}
    now = time.time()

    if abs(diff) >= 110:
        elapsed = now - turn_timestamp

        if turn_state == 'IDLE':
            # 멈추고 대기 1초
            cmd = {
                'moveWS': {'command': 'STOP', 'weight': 1.0},
                'moveAD': {'command': 'STOP', 'weight': 0.0}
            }
            if elapsed >= 0.2:
                turn_state = 'TURN'
                turn_timestamp = now

        elif turn_state == 'TURN':
            # 회전 1초
            cmd = {
                'moveWS': {'command': 'STOP', 'weight': 1.0},
                'moveAD': {'command': 'A' if diff > 0 else 'D', 'weight': 1.0}
            }
            if elapsed >= 0.1:
                turn_state = 'STOP_AFTER_TURN'
                turn_timestamp = now

        elif turn_state == 'STOP_AFTER_TURN':
            # 회전 후 멈춤 1초
            cmd = {
                'moveWS': {'command': 'STOP', 'weight': 1.0},
                'moveAD': {'command': 'STOP', 'weight': 0.0}
            }
            if elapsed >= 0.1:
                turn_state = 'FORWARD'
                turn_timestamp = now

        elif turn_state == 'FORWARD':
            # 전진 1초
            cmd = {
                'moveWS': {'command': 'W', 'weight': 0.3},
                'moveAD': {'command': 'A' if diff > 0 else 'D', 'weight': 0.3}
            }
            if elapsed >= 0.1 or abs(diff) < 10:
                turn_state = 'IDLE'
                turn_timestamp = now

    else:
        # diff 작으면 그냥 전진+조향, 상태 초기화
        cmd = {
            'moveWS': {'command': 'W' , 'weight': 0.3},
            'moveAD': {'command': 'A' if diff > 0 else 'D', 'weight': w_degree}
        }
        turn_state = 'IDLE'
        turn_timestamp = now

    # 최초 경로 계산 및 기록
    if len(position_history) == 0:
        path = a_star((int(pos_x), int(pos_z)), destination)
        df = pd.DataFrame(path, columns=["x", "z"])
        df.to_csv("a_star_path.csv", index=False)

    if current_grid:
        last_position = current_grid
    position_history.append(current_grid)

    df = pd.DataFrame(position_history, columns=["x", "z"])
    df.to_csv("tank_path0.csv", index=False)

    print(f"📍 pos=({pos_x:.1f},{pos_z:.1f}) yaw={current_yaw:.1f} trg={target_angle:.1f} diff={diff:.1f}")
    print(f"🚦 state={turn_state} ⏳ elapsed={now - turn_timestamp:.2f}s 🚀 cmd={cmd}")
    return jsonify(cmd)




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
    
collision_count = 0  # 충돌 횟수 카운터 추가
collision_points = [] # 전역변수에 collision point 추가(충돌 그림에 필요)
original_obstacles = []  # 원본 장애물 좌표 저장용 (버퍼 없이)

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


original_obstacles = []  # 원본 장애물 좌표 저장용 (버퍼 없이)

@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global maze, original_obstacles
    data = request.get_json(force=True)

    obstacles = data.get("obstacles", [])
    print(f"🪨 장애물 업데이트 요청: {len(obstacles)}개")

    for obs in obstacles:
        try:
            # 원본 좌표 저장 (시각화용)
            original_obstacles.append({
                "x_min": int(obs["x_min"]),
                "x_max": int(obs["x_max"]),
                "z_min": int(obs["z_min"]),
                "z_max": int(obs["z_max"]),
            })

            # A* 계산용 좌표는 buffer 포함
            buffer = 7
            x_min = max(0, int(obs["x_min"]) - buffer)
            x_max = min(GRID_SIZE - 1, int(obs["x_max"]) + buffer)
            z_min = max(0, int(obs["z_min"]) - buffer)
            z_max = min(GRID_SIZE - 1, int(obs["z_max"]) + buffer)

            for x in range(x_min, x_max + 1):
                for z in range(z_min, z_max + 1):
                    maze[z][x] = 1

        except KeyError as e:
            print(f"❌ 누락된 키: {e}")

    # 장애물 저장 (옵션)
    np.save("maze.npy", np.array(maze))
    np.savetxt("maze.csv", np.array(maze), fmt="%d", delimiter=",")

    # ✅ 원본 장애물 저장
    try:
        json_path = os.path.join(os.path.dirname(__file__), "original_obstacles.json")
        with open(json_path, "w") as f:
            json.dump(original_obstacles, f, indent=2)
        print("✅ original_obstacles.json 저장 완료!")
    except Exception as e:
        print(f"❌ JSON 저장 실패: {e}")

    return jsonify({"status": "OK", "count": len(obstacles)})



@app.route('/info', methods=['POST'])
def info():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

      

    # 전체 구조 출력 (디버그용)
    # print("📨 /info data received:", data)

    # playerBodyX에서 각도 추출
    global current_angle
    current_angle = float(data.get("playerBodyX", current_angle))

    return jsonify({"status": "success"})

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)