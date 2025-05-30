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

# Flask 앱 초기화 및 YOLO 모델 로드
app = Flask(__name__)
model = YOLO('yolov8n.pt')

# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 250
start_z = 250
start = (start_x, start_z)
# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
destination_x = 40
destination_z = 150
destination = (destination_x, destination_z)
print(f"🕜️ 초기 destination 설정: {destination}")

INITIAL_YAW = 0.0  # 초기 YAW 값 - 맨 처음 전차의 방향이 0도이기 때문에 0.0 줌. 이를  
current_yaw = INITIAL_YAW  # 현재 차체 방향 추정치 -> playerBodyX로 바꾸면 좋겠으나 실패... playerBodyX의 정보를 받아 오는데 딜레이가 걸린다면 지금처럼 current_yaw값 쓰는게 좋다고 함(by GPT)
previous_position = None  # 이전 위치 (yaw 계산용)
target_reached = False  # 목표 도달 유무 플래그

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

def get_neighbors(pos): # 현재 위치에서 갈 수 있는 다음 위치(A*라서 8방향)
    neighbors = []
    for dx, dz in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        x, z = pos[0] + dx, pos[1] + dz
        if 0 <= x < GRID_SIZE and 0 <= z < GRID_SIZE:
            if maze[z][x] == 0: # maze에 0이 있을 때만 갈 수 있음.(1은 장애물이라 못감)
                neighbors.append((x, z))
            else:
                print(f"❌ 차단된 위치: ({x},{z})는 장애물임")
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
    data = request.get_json(force=True)
    pos = data.get('position', {})
    pos_x = float(pos.get('x', 0))
    pos_z = float(pos.get('z', 0))

    if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 10.0:
        target_reached = True
        print("✨ 목표 도달: 전차 정지 플래그 설정")
        
    if target_reached:
        stop_cmd = {k: {'command': 'STOP', 'weight': 1.0} for k in ['moveWS', 'moveAD']}
        return jsonify(stop_cmd)

    # if not (0 <= pos_x < GRID_SIZE and 0 <= pos_z < GRID_SIZE):
    #     stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
    #     return jsonify(stop_cmd)

    if previous_position is not None:
        dx = pos_x - previous_position[0]
        dz = pos_z - previous_position[1]
        if math.hypot(dx, dz) > 0.01:
            current_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
    previous_position = (pos_x, pos_z)

    current_grid = (int(pos_x), int(pos_z))
    path = a_star(current_grid, destination)
    # print_maze_path(path, maze, GRID_SIZE)
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
    
    if distance < 80:
        w_weight = 0.3
    else :
        w_weight = 0.8
    
    forward = {'command': 'W' if distance > 70 else 'S', 'weight': w_weight}
    turn = {'command': 'A' if diff > 0 else 'D', 'weight': min(abs(diff) / 180, 1.0)}

    cmd = {
        'moveWS': forward,
        'moveAD': turn
    }

    if current_grid:
        last_position = current_grid
    position_history.append(current_grid)
    
    df = pd.DataFrame(position_history, columns=["x", "z"])
    df.to_csv("tank_path0.csv", index=False)

    print(f"📍 pos=({pos_x:.1f},{pos_z:.1f}) yaw={current_yaw:.1f} trg={target_angle:.1f} diff={diff:.1f}")
    print(f"🚀 cmd {cmd}")
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

@app.route('/collision', methods=['POST'])
def collision():
    d = request.get_json(force=True)
    obj = d.get('objectName')
    p = d.get('position', {})
    print(f"Collision {obj} at ({p.get('x')},{p.get('y')},{p.get('z')})")
    return jsonify({'status': 'success', 'message': 'Collision received'})

# 장애물 정보 수신용 API 만들기 - 장애물 못 피함.
@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global maze
    data = request.get_json(force=True)

    obstacles = data.get("obstacles", [])
    print(f"🪨 장애물 업데이트 요청: {len(obstacles)}개")
    print("🪨 Obstacle Data:", data)

    for obs in obstacles:
        try:
            # ✅ 여유 영역(padding) 적용
            buffer = 5
            x_min = max(0, int(obs["x_min"]) - buffer)
            x_max = min(GRID_SIZE - 1, int(obs["x_max"]) + buffer)
            z_min = max(0, int(obs["z_min"]) - buffer)
            z_max = min(GRID_SIZE - 1, int(obs["z_max"]) + buffer)

            for x in range(x_min, x_max + 1):
                for z in range(z_min, z_max + 1):
                    if 0 <= x < GRID_SIZE and 0 <= z < GRID_SIZE:
                        maze[z][x] = 1  # ✅ 좌표 순서 유의!
                        print(f"✅ 장애물 등록: ({x}, {z}) → {maze[z][x]}")
        except KeyError as e:
            print(f"❌ 누락된 키: {e} → 해당 장애물은 무시됨")

    # ✅ 장애물 저장 (옵션)
    np.save("maze.npy", np.array(maze))
    np.savetxt("maze.csv", np.array(maze), fmt="%d", delimiter=",")

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
