# | 현각도 - 목표 각도| 값이 30 이상이면 멈췄다가감 추가_ 희연연
# path 2개 이동후 재계산 추가_ 희연
# 장애물 근접시 속도 줄이기 추가_김기홍님
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

# Flask 앱 초기화 및 YOLO 모델 로드
app = Flask(__name__)
model = YOLO('yolov8n.pt')

# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 50
start_z = 50
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

# 시각화 관련 부분

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
        "rdStartX": destination_x, "rdStartY": 10, "rdStartZ": destination_z,
        "trackingMode": False, "detectMode": False, "logMode": False,
        "enemyTracking": False, "saveSnapshot": False,
        "saveLog": False, "saveLidarData": False, "lux": 30000
    }
    print("🛠️ /init config:", config)
    return jsonify(config)

# 여기 리스트에 cmd 2개를 넣는다
combined_command_cache = []

@app.route('/get_action', methods=['POST'])
def get_action():
    global target_reached, previous_position, current_yaw, current_position, last_position
    data = request.get_json(force=True)
    pos = data.get('position', {})
    pos_x = float(pos.get('x', 0))
    pos_z = float(pos.get('z', 0))

    if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0:
        target_reached = True
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

    ####################### 여기서부터 해보기 (희연)################################################################
    # 2 좌표 이동한 후. astar(현좌표, 최종목적지) 함수 실행해서 path 새로 뽑기 반복

    # 예전 코드
    # next_grid = path[1] if len(path) > 1 else current_grid

    if combined_command_cache:
    # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
        cmd = combined_command_cache.pop(0)
        return jsonify(cmd)

    
    if len(path) > 2:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
        next_grid = path[1:3]  # 두번째 좌표 참조
    elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
        next_grid = [path[1]]      # 한개씩 참조  
    else: 
        next_grid = current_grid   # 0개면 멈춰라! 도착한거니까!

    for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1

        # next_grid[1]의 회전 각도는 current 가 아니라 next_grid[0]에서 게산해야 맞음 
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
        distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)

        # 전방 장애물 감지 _ 기홍님이 새로 추가 0602_ 오늘 아침에 깃허브에서 받음
        ahead_obstacle = is_obstacle_ahead(base_pos, current_yaw, maze)

        if distance < 50 :   # 앞으로 가는 weight
            w_weight = 0.2
            acceleration = 'S'
        elif ahead_obstacle:
            w_weight = 0.2  # 전방에 장애물 있을 경우 감속
            acceleration = 'S'
        else:
            w_weight = 0.5
            acceleration = 'W'


        # 각도가 많이 꺾이면 멈췄다가 가기_희연 
        #여기에 추가로 stop을 넣어야함.
        abs_diff = abs(diff)
        stop = 30 <= abs_diff # 틀어야하는 각도가 30도 이상이면 stop 은 true! 그 아래면 false!!

        if 0 < abs_diff < 30 :  
            w_degree = 0.3
        elif 30 <= abs_diff < 60 :    
            w_degree = 0.6
        elif 60 <= abs_diff < 90 : 
            w_degree = 0.75
        else :
            w_degree = 1.0
    
        forward = {'command': acceleration, 'weight': w_weight}
        turn = {'command': 'A' if diff > 0 else 'D', 'weight': w_degree}

        cmd = {
            'moveAD': turn,
            'moveWS': forward  # 여기 바꿈꿈
        }

        combined_command_cache.append(cmd)   # 두 좌표에 대한 명령값 2개가 여기 리스트에 저장됨

        if stop:
            print("멈추고 갈게요!")
            cmd_stop = {
                'moveWS': {'command': "STOP", 'weight': 1.0},
                'moveAD': {'command': "", 'weight': 0.0}
            }

            combined_command_cache.append(cmd_stop)

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
    print(f"🚀 cmd 2개 이상 {combined_command_cache}")
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
    d = request.get_json(force=True)
    obj = d.get('objectName')
    p = d.get('position', {})
    print(f"Collision {obj} at ({p.get('x')},{p.get('y')},{p.get('z')})")
    return jsonify({'status': 'success', 'message': 'Collision received'})


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