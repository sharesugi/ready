# 0619_ 가져오는 라이더 설정값 변경, info 함수와 info 함수가 쓰는 함수들 위로 가져옴, path 3로 바꿔봄: 일단 적용해보니 부딫히는건 3번중 한번꼴(거의 스치듯이 부딫힘)
# 라이다 고정 설정(포격팀과 통일한)_ interval:0.3   Ypos: 1.61   Channel: 45      minimapChannel: -    max_distance: 110     lidar_position: turret
# 0614_ path 2 추가 
# 0613_split_by_distance: 라이더로 감지한 물체들을 거리가반으로 객체를 나눔 
# 0613_detect_obstacle_and_hill: 각도 계산을 해서 언덕과 장애물 구분 함수
# 0613_map_obstacle: 감지한 장애물을 맵에 반영
# 0609 LiDAR 적용을 시작
# 0605_ 시작지점 -> 목적지점 도달 시간 추가_희연
# 0604_휴리스틱 함수 추가
# path 2개 이동후 재계산 추가_ 희연(틀어야할 각도가 클때 멈추는건 뺌. 같이 있으면 성능 안 좋아짐)
# 장애물 근접시 속도 줄이기 추가
# Flask 및 필요한 라이브러리 불러오기
from flask import Flask, request, jsonify
from queue import PriorityQueue
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
import time  # 추가0605_이동 시간 타이머
import numpy as np

# Flask 앱 초기화 및 YOLO 모델 로드
app = Flask(__name__)
model = YOLO('yolov8n.pt')


# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 20
start_z = 50
start = (start_x, start_z)
# 최종 목적지 위치 - 적 전차는 이 위치에 없음

dest_list = [(260, 46), (228,34),(235,225),(250, 240), (220, 50),(55, 230), (20, 50)]

destination_x = dest_list[1][0] 
destination_z = dest_list[1][1]
destination = (destination_x, destination_z)

# 238, 40
print(f"🕜️ 초기 destination 설정: {destination}")

INITIAL_YAW = 90.0  # 초기 YAW 값 - 맨 처음 전차의 방향이 0도이기 때문에 0.0 줌. 이를  
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

# 여기 리스트에 cmd 2개를 넣는다...  
# combined_command_cache = []

astar_how_many_implement = 0

# 충돌 없을 때 파일 저장
with open('collision_points.json', 'w') as f:
    json.dump({
        "collision_count": 0,
        "collision_points": []
    }, f, indent=2)

# 시간 세는 부분
start_time = None
end_time = None


# DBSCAN 대체 방안 함수... 인접한 좌표들의 거리 차이를 통해서 라벨링을 함.
# 단점?_ 값이 자주 튀는 언덕이나 곡선이면 연결된 선의 형태라도 나뉘어질 수 있다... 일단 동작에는 문제 없
def split_by_distance(lidar_data, threshold=4, min_group_size=4):
    lidar_data = lidar_data.copy()
    lidar_data['line_group'] = -1  # 초기화
    group_counter = 0  # 전역 고유 그룹 ID

    for angle in lidar_data['verticalAngle'].unique():
        group = lidar_data[lidar_data['verticalAngle'] == angle].copy()

        x = group['x'].astype(float)
        z = group['z'].astype(float)
        coords = np.column_stack((x, z))

        if len(coords) < 2:
            continue  # 이미 -1로 되어 있음

        dist = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        split_idx = np.where(dist > threshold)[0] + 1

        local_group_ids = np.zeros(len(group), dtype=int)
        for i, idx in enumerate(split_idx):
            local_group_ids[idx:] += 1

        # 각 소그룹에 대해 처리
        for local_id in np.unique(local_group_ids):
            mask = (local_group_ids == local_id)
            indices = group.index[mask]
            if mask.sum() < min_group_size:
                lidar_data.loc[indices, 'line_group'] = -1
            else:
                lidar_data.loc[indices, 'line_group'] = group_counter
                group_counter += 1

    return lidar_data


def detect_obstacle_and_hill(df):

    hill_groups = set()  # 언덕 그룹 저장용...
    
    for i in df['line_group'].unique():
        group = df[df['line_group'] == i]

        if i == -1:
            hill_groups.add(i)
            continue

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

        # print(f"Group {i}: {len(group)} points")
        
        arr = np.array(no_dup_coords)  # 차이 계산을 위해서 리스트로 풀어줌.
        dx = np.diff(arr[:, 0])        # x 값들만 뽑아서 차이 계산
        dz = np.diff(arr[:, 1])
    
        angles = np.arctan2(dx, dz)
        angle_deg = np.degrees(angles)  # 우리가 아는 각도 값으로 바꿈
    
        angle_diff_deg = np.diff(angle_deg) # 각도의 차이를 알자_ 확실한거는 다 0이면 직선이라는 것!!
        sum_angle = sum(angle_diff_deg)

        if 3 <= len(coords) <= 4:   # 4개에서 3개인데 직선이면...
            if np.all(np.abs(sum_angle) < 1):
                # print("⚠️ small wall (데이터 부족하지만 직선)")  # 소형벽
                continue
        elif len(coords) <= 5:
            # print("❌ 데이터 부족하고 직선도 아님 → 제외")
            hill_groups.add(i)
            continue

        # 각도가 잘 가다가 갑자기 90도로 꺾일때(차이)를 봐야하니까 angle_diff_deg 가 맞음. 
        # angle_deg면 90도 방향의 직선에서 문제 생김!!!!
        # 90도나 270이 생길 수 있음.
        sharp_turns = np.sum((np.abs(angle_diff_deg) >= 80) & (np.abs(angle_diff_deg) <= 100) |
                             (np.abs(angle_diff_deg) >= 260) & (np.abs(angle_diff_deg) <= 280))   

        loose_turns = np.sum((np.abs(angle_diff_deg) <= 50) & (np.abs(angle_diff_deg) > 0))    # 곡선 판단용...

    
        if sum_angle == 0 and sharp_turns == 0 and loose_turns == 0:
            # print(f"ㅡ ㅣ 장애물_ len(coords): {len(coords)}")
            continue
            
        # 대신 sum_angle이 0은 아님,...   // and abs(sum_angle) == 90   이거 270이 될 수도 있음
        elif sharp_turns == 1  and loose_turns <=1 and (abs(sum_angle) == 90 or abs(sum_angle) == 270):   
            print("ㄱ 장애물")  #_loose_turns : {loose_turns}, sum_angle: {sum_angle}, sharp_turns: {sharp_turns}")
            continue
         # 급하게 꺾이는 구간이 3개 이상이고(전차는 꺾임 구간이 2개라서 혹시 몰라서 임시방편으로...) 
        # and 각도가 느슨하게 꺾이는 것이 3번 이상 발생하면 언덕...
        elif sharp_turns > 1 and loose_turns >=3:  
            print("언덕")
            hill_groups.add(i)
            
        elif sharp_turns and loose_turns:  # 급하게 꺾이는 구간은 없지만 느슨하게 서서히 꺾일 때
            print("언덕")
            hill_groups.add(i)
        else:  
            # 이 부분 추후 수정 필요...
            # print(f"분류안함(언덕)_sum_angle: {sum_angle}, sharp_turns: {sharp_turns}, loose_turns: {loose_turns}")
            hill_groups.add(i)
        # print()

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
        buffer = 10
        x_min = max(0, x_min_raw - buffer)
        x_max = min(GRID_SIZE - 1, x_max_raw + buffer)
        z_min = max(0, z_min_raw - buffer)
        z_max = min(GRID_SIZE - 1, z_max_raw + buffer)

        # map에 적용. 따로 일반 함수로 빼놔도 좋을 듯...
        for x in range(x_min, x_max + 1):
            for z in range(z_min, z_max + 1):
                if maze[z][x] == 0:  # 이미 마킹된 경우는 생략
                    maze[z][x] = 1


# 초기할 인덱스 위치 계산(start_row, start_col, end_row, end_col)
def clamp_range(center, delta = 25, grid_size = 300):  # delta가 buffer 같은 것 
    start = max(center - delta, 0)
    end = min(center + delta, grid_size - 1)
    return start, end


# 맵, 지나온 길만 초기화하고 현 위치는 초기화 X
def initialize_maze(current_pos, maze):

    maintain_start_x, maintain_end_x = clamp_range(current_pos[0]) #, MAINTAIN_NUM, GRID_SIZE)
    maintain_start_z, maintain_end_z = clamp_range(current_pos[1]) #, MAINTAIN_NUM, GRID_SIZE)
    # 함수 검증용 print문
    # print("current_pos: ",current_pos)
    # print("maintain_area_z: ", maintain_start_z, "~", maintain_end_z)
    # print("maintain_area_x: ", maintain_start_x, "~", maintain_end_x)
    
    old_maze = []
    for x in range(maintain_start_x, maintain_end_x + 1):
        row = []
        for z in range(maintain_start_z, maintain_end_z + 1):
            row.append(maze[x][z])
        old_maze.append(row)

    maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 0으로 전부 초기화...
    
    for r_idx, r in enumerate(range(maintain_start_x, maintain_end_x + 1)): # old_maze에 저장된 부분 넣기
        for c_idx, c in enumerate(range(maintain_start_z, maintain_end_z + 1)): 
                maze[r][c] = old_maze[r_idx][c_idx]
            
    original_obstacles = []  # 초기화
    return maze, original_obstacles  # 지나온 길에 대한 장애물 값은 지워진 맵, original_obstacles 도 초기화해서 return 

info_func_implement = 0
how_many_init = 0
@app.route('/info', methods=['POST'])
def info():
    global maze, original_obstacles
    global info_func_implement, how_many_init

    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

        # maze
    info_func_implement+=1
    if info_func_implement == 3:
        # data = request.get_json(force=True)  # 현위치 데이터로 받아오기
        pos = data.get('playerPos', {})
        pos_x = int(pos.get('x', 0))
        pos_z = int(pos.get('z', 0))
        current_pos = (pos_x, pos_z)

        if 'x' not in pos or 'z' not in pos:
            print("현재 위치 좌표를 못 받아옴.")
        else: 
            maze, original_obstacles = initialize_maze(current_pos, maze)
            # print("maze 초기화")
            # np.save(f'./maze_backup/maze_backup{how_many_init}.npy', np.array(maze))
        how_many_init+=1
        info_func_implement = 0


    # 여기서부터 수정 코드
    # 설정... 
    # channel 45, MinimapChannel -, Y position 1.61, lidar position: Turret, sdl -, distance 110
    lidar_data = [
        (pt["position"]["x"], pt["position"]["z"], pt["verticalAngle"])
        for pt in data.get("lidarPoints", [])
        if (
             2 < pt.get("verticalAngle", 0) < 7 and
           # pt.get("verticalAngle") != 2.045455 and
            pt.get("isDetected", False) == True
        )]
    
    if not lidar_data:
        print("라이다 감지되는 것 없음")
        return jsonify({"status": "no lidar points"})

    # 라이다 데이터 -> df로 변환...
    lidar_df = pd.DataFrame(lidar_data, columns=['x', 'z', 'verticalAngle']) 
    split_lidar_df = split_by_distance(lidar_df)  # line_group 이라는 칼럼이 추가된 형태가 됨

    hill_groups = detect_obstacle_and_hill(split_lidar_df)  # 언덕으로 분류된 line_group 값을 알아옴
    if hill_groups:  # 언덕으로 분류된게 있으면
        only_obstacle_df = split_lidar_df[~split_lidar_df['line_group'].isin(hill_groups)]  # 언덕으로 분류된 것 죄다 버리기...
    else:
        only_obstacle_df = split_lidar_df

    if len(only_obstacle_df) == 0:
        # print("감지되는 장애물 없음")
        # continue  #  ..?
        return jsonify({"status": "no obstacles detected"})  # 끝내기.
    else:
        map_obstacle(only_obstacle_df)
    
    #여기서부터 수정 끝##############

    try:
        json_path = os.path.join(os.path.dirname(__file__), "original_obstacles.json")
        with open(json_path, "w") as f:
            json.dump(original_obstacles, f, indent=2)
        # print("✅ original_obstacles.json 저장 완료")

        np.save("maze.npy", np.array(maze))
        np.savetxt("maze.csv", np.array(maze), fmt="%d", delimiter=",")
    except Exception as e:
        print(f"❌ 장애물 저장 실패: {e}")

    return jsonify({"status": "success", "obstacle_clusters": ""})
    

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
    global astar_how_many_implement

    astar_how_many_implement+=1
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

# 현재 A* 결과. 어차피 get_action 함수에서 실행하니 좌표 두번 이동하고 a_star 실행하니까, 의미 없다 판단해서 주석처리
# path = a_star(start, destination)  

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
                # print(f"⚠️ 전방 장애물 감지: ({nx},{nz})")
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
    global combined_command_cache
    
    current_yaw = INITIAL_YAW
    previous_position = None
    target_reached = False

    config = {
        "startMode": "start",
        "blStartX": start_x, "blStartY": 10, "blStartZ": start_z,
        "rdStartX": 160, "rdStartY": 0, "rdStartZ": 260,
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

tank_detected = False
tank_detect_time = None

combined_command_cache = []

@app.route('/get_action', methods=['POST'])
def get_action():
    global target_reached, previous_position, current_yaw, current_position, last_position
    global start_time, end_time
    global tank_detected, tank_detect_time
    global combined_command_cache
    
    data = request.get_json(force=True)
    pos = data.get('position', {})
    pos_x = float(pos.get('x', 0))
    pos_z = float(pos.get('z', 0))

        #주행하고 있는 중에 tank가 감지되면 멈추고 출발하기 
    if tank_detected:
        if tank_detect_time and time.time() - tank_detect_time < 3.0:  # 3초 정지
            print("🛑 탱크 감지됨 → 일시 정지 중")
            stop_cmd = {k: {'command': 'STOP', 'weight': 1.0} for k in ['moveWS', 'moveAD']}
            return jsonify(stop_cmd)
        else:
            tank_detected = False
            tank_detect_time = None
            print("✅ 탱크 정지 해제, 이동 재개")
        

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
         # maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        return jsonify(stop_cmd)

    if previous_position is not None:
        dx = pos_x - previous_position[0]
        dz = pos_z - previous_position[1]
        if math.hypot(dx, dz) > 0.01:
            current_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
    previous_position = (pos_x, pos_z)

    current_grid = (int(pos_x), int(pos_z))

    #######################################################################
    # 2 좌표 이동한 후. astar(현좌표, 최종목적지) 함수 실행해서 path 새로 뽑기 반복

    if combined_command_cache:  # 명령어가 남아있다면
    # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
        cmd = combined_command_cache.pop(0)
        # print(f"👊두번째 명령어 실행_cmd : {cmd}")
        print(f"🚀 cmd 1개 {cmd}")
        return jsonify(cmd)
    elif not combined_command_cache: #or combined_command_cache is None:  # 비어있다면 = 명령어 두개 다 실행했다면, 이동 
        # print("combined_command_cache 비어있어서 a_star 실행해요...")
        path = a_star(current_grid, destination)  
        # a_star 경로 CSV로 저장
        if path:
            df = pd.DataFrame(path, columns=["x", "z"])
            os.makedirs("logs", exist_ok=True)  # logs 디렉토리 없으면 생성
            df.to_csv("logs/a_star_path_log.csv", index=False)
            # print("✅ A* 경로가 logs/a_star_path_log.csv 에 저장되었습니다.")
        else:
            print("❌ A* 경로가 비어 있어 저장하지 않았습니다.")
    
    if len(path) > 3:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
        next_grid = path[1:4]  # 3개 좌표 참조
        # print(f"👊👊next_grid가 두개예요: {next_grid}👊👊")
    elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
        next_grid = [path[1]]      # 한개씩 참조  
    else: 
        next_grid = [current_grid]   # 0개면 멈춰라! 도착한거니까!

    for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1
        # print(f"i:{i},  (len(next_grid)) : {len(next_grid)}")
        # next_grid[1]의 회전 각도는 current 가 아니라 next_grid[0]에서 게산해야 맞음 
        base_pos = current_grid if i == 0 else next_grid[i - 1]  

        # print(f"next_grid: {next_grid[i]}")
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

    # print(f"반복 나옴 {i}")
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
    print(f"📍 현재 pos=({pos_x:.1f},{pos_z:.1f})")
    # yaw={current_yaw:.1f} 두번째 좌표로 가는 앵글 ={target_angle:.1f} 차이 ={diff:.1f}")
    # print(f"🚀 cmd 3개 {combined_command_cache}")
    cmd = combined_command_cache.pop(0)
    print(f"cmd 1개_ : {cmd}")
    return jsonify(cmd)

@app.route('/detect', methods=['POST'])
def detect():
    global tank_detected, tank_detect_time

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    target_classes = {0: "car1", 1: "car2", 2: "human", 3: "tank"}
    filtered_results = []

    detected_classes = set()
    tank_close = False

    # 기준 크기 (예: 높이 또는 너비가 200픽셀 이상이면 가까움)
    BBOX_SIZE_THRESHOLD = 200

    for box in detections:
        class_id = int(box[5])
        if class_id in target_classes:
            class_name = target_classes[class_id]
            x1, y1, x2, y2 = box[:4]
            width = x2 - x1
            height = y2 - y1

            detected_classes.add(class_name)
            filtered_results.append({
                'className': class_name,
                'bbox': [float(coord) for coord in box[:4]],
                'confidence': float(box[4]),
                'color': '#00FF00',
                'filled': False,
                'updateBoxWhileMoving': False
            })

            # 탱크일 경우, 크기 판단
            if class_name == "tank":
                print(f"📦 탱크 bbox 크기: width={width:.1f}, height={height:.1f}")
                if height >= BBOX_SIZE_THRESHOLD or width >= BBOX_SIZE_THRESHOLD:
                    tank_close = True

    if tank_close:
        tank_detected = True
        tank_detect_time = time.time()
        print("🛑 탱크 bbox 큼 → 일시정지 준비")
    else:
        tank_detected = False

    return jsonify(filtered_results)

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


# 서버 실행
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=4000)
    except KeyboardInterrupt:
        print("\n🛑 서버 종료 감지됨 (Ctrl+C)")
    finally:
        print(f"📊 총 충돌 횟수: {collision_count}회")
        print(f"astar_how_many_implement: {astar_how_many_implement}")