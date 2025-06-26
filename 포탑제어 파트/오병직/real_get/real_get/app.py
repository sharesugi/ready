import os, json, time, math
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
from queue import PriorityQueue
from flask import Flask, render_template, request, jsonify
import fire, drive
import matplotlib # 이것 추가함
matplotlib.use('Agg')  # GUI 없는 서버에서도 작동하게 함 # 이것 추가함
import matplotlib.pyplot as plt # 이것 추가함

app = Flask(__name__)
model_yolo = YOLO('./models/best_8s.pt')

# 화면 해상도 (스크린샷 찍었을 때 이미지 크기)
IMAGE_WIDTH = 1921
IMAGE_HEIGHT = 1080

# 카메라 각도
FOV_HORIZONTAL = 47.81061 
FOV_VERTICAL = 28       

# 적 전차를 찾는 상태
DRIVE_MODE = True
END = False

# 전역 설정값 및 변수 초기화
GRID_SIZE = 300  # 맵 크기
maze = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]  # 장애물 맵

# 내 전차 시작 위치
start_x = 20
start_z = 20
start = (start_x, start_z)

# 최종 목적지 위치 - 적 전차도 이 위치에 갖다 놓음.
dest_list = []
dest_idx = 0

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

enemy_pos_locked = False
locked_enemy_pos = None
lock_frame_counter = 0
LOCK_TIMEOUT = 100  # 예: 100프레임 = 인터벌 * 100초

# 충돌 없을 때 파일 저장
with open('collision_points.json', 'w') as f:
    json.dump({
        "collision_count": 0,
        "collision_points": []
    }, f, indent=2)

# 시간 세는 부분
start_time = None
end_time = None

GRID_SIZE = 300  # 맵 크기
astar_how_many_implement = 0

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

lidar_data = [] # /info 에서 가져오는 라이다 데이터 저장

@app.route('/detect', methods=['POST'])
def detect():
    global lidar_data, enemy_pos, DRIVE_MODE, yolo_results
    global enemy_pos_locked, locked_enemy_pos, lock_frame_counter

    image = request.files.get('image')
    if not image:
        return jsonify({"error": "No image received"}), 400

    image_path = 'temp_image.jpg'
    image.save(image_path)

    results = model_yolo(image_path)
    detections = results[0].boxes.data.cpu().numpy()

    is_tank = False
    target_classes = {2: "human", 3: "tank"}
    filtered_results = []
    current_bboxes = []

    for i, box in enumerate(detections):
        if box[4] >= 0.80:
            class_id = int(box[5])
            if class_id == 3:
                is_tank = True
                current_bboxes.append({
                    'id': i,
                    'x1': float(box[0]), 'y1': float(box[1]),
                    'x2': float(box[2]), 'y2': float(box[3])
                })

            if class_id in target_classes:
                filtered_results.append({
                    'className': target_classes[class_id],
                    'bbox': [float(coord) for coord in box[:4]],
                    'confidence': float(box[4]),
                    'color': '#00FF00',
                    'filled': False,
                    'updateBoxWhileMoving': True
                })

    # YOLO + LiDAR 매칭
    yolo_results = fire.match_yolo_to_lidar(
        bboxes=current_bboxes,
        lidar_points=lidar_data,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        fov_h=FOV_HORIZONTAL,
        fov_v=FOV_VERTICAL
    )

    print(f'🗺️ yolo_results : {yolo_results}')

    for i, r in enumerate(yolo_results):
        print(f"탐지된 전차 {i+1}:")
        print(f"  바운딩 박스: {r['bbox']}")
        print(f"  LiDAR 좌표: {r['matched_lidar_pos']}")
        print(f"  거리: {r['distance']:.2f}m\n")

    ## 🔒 타겟 락온 및 추적 로직
    lock_frame_counter += 1

    if not enemy_pos_locked:
        # 락 안 걸려 있을 때 → 가장 가까운 전차 락온
        min_distance = 100
        closest_result = None
        for r in yolo_results:
            if r['distance'] < min_distance:
                closest_result = r
                min_distance = r['distance']

        if closest_result:
            matched = closest_result['matched_lidar_pos']
            enemy_pos = {
                'x': matched.get('x', 0),
                'y': matched.get('y', 0),
                'z': matched.get('z', 0)
            }
            locked_enemy_pos = matched
            enemy_pos_locked = True
            lock_frame_counter = 0
            print("🔒 타겟 락온 완료!")

    else:
        # 이미 락이 걸려 있는 상태 → 좌표 기반으로 가장 비슷한 전차 추적
        updated = False
        for r in yolo_results:
            matched = r['matched_lidar_pos']
            if fire.distance_3d(matched, locked_enemy_pos) < 2.0:  # 2m 이내면 동일 타겟으로 간주
                enemy_pos = {
                    'x': matched.get('x', 0),
                    'y': matched.get('y', 0),
                    'z': matched.get('z', 0)
                }
                locked_enemy_pos = matched
                updated = True
                break

        if not updated:
            print("⚠️ 락된 전차 위치 유실 → 락 해제")
            enemy_pos_locked = False
            locked_enemy_pos = None
            lock_frame_counter = 0

        if lock_frame_counter > LOCK_TIMEOUT:
            print("⏲️ 락 유지 시간 초과 → 해제")
            enemy_pos_locked = False
            locked_enemy_pos = None
            lock_frame_counter = 0

    if is_tank and yolo_results and enemy_pos_locked:
        DRIVE_MODE = False

    return jsonify(filtered_results)

# 아래 세 변수 모두 사격 불가능한 각도 판별할 때 사용하는 변수
angle_hist = []
save_time = 0
len_angle_hist = -1

# 여기 리스트에 cmd 2개를 넣는다
combined_command_cache = []
next_group = 1
three_moved = False  # 0626 추가 (lidar data를 astar 실행했을 때만 받아오게)
new_df = pd.DataFrame()

aim_ready_since = None
aim_lost_time = None
last_logged_second = None
fired_logged = False

@app.route('/get_action', methods=['POST'])
def get_action():
    global enemy_pos, last_bullet_info, angle_hist, save_time, len_angle_hist, DRIVE_MODE, yolo_results
    global target_reached, previous_position, current_yaw, dest_list, dest_idx
    global body_x, END
    global three_moved # 0626 추가
    
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

    if END:
        stop_cmd = {k: {'command': 'STOP', 'weight': 1.0} for k in ['moveWS', 'moveAD']}
        stop_cmd['fire'] = False
        return jsonify(stop_cmd) 

    if DRIVE_MODE: # 적 전차를 탐색하는 상태일 때 
        if dest_idx < len(dest_list):
            destination = dest_list[dest_idx]
        else:
            destination = (-1, -1)
        print(f'🗺️ Destination coord : {destination}')

        if not target_reached and math.hypot(pos_x - destination[0], pos_z - destination[1]) < 5.0: # 거리 5 미만이면 도착으로 간주
            target_reached = True 
            
        if target_reached:
            if dest_idx < len(dest_list):
                target_reached = False
                dest_idx += 1
            
            if dest_idx >= len(dest_list):
                DRIVE_MODE = False
                END = True

        if previous_position is not None:
            dx = pos_x - previous_position[0]
            dz = pos_z - previous_position[1]
            if math.hypot(dx, dz) > 0.01:
                current_yaw = (math.degrees(math.atan2(dz, dx)) + 360) % 360
        previous_position = (pos_x, pos_z)

        current_grid = (int(pos_x), int(pos_z))

        if combined_command_cache:
        # 캐시에 남은 명령이 있으면 그걸 먼저 보내고 pop
            cmd = combined_command_cache.pop(0)
            #print(f"🚀 cmd 1개 {cmd}")
            return jsonify(cmd)
        elif not combined_command_cache:  # 명령어 두 개 다 실행해서 비어있으면
            path = a_star(current_grid, destination)  # 이 때만 astar 실행
            three_moved = True   # 0626
            # print("three_moved = true, get action에서") # 0626  디버깅용_(잘 되는거 확인하면 나중에 지워도 상관 무)

        if len(path) > 3:   # 최종목적지까지 3개 이상의 좌표가 남았으면 
            next_grid = path[1:4]  # 1~2번째 좌표 참조
        elif len(path) > 1:          # 최종목적지까지 2개 이하의 좌표가 남았으면 
            next_grid = [path[1]]      # 한개씩 참조  
        else: 
            next_grid = [current_grid]   # 0개면 멈춰라! 도착한거니까!

        for i in range(len(next_grid)):  # 두개의 좌표가 맵을 빠져나기지 않는지 확인 # 0, 1
            base_pos = current_grid if i == 0 else next_grid[i - 1]  
        
            if not drive.is_valid_pos(next_grid[i]):  # 가야하는 곳이 맵 외에 있으면 움직이는거 멈춤
                stop_cmd = {k: {'command': '', 'weight': 0.0} for k in ['moveWS', 'moveAD']}
                stop_cmd['fire'] = False
                return jsonify(stop_cmd)

            target_angle = drive.calculate_angle(base_pos, next_grid[i])  # 현재 좌표에서 두번째 좌표로
            diff = (target_angle - current_yaw + 360) % 360   # 현 각도랑 틀어야할 각도 차이 알아내고
            if diff > 180:  # 이거는 정규화 비슷
                diff -= 360

            # 이건 그냥 유클리드 거리. sqrt는 제곱근! 현위치랑 목적좌표까지의 거리
            # 목적지 근처에 가면 천천히 뒷 키 잡으려고, 목적지까지 남은 거리를 계산.
            distance = math.sqrt((pos_x - destination[0])**2 + (pos_z - destination[1])**2)
    
            w_weight = 0.3
            acceleration = 'W'

            abs_diff = abs(diff)
            if 0 < abs_diff < 30 :  
                w_degree = 0.3
            elif 30 <= abs_diff < 60 :    
                w_degree = 0.6
            elif 60 <= abs_diff < 90 : 
                w_degree = 0.75
            else :
                w_degree = 1.0

            # 현재 터렛 각도와 목표 각도 차이 계산
            yaw_diff = body_x - turret_x

            # 각도 차이 보정 (-180 ~ 180)
            if yaw_diff > 180:
                yaw_diff -= 360
            elif yaw_diff < -180:
                yaw_diff += 360

            # 최소 가중치 0.01 설정, 최대 1.0 제한
            def calc_yaw_weight(diff):
                w = min(max(abs(diff) / 30, 0.01), 0.3)  # 30도 내외로 가중치 조절 예시
                return w
            
            # 위 두 함수에서 최소 가중치를 낮게 할수록 조준 속도는 낮아지지만 정밀 조준 가능
            yaw_weight = calc_yaw_weight(yaw_diff)

            # 좌우 회전 명령 결정
            if yaw_diff > 0.1:  # 목표가 오른쪽
                turretQE_cmd = "E"
            elif yaw_diff < -0.1:  # 목표가 왼쪽
                turretQE_cmd = "Q"
            else:
                turretQE_cmd = ""

            forward = {'command': acceleration, 'weight': w_weight}
            turn = {'command': 'A' if diff > 0 else 'D', 'weight': w_degree}
            turret = {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0}

            cmd = {
                'moveWS': forward,
                'moveAD': turn,
                "turretQE" : turret
            }

            combined_command_cache.append(cmd)   # 두 좌표에 대한 명령값 2개가 여기 리스트에 저장됨

        #print문 살짝 수정-희연
        #print(f"📍 현재 pos=({pos_x:.1f},{pos_z:.1f})") # yaw={current_yaw:.1f} 두번째 좌표로 가는 앵글 ={target_angle:.1f} 차이 ={diff:.1f}")
        #print(f"🚀 cmd 3개 {combined_command_cache}")
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
        target_yaw = fire.get_yaw_angle(player_pos, enemy_pos)

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
        target_pitch_dnn = fire.find_angle_for_distance_dy_dnn(distance, dy)
        target_pitch_xgb = fire.find_angle_for_distance_dy_xgb(distance, dy)
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
            w = min(max(abs(diff) / 10, 0.1), 3.0)  # 30도 내외로 가중치 조절 예시
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

        import time
        global aim_ready_since, aim_lost_time, last_logged_second, fired_logged
        
        now = time.time()

        if aim_ready:
            if aim_ready_since is None:
                aim_ready_since = now
                # print("🎯 목표물 포착됨! 조준 시작")
                fired_logged = False
                last_logged_second = None
            aim_lost_time = None
        else:
            if aim_ready_since is not None:
                if aim_lost_time is None:
                    aim_lost_time = now
                elif now - aim_lost_time > 1.0:
                    aim_ready_since = None
                    aim_lost_time = None
                    last_logged_second = None
                    fired_logged = False

        fire_cmd = False
        if aim_ready_since is not None:
            elapsed = now - aim_ready_since
            remaining = 3 - int(elapsed)
            if elapsed >= 3.0:
                fire_cmd = True
                if not fired_logged:
                    print("🔥 조준 완료 → 3초 경과 → 발사!")
                    fired_logged = True
            else:
                if last_logged_second != remaining:
                    print(f"🕔 조준 완료! 발사까지 {remaining}초...")
                    if remaining == 3:
                        print(f'🏹target_yaw : {target_yaw}, 🏹target_pitch : {target_pitch}')
                    last_logged_second = remaining

        else:
            aim_ready_since = None
            text_shown = False
            fire_cmd = False
        #print(f'🏹target_yaw : {target_yaw}, 🏹target_pitch : {target_pitch}')

        # 이동은 일단 멈춤, 위에서 계산한 각도 오차에 따른 가중치로 조준
        command = {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": turretQE_cmd, "weight": yaw_weight if turretQE_cmd else 0.0},
            "turretRF": {"command": turretRF_cmd, "weight": pitch_weight if turretRF_cmd else 0.0},
            "fire": fire_cmd
        }
    return jsonify({
        "target_reached": target_reached, 
        "remaining": remaining if remaining and remaining > 0 else 0,
        "fire": fire_cmd,})
    
    

# 전역 상태 저장 (시뮬레이터 reset 시킬 때 사용)
last_bullet_info = {}

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global last_bullet_info, enemy_pos_locked, locked_enemy_pos, lock_frame_counter
    last_bullet_info = request.get_json()
    print("💥 탄 정보 갱신됨:", last_bullet_info)

    if last_bullet_info.get('hit', False):
        print("🎯 적중 확인 → 락 해제")
        enemy_pos_locked = False
        locked_enemy_pos = None
        lock_frame_counter = 0

    return jsonify({"yolo_results": "ok"})

enemy_pos = {} # 적 전차의 위치
true_hit_ratio = [] # 평가를 위해서 사용했던 변수
s_time = 0 # 시뮬레이터 시간
body_x = 0

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
    print("current_pos: ",current_pos)
    print("maintain_area_z: ", maintain_start_z, "~", maintain_end_z)
    print("maintain_area_x: ", maintain_start_x, "~", maintain_end_x)
    
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

def map_obstacle(original_obstacles, only_obstacle_df):   
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
    
info_func_implement = 0
how_many_init = 0
@app.route('/info', methods=['GET', 'POST'])
def get_info():
    global last_bullet_info, true_hit_ratio, s_time, lidar_data, DRIVE_MODE, enemy_pos
    global maze, original_obstacles, body_x
    global info_func_implement, how_many_init
    global three_moved   # 0626

    data = request.get_json()

    lidar_data = data.get('lidarPoints', [])
    s_time = data.get("s_time", 0)
    body_x = data.get('playerBodyX', 0)
    control = ""
    
    if three_moved:
        
        info_func_implement += 1
        if info_func_implement == 10:
            pos = data.get('playerPos', {})
            pos_x = int(pos.get('x', 0))
            pos_z = int(pos.get('z', 0))
            current_pos = (pos_x, pos_z)
    
            if 'x' not in pos or 'z' not in pos:
                print("현재 위치 좌표를 못 받아옴.")
            else: 
                maze, original_obstacles = initialize_maze(current_pos, maze)
                print("maze 초기화")
                np.save(f'./maze_backup/maze_backup{how_many_init}.npy', np.array(maze))
                how_many_init += 1
                info_func_implement = 0
    
        drive_lidar_data = [
            (pt["position"]["x"], pt["position"]["z"], pt["verticalAngle"])
            for pt in data.get("lidarPoints", [])
            if (
                2 < pt.get("verticalAngle", 0) < 7 and
                pt.get("isDetected", False) == True
            )]
        if not drive_lidar_data:
            print("라이다 감지되는 것 없음")
            return jsonify({"status": "no lidar points"})
    
        # 라이다 데이터 -> df로 변환...
        lidar_df = pd.DataFrame(drive_lidar_data, columns=['x', 'z', 'verticalAngle']) 
        split_lidar_df = drive.split_by_distance(lidar_df)  # line_group 이라는 칼럼이 추가된 형태가 됨
    
        hill_groups = drive.detect_obstacle_and_hill(split_lidar_df)  # 언덕으로 분류된 line_group 값을 알아옴
        if hill_groups:  # 언덕으로 분류된게 있으면
            only_obstacle_df = split_lidar_df[~split_lidar_df['line_group'].isin(hill_groups)]  # 언덕으로 분류된 것 죄다 버리기...
        else:
            only_obstacle_df = split_lidar_df
    
        if len(only_obstacle_df) == 0:
            print("감지되는 장애물 없음")
            # continue  #  ..?
            # return jsonify({"status": "no obstacles detected"})  # 끝내기.
        else:
            map_obstacle(original_obstacles, only_obstacle_df)
    
        try:
            json_path = os.path.join(os.path.dirname(__file__), "original_obstacles.json")
            with open(json_path, "w") as f:
                json.dump(original_obstacles, f, indent=2)
            print("✅ original_obstacles.json 저장 완료")
    
            np.save("maze.npy", np.array(maze))
            np.savetxt("maze.csv", np.array(maze), fmt="%d", delimiter=",")
        except Exception as e:
            print(f"❌ 장애물 저장 실패: {e}")

        three_moved = False # 0626
        # print("three_moved = false, info에서")  # 0626  디버깅용_(잘 되는거 확인하면 나중에 지워도 상관 무)
    
        # 발사된 탄이 어딘가에 떨어졌을 때
    if last_bullet_info:
            DRIVE_MODE = True
            control = ""
            last_bullet_info = {}
            enemy_pos = {}

    return jsonify({
        "status": "success",
        "message": "Data received",
        "control": control,
    })

# w
@app.route('/set_destinations', methods=['GET', 'POST'])
def set_destinations():
    global dest_list, dest_idx

    if request.method == 'POST':
        if request.is_json:
            data = request.get_json(force=True)
        else:
            data = request.form

        x1 = float(data.get('destination_x1', 0.0))
        z1 = float(data.get('destination_z1', 0.0))
        x2 = float(data.get('destination_x2', 0.0))
        z2 = float(data.get('destination_z2', 0.0))
        x3 = float(data.get('destination_x3', 0.0))
        z3 = float(data.get('destination_z3', 0.0))

        dest_list = [(x1, z1), (x2, z2), (x3, z3)]
        dest_idx = 0

        # JSON 요청이면 JSON 반환
        if request.is_json:
            return jsonify({"status": "success", "destinations": dest_list})
        
        # 폼 전송이면 HTML 반환
        return render_template('success.html', destinations=dest_list, status='success')

    # GET 요청은 HTML 반환
    return render_template('success.html', destinations=dest_list, status='ready')
# w
@app.route('/get_destinations')
def get_destinations():
    global dest_list, dest_idx
    # dest_list가 [(10,10), (20,20), (30,30)] 이런 튜플 리스트라면 JSON에 맞게 딕셔너리 리스트로 변환 필요
    curr_dest = {'x': dest_list[dest_idx][0] if dest_idx < len(dest_list) else -1,
                 'z': dest_list[dest_idx][1] if dest_idx < len(dest_list) else -1,
                 'idx' : dest_idx,
                 'over' : int(len(dest_list) <= dest_idx)}
    return jsonify({'destinations': curr_dest})

IMAGE_FOLDER = r'/home/eva/jupyter_home/Tank Challenge/Eend/real_get/capture_images'


@app.route('/paths/image', methods=['GET'])
def get_path_image():
    try:
        df = pd.read_csv("a_star_path.csv")

        fig, ax = plt.subplots(figsize=(6, 5), dpi=100)  # 600x500 크기 맞춤
        x_vals = df["x"].values
        z_vals = df["z"].values

        ax.plot(x_vals, z_vals, color='blue', linewidth=2, label="Current A* Path")
        ax.scatter([x_vals[0]], [z_vals[0]], c='green', s=100, marker='s', label="Start")
        ax.scatter([x_vals[-1]], [z_vals[-1]], c='red', s=100, marker='*', label="Destination")

        ax.set_xlim(0, 299)
        ax.set_ylim(0, 299)
        ax.set_title("Latest A* Path")
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal')
        plt.tight_layout()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        plt.close(fig)

        return send_file(img_buf, mimetype='image/png')
    except Exception as e:
        return f"❌ 시각화 에러: {e}", 500


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
    global start_distance, DRIVE_MODE, last_bullet_info, enemy_pos
    global current_yaw, previous_position, target_reached
    current_yaw = INITIAL_YAW
    previous_position = None
    target_reached = False

    DRIVE_MODE = True
    last_bullet_info = {}
    enemy_pos = {}

    print("🛠️ /init 라우트 진입 확인!")

    config = {
        "startMode": "start",
        "blStartX": start_x, 
        "blStartY": 10, 
        "blStartZ": start_z,
        "rdStartX": 160, 
        "rdStartY": 0, 
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
# w
@app.route('/start', methods=['GET', 'POST'])
def start():
    return render_template('destination_input.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=False, use_reloader=False)
