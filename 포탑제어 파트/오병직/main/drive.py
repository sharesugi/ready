import math
import numpy as np

GRID_SIZE = 300 

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
total_distance = 0
def calculate_actual_path(position_history):
    global total_distance
    
    if len(position_history) > 1:
        for i in range(len(position_history) -1):
            x1, z1 = position_history[i] # 이전 좌표
            x2, z2 = position_history[i+1] # 현재 좌표
            step_distance = math.sqrt((x2 - x1)**2 + (z2 - z1)**2) # 가장 최근 두 지점의 좌표 추출
            total_distance += step_distance                        # 지금 이동한 거리(step_distance)를 누적 거리(total_distance)에 더함
    return total_distance

# DBSCAN 대체 방안 함수... 인접한 좌표들의 거리 차이를 통해서 라벨링을 함.
# 단점?_ 값이 자주 튀는 언덕이나 곡선이면 연결된 선의 형태라도 나뉘어질 수 있다... 일단 동작에는 문제 없음
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
           # print(f"ㄱ 장애물_loose_turns : {loose_turns}, sum_angle: {sum_angle}, sharp_turns: {sharp_turns}")
            continue
            
         # 급하게 꺾이는 구간이 3개 이상이고(전차는 꺾임 구간이 2개라서 혹시 몰라서 임시방편으로...) 
        # and 각도가 느슨하게 꺾이는 것이 3번 이상 발생하면 언덕...
        elif sharp_turns > 1 and loose_turns >=3:  
            # print("급변하는 언덕")
            hill_groups.add(i)
            
        elif sharp_turns and loose_turns:  # 급하게 꺾이는 구간은 없지만 느슨하게 서서히 꺾일 때
            # print("느슨한 언덕")
            hill_groups.add(i)
        else:  
            # 이 부분 추후 수정 필요...
            # print(f"분류안함(언덕)_sum_angle: {sum_angle}, sharp_turns: {sharp_turns}, loose_turns: {loose_turns}")
            hill_groups.add(i)
        # print()

    # print(f"hill_groups: {hill_groups}")
    return hill_groups