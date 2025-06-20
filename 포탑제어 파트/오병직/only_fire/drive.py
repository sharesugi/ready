import math
import numpy as np

GRID_SIZE = 300 

# 현재 위치와 다음 위치 간 각도 계산 함수
def calculate_angle(current, next_pos): # A*알고리즘을 통해서 어디로 갈지 전체 경로를 정했기 때문에 다음 위치로만 가면 됨.
    dx = next_pos[0] - current[0]
    dz = next_pos[1] - current[1]
    return (math.degrees(math.atan2(dz, dx)) + 360) % 360

# 맵 유효 위치 확인
def is_valid_pos(pos, size=GRID_SIZE): # 내 전차 혹은 A* 경로가 300x300 안에 있는지 확인
    x, z = pos
    return 0 <= x < size and 0 <= z < size