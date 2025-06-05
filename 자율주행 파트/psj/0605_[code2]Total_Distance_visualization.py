# 0605 [코드2] 총 이동거리 시각화모드
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====== 1. 기본 데이터 로딩 ======
maze = np.load("maze.npy")
a_star_path = pd.read_csv("a_star_path.csv") # a_star_path.csv는 경로 탐색 알고리즘 A*가 계산한 경로 결과(x, z 좌표가 담긴 csv파일)
actual_path = pd.read_csv("tank_path0.csv")  # tank_path0.csv는 시뮬레이터에서 탱크가 실제로 따라간 경로(x, z 좌표가 담긴 csv파일)

# ====== 2. 시작점과 목적지 (Flask 코드에서 가져온 값) ======
start = (250, 250)
destination = (40, 150)

# ====== 3. 좌표 추출 ======
a_star_x = a_star_path["x"].tolist()
a_star_z = a_star_path["z"].tolist()

actual_x = actual_path["x"].tolist()
actual_z = actual_path["z"].tolist()

# ====== 4. 시각화 시작 ======
plt.figure(figsize=(10, 10))

# ========================================================================================

# ====== 4.5 실제 이동 거리 계산 ======
def calculate_total_distance(x_coords, z_coords):
    total = 0.0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i - 1]
        dz = z_coords[i] - z_coords[i - 1]
        total += np.sqrt(dx**2 + dz**2)
    return total

real_path_distance = calculate_total_distance(actual_x, actual_z)
print(f"📏 실제 이동 거리: {real_path_distance:.2f} 단위")

# ====== 5. 꾸미기 전 ======
plt.title(f"A* Path vs Real Path\n📏 실제 이동 거리: {real_path_distance:.2f} 단위")
# ========================================================================================

# 🔲 장애물 시각화 (maze에서 1인 좌표만 그림)
obstacle_coords = np.argwhere(maze == 1)
if len(obstacle_coords) > 0:
    obs_z, obs_x = zip(*obstacle_coords)
    plt.scatter(obs_x, obs_z, color='black', s=5, label="Obstacle")

# 🔵 A* 경로
plt.plot(a_star_x, a_star_z, color='blue', linewidth=2, label="A* Path")

# 🔸 실제 이동 경로
plt.plot(actual_x, actual_z, color='orange', linestyle='--', linewidth=2, label="Real Path")

# 🟢 시작점
plt.scatter([start[0]], [start[1]], c='green', s=100, marker='s', label="Start Point")

# 🔴 목적지
plt.scatter([destination[0]], [destination[1]], c='red', s=100, marker='X', label="Destination")

# ====== 5. 꾸미기 ======
plt.title("A* Path, Real Path, Destination, Total Distance")
# ========================================================================================

plt.plot([], [], ' ', label=f"Total Distance: {real_path_distance:.2f} ")
plt.legend(loc='upper right')
# ========================================================================================

plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.xlabel("X") # x축으로 대응
plt.ylabel("Z") # y축으로 대응
plt.imshow(maze, cmap=plt.cm.gray_r, origin='lower', alpha=0.2)  # 배경으로 maze 회색 표시 / origin='lower'는 좌표 (0,0)이 왼쪽 아래에 위치합니다.
plt.show()

# 🔵 A* 경로 → blue line
# 🔸 실제 경로 → 주황 점선
# ⚫ 장애물 → black dots
# 🟢 시작점 → green square
# 🔴 목적지 → red X