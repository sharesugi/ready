import math
import numpy as np
import requests

class TankEnv:
    def __init__(self, server_url="http://localhost:5000"):
        self.server_url = server_url
        self.reset()

    def reset(self):
        # 무작위 위치 초기화
        self.tank_x = np.random.uniform(-20, 20)
        self.tank_z = np.random.uniform(-20, 20)
        self.enemy_x = np.random.uniform(-20, 20)
        self.enemy_z = np.random.uniform(-20, 20)

        # 서버에 초기화 요청
        requests.post(f"{self.server_url}/init", json={
            "tank_x": self.tank_x,
            "tank_z": self.tank_z,
            "enemy_x": self.enemy_x,
            "enemy_z": self.enemy_z
        })

        self.steps = 0
        return self.get_state()

    def get_state(self):
        info = requests.get(f"{self.server_url}/info").json()

        print(info)

        # time = info['Time']
        tank_x = info["tank_x"]
        tank_z = info["tank_z"]
        enemy_x = info["enemy_x"]
        enemy_z = info["enemy_z"]
        turret_yaw = info["turret_yaw"]
        turret_pitch = info["turret_pitch"]

        # 상태 벡터 구성
        return np.array([
            tank_x / 300, tank_z / 300,
            enemy_x / 300, enemy_z / 300,
            turret_yaw / 360, turret_pitch / 90,
        ], dtype=np.float32)

    def step(self, action):
        self.latest_action = action  # 여기서 최신 행동 저장

        # 정보 받아오기
        info = requests.get(f"{self.server_url}/info").json()
        cooldown = info["cooldown_norm"]

        reward = 0
        done = False

        if action[2] > 0.5 and cooldown >= 0.99:
            hit = self.check_hit(info)
            if hit:
                reward = 1.0
                done = True
            else:
                reward = -0.5

        self.steps += 1
        if self.steps >= 200:
            done = True

        return self.get_state(), reward, done

    def check_hit(self, info):
        tank_x = info["tank_x"]
        tank_y = 10.0
        tank_z = info["tank_z"]
        turret_yaw = info["turret_yaw"]
        turret_pitch = info["turret_pitch"]

        enemy_x = info["enemy_x"]
        enemy_z = info["enemy_z"]

        # 탄도 계산
        initial_speed = 60
        gravity = 9.81

        yaw_rad = math.radians(turret_yaw)
        pitch_rad = math.radians(turret_pitch)

        dx = math.cos(pitch_rad) * math.sin(yaw_rad)
        dy = math.sin(pitch_rad)
        dz = math.cos(pitch_rad) * math.cos(yaw_rad)

        v0x = initial_speed * dx
        v0y = initial_speed * dy
        v0z = initial_speed * dz

        if v0y <= 0:
            return False

        t_impact = 2 * v0y / gravity
        impact_x = tank_x + v0x * t_impact
        impact_z = tank_z + v0z * t_impact

        dist_to_enemy = math.sqrt((impact_x - enemy_x) ** 2 + (impact_z - enemy_z) ** 2)

        return dist_to_enemy < 5.0
