from flask import Flask, request, jsonify
import math, random
import numpy as np
from tank_env import TankEnv

env = TankEnv()
app = Flask(__name__)

# 초기 상태 저장용 변수

@app.route('/init', methods=['GET'])
def init():
    min_x, max_x = 0, 300
    fixed_y = 10
    min_z, max_z = 0, 300

    while True:
        bl_x = random.uniform(min_x, max_x)
        bl_y = fixed_y
        bl_z = random.uniform(min_z, max_z)

        rd_x = random.uniform(min_x, max_x)
        rd_y = fixed_y
        rd_z = random.uniform(min_z, max_z)

        dist = math.sqrt((bl_x - rd_x) ** 2 + (bl_y - rd_y) ** 2 + (bl_z - rd_z) ** 2)
        if dist >= 10:
            break

    config = {
        "startMode": "start",
        "blStartX": bl_x,
        "blStartY": bl_y,
        "blStartZ": bl_z,
        "rdStartX": rd_x,
        "rdStartY": rd_y,
        "rdStartZ": rd_z,
        "trackingMode": False,
        "detactMode": False,
        "logMode": True,
        "enemyTracking": True,
        "saveSnapshot": False,
        "saveLog": True,
        "saveLidarData": False,
        "lux": 30000
    }

    print("🛠️ Initialization config sent via /init:", config)
    return jsonify(config)


@app.route("/info", methods=["GET"])
def get_info():
    data = request.get_json()
    return jsonify({
        # "Time" : data['Time'],
        "tank_x": data["Player_Pos_X"],
        "tank_z": data["Player_Pos_Z"],
        "enemy_x": data["Enemy_Pos_X"],
        "enemy_z": data["Enemy_Pos_Z"],
        "turret_yaw": data["Player_Turret_X"],
        "turret_pitch": data["Player_Turret_Y"]
    })


@app.route("/get_action")
def get_action():
    # 가장 최근의 DQN 행동 불러오기
    turret_dx, turret_dy, fire = env.latest_action

    # 움직임은 항상 전진 (임시), 포탑 조작은 행동값 기반
    moveWS = {"command": "W", "weight": 1.0}
    moveAD = {"command": "", "weight": 0.0}

    turret_qe = "Q" if turret_dx < 0 else "E" if turret_dx > 0 else ""
    turret_rf = "R" if turret_dy > 0 else "F" if turret_dy < 0 else ""

    fire_cmd = fire > 0.5

    command = {
        "moveWS": moveWS,
        "moveAD": moveAD,
        "turretQE": {"command": turret_qe, "weight": abs(turret_dx)},
        "turretRF": {"command": turret_rf, "weight": abs(turret_dy)},
        "fire": fire_cmd
    }

    print(f"🧠 Action: {env.latest_action} → 🚗 Command: {command}")
    return jsonify(command)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 Simulation Started")
    return jsonify({"control": ""})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
