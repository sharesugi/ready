from flask import Flask, request, jsonify
import random
import pandas as pd
import requests
import time  # ⏱️ 시간 체크용
import os 

app = Flask(__name__)

# === 전역 상태 ===
current_position = None
last_position = None
position_history = []
turret_info = []
bullet_data = []
obstacles = []

# 회전 & 발사 상태 관리
mode = "rotate"  # rotate → wait_for_fire → wait_for_impact
rotation_ticks = random.randint(10, 30)
is_waiting_for_bullet = False
fire = False
fire_timestamp = None  # 🔥 발사 시각 기록용

# 터렛 상태
turret_qe_action = random.choice(["Q", "E"])
turret_qe_ticks = random.randint(10, 30)
turret_rf_action = random.choice(["R", "F"])
turret_rf_ticks = random.randint(10, 30)

# 시뮬레이터 주소
SIMULATOR_URL = "http://127.0.0.1:5000/update_obstacle"

def calculate_turret_action():
    global turret_qe_action, turret_qe_ticks
    global turret_rf_action, turret_rf_ticks

    if turret_qe_ticks <= 0:
        turret_qe_action = random.choice(["Q", "E"])
        turret_qe_ticks = random.randint(10, 30)
    if turret_rf_ticks <= 0:
        turret_rf_action = random.choice(["R", "F"])
        turret_rf_ticks = random.randint(10, 30)

    turret_qe_ticks -= 1
    turret_rf_ticks -= 1

    return turret_qe_action, turret_rf_action

@app.route('/get_action', methods=['POST'])
def get_action():
    global current_position, last_position, turret_info, fire
    global is_waiting_for_bullet, mode, rotation_ticks, fire_timestamp
    global turret_qe_action, turret_rf_action

    data = request.get_json(force=True)
    position = data.get("position", {})
    turret = data.get("turret", {})

    x, y, z = position.get("x", 0), position.get("y", 0), position.get("z", 0)
    if current_position:
        last_position = current_position
    current_position = (x, z)
    position_history.append(current_position)

    moveWS = {"command": "STOP", "weight": 1.0}
    moveAD = {"command": "", "weight": 0.0}
    turretQE = {"command": "", "weight": 0.0}
    turretRF = {"command": "", "weight": 0.0}

    if mode == "rotate":
        if rotation_ticks <= 0:
            mode = "wait_for_fire"
            print("⏸️ 회전 정지 → 발사 준비")
        else:
            moveAD["command"] = random.choice(["A", "D"])
            moveAD["weight"] = 1.0
            qe, rf = calculate_turret_action()
            turretQE = {"command": qe, "weight": 1.0}
            turretRF = {"command": rf, "weight": 1.0}
            rotation_ticks -= 1

    elif mode == "wait_for_fire":
        if not is_waiting_for_bullet:
            fire = True
            is_waiting_for_bullet = True
            fire_timestamp = time.time()  # 🔥 발사 시간 기록
            mode = "wait_for_impact"
            turret_info.clear()
            my_tank_info = [x, y, z, turret.get('x', 0), turret.get('y', 0)]
            turret_info.extend(my_tank_info)
            print(f"💥 발사! 위치 저장: {turret_info}")

    elif mode == "wait_for_impact":
        if fire_timestamp and time.time() - fire_timestamp > 5:
            print("⚠️ 낙하 응답 없음. 데이터 저장 건너뜀. 회전 재개")
            is_waiting_for_bullet = False
            fire_timestamp = None
            turret_info.clear()
            mode = "rotate"
            rotation_ticks = random.randint(10, 30)

    command = {
        "moveWS": moveWS,
        "moveAD": moveAD,
        "turretQE": turretQE,
        "turretRF": turretRF,
        "fire": fire
    }

    if fire:
        fire = False

    print(f"🚗 [mode={mode}] 위치: {current_position} | 명령: {command}")
    return jsonify(command)

@app.route('/update_bullet', methods=['POST'])
def update_bullet():
    global turret_info, is_waiting_for_bullet
    global mode, rotation_ticks, fire_timestamp

    data = request.get_json()
    bullet_x, bullet_y, bullet_z = data.get("x", 0), data.get("y", 0), data.get("z", 0)

    if len(turret_info) == 5:
        turret_info.extend([bullet_x, bullet_y, bullet_z])
        new_row = pd.DataFrame([turret_info], columns=[
            "x_pos", "y_pos", "z_pos",
            "x_angle", "y_angle",
            "x_target", "y_target", "z_target"
        ])

        filename = "turret_info_0602/turret_info_0602.csv"

        # 🔽 파일이 없으면 header 포함해서 저장, 있으면 header 없이 append
        if not os.path.exists(filename):
            new_row.to_csv(filename, index=False)
            print("📥 CSV 생성 및 첫 행 저장 완료")
        else:
            new_row.to_csv(filename, mode='a', index=False, header=False)
            print("📎 CSV 이어쓰기 완료")

        # 상태 초기화
        is_waiting_for_bullet = False
        fire_timestamp = None
        mode = "rotate"
        rotation_ticks = random.randint(10, 30)
        print("🔄 회전 재개")

    return jsonify({'status': 'bullet processed'})


@app.route('/update_obstacle', methods=['POST'])
def update_obstacle():
    global obstacles
    data = request.get_json()
    obstacles = data.get("obstacles", [])
    print(f"🪨 장애물 업데이트: {len(obstacles)}개")
    return jsonify({'status': 'success'})

@app.route('/init', methods=['GET'])
def init():
    config = {
        "startMode": "pause",
        "blStartX": 150,
        "blStartY": 7.8,
        "blStartZ": 150,
        "rdStartX": 171.1265,
        "rdStartY": 8,
        "rdStartZ": 275.5405,
        "trackingMode": False,
        "detactMode": False,
        "logMode": False,
        "enemyTracking": False,
        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000
    }
    print("🛠️ Init config:", config)
    return jsonify(config)

@app.route('/start', methods=['GET'])
def start():
    print("🚀 시뮬레이션 시작")
    return jsonify({"control": ""})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'obstacle_count': len(obstacles),
        'obstacles': obstacles
    })

@app.route('/info', methods=['POST'])
def info():
    return jsonify({"status": "success", "control": ""})

@app.route('/collision', methods=['POST'])
def collision():
    data = request.get_json()
    print(f"💥 충돌: {data.get('objectName')} at {data.get('position')}")
    return jsonify({'status': 'success'})

@app.route('/set_destination', methods=['POST'])
def set_destination():
    data = request.get_json()
    if "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400
    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 목적지 설정: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
