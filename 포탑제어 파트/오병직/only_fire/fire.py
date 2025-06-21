import math
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# 터렛 각도 예측 모델 및 전처리기기 파일 경로
DNN_PATH = "./models/best_dnn_model.h5"
XGB_PATH = "./models/best_xgb_model.pkl"
SCALER_PATH = "./models/scaler.pkl"
POLY_PATH = "./models/poly_transformer.pkl"

# 모델 및 전처리기 불러오기
model = load_model(DNN_PATH)
xgb_model = joblib.load(XGB_PATH)
scaler = joblib.load(SCALER_PATH)
poly = joblib.load(POLY_PATH)

# 화면 상에 그려진 바운딩 박스가 라이다의 어느 각도에 있는지를 찾는 함수
def get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_horizontal, fov_vertical):
    # 중심 좌표
    x_center = (bbox["x1"] + bbox["x2"]) / 2
    y_center = (bbox["y1"] + bbox["y2"]) / 2

    # 정규화 (0~1)
    x_norm = x_center / image_width
    y_norm = y_center / image_height

    # 각도 변환 (중앙 기준, 좌/위가 음수, 우/아래가 양수) - 화면상의 위치로 라이다에 매칭시킴
    h_angle = (x_norm - 0.5) * fov_horizontal
    v_angle = (0.5 - y_norm) * fov_vertical  # y축은 반대로 계산 (위가 0)

    return h_angle, v_angle

# 위 함수에서 가져온 각도로 바운딩 박스 안에 찍히는 라이다 값을 모두 가져와 평균값을 return 하는 함수
def find_lidar_cluster_center_adaptive(lidar_points, h_angle, v_angle,
                                       bbox_width_ratio, bbox_height_ratio,
                                       fov_horizontal=47.81061,
                                       fov_vertical=28.0):
    BOX_THRESHOLD = 0.5

    # 바운딩박스 크기에 따라 허용 각도 조정
    h_angle_tol = bbox_width_ratio * fov_horizontal * BOX_THRESHOLD
    v_angle_tol = bbox_height_ratio * fov_vertical * BOX_THRESHOLD

    # 전체 라이다 데이터에서 박스안에 해당하는 라이다 포인트만 저장
    candidates = [
        p for p in lidar_points
        if p["isDetected"]
        and abs((p["angle"] - h_angle + 180) % 360 - 180) < h_angle_tol
        and abs(p.get("verticalAngle", 0) + v_angle) < v_angle_tol
    ]

    # 박스가 그려진 각도에 라이다 값이 없다면 (여기가 문제. 라이다 데이터를 촘촘하게 받지 않으면 못찾음.)
    if not candidates:
        print(f'❌ There is no candidates')
        return None

    # 바운딩 박스 안에 찍힌 라이다 포인트들의 평균 좌표 및 거리
    avg_x = sum(p["position"]["x"] for p in candidates) / len(candidates)
    avg_y = sum(p["position"]["y"] for p in candidates) / len(candidates)
    avg_z = sum(p["position"]["z"] for p in candidates) / len(candidates)
    avg_dist = sum(p["distance"] for p in candidates) / len(candidates)

    return {
        "position": {"x": avg_x, "y": avg_y, "z": avg_z},
        "distance": avg_dist
    }

# 위 두 함수를 사용하여 우리가 필요한 실제 감지된 전차의 좌표를 return 해주는 함수
def match_yolo_to_lidar(bboxes, lidar_points, image_width, image_height, fov_h, fov_v):
    results = []
    for bbox in bboxes:
        h_angle, v_angle = get_angles_from_yolo_bbox(bbox, image_width, image_height, fov_h, fov_v)

        # 바운딩박스 비율 계산
        bbox_width_ratio = (bbox["x2"] - bbox["x1"]) / image_width
        bbox_height_ratio = (bbox["y2"] - bbox["y1"]) / image_height

        # LiDAR 클러스터 추정
        cluster = find_lidar_cluster_center_adaptive(
            lidar_points, h_angle, v_angle,
            bbox_width_ratio, bbox_height_ratio,
            fov_horizontal=fov_h,
            fov_vertical=fov_v
        )

        if cluster:
            results.append({
                "bbox": bbox,
                "matched_lidar_pos": cluster["position"],
                "distance": cluster["distance"]
            })
    return results

# 내 전차의 x, z좌표, 목표 전차의 x, z좌표로 터렛이 바라봐야 하는 x각도 return
# 모델 x 단순 계산
def get_yaw_angle(player_pos, enemy_pos):
    dx = enemy_pos['x'] - player_pos['x']
    dz = enemy_pos['z'] - player_pos['z']

    # atan2는 radian 기준, z를 먼저 넣는 이유는 좌표계 때문
    angle_rad = math.atan2(dx, dz)
    angle_deg = math.degrees(angle_rad)

    # 0~359로 변환
    angle_deg = (angle_deg + 360) % 360

    return round(angle_deg, 2)

# 학습시킨 dnn 모델로 터렛의 y 각도 예측
# 내 전차의 좌표, 적 전차의 좌표를 사용해 거리와 dy를 구하여 입력으로 넣음.
# 출력은 y 각도
def find_angle_for_distance_dy_dnn(distance, dy):
    # ✅ 예측용 입력 설정 (Distance + dy)
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)

    # ✅ 예측
    y_pred = model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

    return float(y_pred_angle)

# 학습시킨 xgb 모델로 터렛의 y 각도 예측
# 내 전차의 좌표, 적 전차의 좌표를 사용해 거리와 dy를 구하여 입력으로 넣음.
# 출력은 y 각도
def find_angle_for_distance_dy_xgb(distance, dy):
    # ✅ 예측용 입력 설정 (Distance + dy)
    X_input = np.array([[distance, dy]])
    X_poly = poly.transform(X_input)
    X_scaled = scaler.transform(X_poly)

    # ✅ 예측
    y_pred = xgb_model.predict(X_scaled)
    y_pred_angle = np.rad2deg(np.arctan2(y_pred[:, 0], y_pred[:, 1]))

    return float(y_pred_angle)

def distance_3d(a, b):
    return math.sqrt(
        (a['x'] - b['x'])**2 +
        (a['y'] - b['y'])**2 +
        (a['z'] - b['z'])**2
    )