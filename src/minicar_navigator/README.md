# minicar_navigator

**TurtleBot4 + YOLOv8 웹캠 미니카 감지 → Nav2 자율 이동 패키지**

## 아키텍처

```
[웹캠]
  │
  ▼
[yolo_detector]  ──/minicar_detected──▶  [minicar_nav_manager]  ──/navigate_to_goal──▶  [nav2_controller]
                  ◀─/nav_status──────────────────────────────────────────────────────────────────────────
                                                                                              │
                                                                                              ▼
                                                                                    [Nav2 / TurtleBot4]
```

### 노드 설명

| 노드 | 역할 |
|------|------|
| `yolo_detector` | 웹캠 캡처 + YOLOv8 추론 → `/minicar_detected` 퍼블리시 |
| `minicar_nav_manager` | 디바운싱 + 상태 머신 → Nav2 트리거 조율 |
| `nav2_controller` | Nav2 `NavigateToPose` 액션 클라이언트 |

### 상태 머신 (Manager)

```
IDLE ──(N회 연속 감지)──▶ NAVIGATING ──(완료/실패)──▶ COOLDOWN ──(쿨다운)──▶ IDLE
```

---

## 설치

### 1. 의존성 설치

```bash
# ROS2 Humble Nav2
sudo apt install ros-humble-nav2-msgs ros-humble-cv-bridge ros-humble-vision-msgs

# Python 패키지
pip install ultralytics opencv-python
```

### 2. 패키지 빌드

```bash
cd ~/mini_proj
colcon build --packages-select minicar_navigator
source install/setup.bash
```

---

## 실행

### 기본 실행

```bash
ros2 launch minicar_navigator minicar_nav.launch.py
```

### 목표 지점 변경

```bash
ros2 launch minicar_navigator minicar_nav.launch.py goal_x:=2.5 goal_y:=-1.0 goal_yaw:=1.57
```

### 커스텀 YOLO 모델 사용

```bash
ros2 launch minicar_navigator minicar_nav.launch.py \
  model_path:=/path/to/best.pt \
  target_class:=minicar \
  confidence:=0.6
```

---

## 토픽 목록

| 토픽 | 타입 | 방향 | 설명 |
|------|------|------|------|
| `/minicar_detected` | `std_msgs/Bool` | yolo→manager | 미니카 감지 여부 |
| `/detection_image` | `sensor_msgs/Image` | yolo→외부 | 바운딩박스 시각화 |
| `/detection_result` | `vision_msgs/Detection2DArray` | yolo→외부 | 상세 감지 결과 |
| `/navigate_to_goal` | `std_msgs/Bool` | manager→nav2 | 네비게이션 트리거 |
| `/nav_status` | `std_msgs/String` | nav2→manager | 네비게이션 상태 |
| `/manager_state` | `std_msgs/String` | manager→외부 | 상태머신 현재 상태 |

---

## 파라미터 튜닝

`config/minicar_nav_params.yaml` 파일에서 수정:

```yaml
minicar_nav_manager:
  ros__parameters:
    detection_count_threshold: 5   # ← 낮추면 빠른 반응, 높이면 오감지 방지
    cooldown_sec: 10.0             # ← 도착 후 다시 이동하기 전 대기 시간

yolo_detector:
  ros__parameters:
    confidence: 0.5                # ← 낮추면 더 많이 감지, 높이면 정확도 상승

nav2_controller:
  ros__parameters:
    goal_x: 1.0                    # ← map 좌표계 기준 목표 지점
    goal_y: 1.0
```

---

## 감지 이미지 확인 (rqt)

```bash
ros2 run rqt_image_view rqt_image_view /detection_image
```

## 수동 네비게이션 테스트

```bash
# Nav2 없이 트리거만 테스트
ros2 topic pub --once /navigate_to_goal std_msgs/Bool "data: true"

# 상태 확인
ros2 topic echo /manager_state
ros2 topic echo /nav_status
```

---

## 커스텀 모델 학습 (미니카 전용)

```python
from ultralytics import YOLO

model = YOLO('best.pt')  # 기존 학습 모델 기반 파인튜닝
model.train(
    data='minicar_dataset.yaml',  # 미니카 데이터셋 설정 파일
    epochs=100,
    imgsz=640,
)
# 학습된 모델: runs/detect/train/weights/best.pt
```

데이터셋은 [Roboflow](https://roboflow.com/) 등에서 구성하거나
직접 웹캠으로 촬영 후 라벨링하여 사용하세요.
