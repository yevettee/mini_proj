# mini_proj

TurtleBot4 + OAK-D 기반 **Depth 측정** 도구 (YOLO + 픽셀 클릭) 및 학습 파이프라인.

---

## 🚗 Gazebo 시뮬레이션 (minicar_room) — `gazebo` 브랜치

RC car 탐지→접근 미니프로젝트를 Gazebo(Ignition Fortress)에서 돌리기 위한 세팅입니다.
실로봇 없이 시뮬에서 검출/접근을 개발·테스트할 수 있습니다.

### 0. 사전 준비
```bash
# (1) turtlebot4 시뮬 워크스페이스 (이미 있으면 skip)
#     ~/turtlebot4_ws/src 에 turtlebot4, turtlebot4_simulator 클론 후 colcon build
# (2) 이 저장소를 ~/mini_proj 에 클론 (launch 기본 맵 경로가 ~/mini_proj 기준)
git clone -b gazebo https://github.com/yevettee/mini_proj.git ~/mini_proj
# (3) ultralytics (YOLO) 설치
pip install ultralytics
```

### 1. turtlebot4_ws 오버라이드 적용
방 SDF·SLAM/Nav2 튜닝·map 인자 plumbing이 turtlebot4 패키지 안에 있으므로 덮어씁니다:
```bash
cd ~/mini_proj/gazebo_setup
bash apply_overrides.sh                 # 기본 ~/turtlebot4_ws (--symlink-install이면 재빌드 불필요)
```
자세한 내용은 [`gazebo_setup/README.md`](gazebo_setup/README.md).

### 2. `~/.bashrc` 설정
새 터미널이 sim 모드로 뜨고, `minisim` 한 줄로 시뮬을 띄우게 합니다:
```bash
# ===== ROS2 / TurtleBot4 Gazebo 환경 =====
source /opt/ros/humble/setup.bash
source ~/turtlebot4_ws/install/setup.bash
source ~/mini_proj/install/setup.bash
export IGNITION_VERSION=fortress

# 기본 = SIM 모드 (도메인 0, localhost 전용, discovery 미사용)
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1
unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT

# SIM 모드 재적용 (다른 모드에서 돌아올 때)
alias sim='unset ROS_DISCOVERY_SERVER ROS_SUPER_CLIENT; export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1; ros2 daemon stop; ros2 daemon start; echo "🖥️  SIM 모드"'

# minicar_room 시뮬 실행 (nav2 + localization(저장맵) + rviz)
alias minisim='ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py world:=minicar_room model:=standard nav2:=true localization:=true rviz:=true x:=0.0 y:=0.0 yaw:=3.14159'
```
> 실로봇(디스커버리 서버)을 쓰는 랩 환경이면 별도로 `robot` alias를 추가하세요(랩 IP/도메인 의존). Gazebo만 쓰면 위 sim 설정으로 충분합니다.

### 3. minicar_navigator 빌드
```bash
cd ~/mini_proj
colcon build --packages-select minicar_navigator
source install/setup.bash
```

### 4. 실행

**시뮬 띄우기** (터미널 A):
```bash
sim          # (새 터미널은 이미 sim 모드지만 안전하게)
minisim      # Gazebo + RViz + nav2 + localization(minicar_sim_map) 기동
```

**탐지 + 접근만 격리 테스트** (터미널 B) — RC car가 카메라에 보이게 로봇을 놓은 뒤:
```bash
# 접근 노드만 단독 실행 (manager/nav2_controller 없이)
ros2 run minicar_navigator oakd_approach_node --ros-args \
  --params-file ~/mini_proj/install/minicar_navigator/share/minicar_navigator/config/minicar_nav_params_sim.yaml \
  -r __node:=oakd_approach_node -p use_sim_time:=true

# 시작 트리거 (터미널 C, 한 번)
ros2 topic pub --once /start_approach std_msgs/msg/Bool "{data: true}"
```
상태: `IDLE → SEARCHING →(YOLO 'car' 감지)→ APPROACHING →(depth ≤ 0.5m)→ ARRIVED`. SEARCHING은 제자리 대기(회전 안 함)이므로 차가 시야에 있어야 합니다.

**전체 흐름** (웹캠 없는 B안):
```bash
ros2 launch minicar_navigator minicar_nav_sim.launch.py goal_x:=<x> goal_y:=<y> goal_yaw:=<yaw>
ros2 topic pub --once /minicar_detected std_msgs/msg/Bool "{data: true}"   # 시작
```

### ⚠️ 알려진 이슈 / 트러블슈팅
- **cmd_vel 충돌**: 접근 노드는 `/cmd_vel`에 직접 publish하는데 nav2의 velocity_smoother도 `/cmd_vel`을 점유 → nav2 켠 상태에선 로봇이 안 움직일 수 있음. 접근 단독 테스트는 시뮬을 `nav2:=false`로 띄우거나, 코드에서 접근 노드를 `cmd_vel_nav`로 보내도록 수정 필요.
- **카메라/scan 토픽 수신 안 됨**: 별도 launch로 띄운 노드가 큰 토픽(이미지)을 못 받으면 `export ROS_LOCALHOST_ONLY=0` 후 `ros2 daemon stop && ros2 daemon start` (localhost SHM 이슈).
- **로봇이 떨림 / 중복 스택**: minisim을 여러 번 띄우면 충돌. 정리: `pkill -9 -f "ign gazebo|parameter_bridge|slam_toolbox|turtlebot4_node"` 후 `rm -f /dev/shm/fastrtps_*`, 그리고 하나만 다시 실행.
- **맵이 warehouse로 뜸**: localization 기본맵 미적용 — 오버라이드(`apply_overrides.sh`)를 적용했는지, 저장소가 `~/mini_proj` 인지 확인.

---

> **참고**: 이전 RGB-Stereo alignment 수동 튜닝 도구들(`tb4_cam_align_tuner.py`, 관련 YAML, rqt helper 등)은 OAK-D 기본 설정에 내장/통합되어 더 이상 별도 유지하지 않습니다.  
> Alignment 관련 설정(`stereo.i_align_depth`, 해상도 매칭 등)은 `turtlebot4_bringup` + `depthai_ros_driver` launch 시 params_file 로 직접 지정하세요.

## ⚠️ 중요 주의사항
- **절대 dock / undock 하지 마세요.**  
  TurtleBot4 도킹 중에는 전력 절약을 위해 OAK-D 카메라 노드가 자동 종료됩니다.  
  카메라/Depth 사용하는 작업 중에는 로봇을 자유 공간에 두고 진행하세요.
- Depth 정확도를 위해 **RGB와 Stereo/Depth 해상도 일치** + `stereo.i_align_depth=true` + `i_board_socket_id=0` (RGB 소켓) 설정이 중요합니다.
- 해상도 불일치 시 YOLO bbox와 depth 픽셀이 어긋나 거리 측정이 완전히 틀릴 수 있습니다.

## 제공 노드 (rokey_pjt ROS2 패키지)

### 1. `depth_checker` (YOLO + Depth 측정, 추천)
YOLOv8로 객체 검출 후, 각 바운딩박스 영역의 depth(median/min/center)를 실시간 계산/시각화.

- RGB + Depth 이미지 구독 → YOLO 추론 → BBox별 거리 표시 (m 단위)
- 마우스 클릭으로 해당 픽셀 depth 수동 확인 지원
- ROS 파라미터로 topic, model_path, conf_threshold, depth_method, inference_hz 등 조정 가능
- 시각화 창 + Depth colormap 별도 창 (옵션)

**실행 예시:**
```bash
# 기본 (패키지 빌드 후)
ros2 run rokey_pjt depth_checker

# 커스텀 모델 + 토픽 + 파라미터
ros2 run rokey_pjt depth_checker --ros-args \
  -p rgb_topic:=/oakd/rgb/preview/image_raw \
  -p depth_topic:=/oakd/stereo/image_raw \
  -p model_path:=/home/woody/mini_proj/models/runs/YOLO_Tournament/Round1_v8_Nano/weights/best.pt \
  -p conf_threshold:=0.45 \
  -p depth_method:=median \
  -p inference_hz:=6.0
```

**기본 모델 경로**: `~/mini_proj/models/runs/YOLO_Tournament/Round1_v8_Nano/weights/best.pt` (필요 시 override)

### 2. `depth_checker_mouse` (YOLO 없이 간단 픽셀 클릭 Depth 측정)
가벼운 테스트나 YOLO 없이 depth 이미지만 보고 클릭으로 거리 확인할 때 사용.

- Depth 이미지 colormap 표시
- 왼쪽 클릭 → 해당 픽셀의 거리 (m) 로그 출력
- ROS 파라미터로 topic 쉽게 변경

**실행 예시:**
```bash
ros2 run rokey_pjt depth_checker_mouse
# 또는
ros2 run rokey_pjt depth_checker_mouse --ros-args -p depth_topic:=/oakd/stereo/depth
```

**의존성**: rclpy, sensor_msgs, cv_bridge, opencv-python, numpy (ultralytics/torch 는 YOLO 버전만)

## ROS 패키지 빌드/사용
```bash
cd /path/to/your_ws
colcon build --packages-select rokey_pjt
source install/setup.bash
```

`src/rokey_pjt/` 아래에 소스가 있으며, `setup.py` entry_points 로 두 노드가 등록되어 있습니다.

## 학습 / 데이터 수집 (models/ + tools/)

- `tools/my_capture.py`: USB 웹캠 멀티스레드 고속 캡처 (YOLO 데이터셋 수집용)
  - `c` 키 캡처, `q` 종료
  - 저장 위치: `models/cam_images/img_capture_threaded/`
  ```bash
  python3 tools/my_capture.py
  ```

- `models/` 디렉토리:
  - `yolo_train.py`: 단일 실험 학습
  - `dataset_split.py`: 도메인( AMR / Webcam ) 별 7:2:1 분할
  - `model_comparison/scripts/`: 다중 모델/조건 비교 파이프라인 (runner.py, analyzer.py)
  - `data.yaml`, `dataset_split/`, `runs/`, `cam_images/` (라벨링된 이미지 포함)
  - 학습 결과 best.pt 를 `depth_checker` 의 model_path 로 사용

자세한 학습/실험은 `models/` 내부 README나 스크립트 헤더, `model_comparison/scripts/README.md` 참고.

## 참고
- TurtleBot4 기본 oakd launch 는 `i_pipeline_type: RGB` 입니다. Depth 를 쓰려면 `RGBD` 로 바꾸고 align 관련 파라미터를 적용하세요.
- `i_` 파라미터는 oakd 노드 재시작 필요, `r_` 는 rqt_reconfigure 로 대부분 실시간 조정.
- 공장 calibration 이 크게 틀어진 경우 depthai calibration tool 로 전체 재보정 권장.
- depth_checker 내부에 해상도 미스매치 경고 + alignment 설정 확인 메시지가 포함되어 있습니다.

작업 끝나면 노드 종료하고 안전하게 로봇을 다루세요.
