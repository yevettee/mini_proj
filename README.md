# mini_proj

TurtleBot4 + OAK-D 기반 **Depth 측정** 도구 (YOLO + 픽셀 클릭) 및 학습 파이프라인.

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
