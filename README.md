# mini_proj

TurtleBot4 OAK-D 카메라 (RGB / Stereo-Depth) 픽셀 단위 정렬 수동 튜닝 도구 모음.

## ⚠️ 중요 주의사항
- **절대 dock / undock 하지 마세요.**  
  도킹 중 TurtleBot4는 전력 절약을 위해 OAK-D 카메라 노드를 자동 종료합니다.  
  튜닝 작업 중에는 로봇을 자유 공간에 두고 진행하세요.

## 제공 스크립트

### 1. `tb4_cam_align_tuner.py` (추천)
RGB와 Depth 이미지를 실시간으로 **블렌드 / 사이드바이사이드 / 엣지 오버레이** 로 보여주는 뷰어.

- `ros2 topic list` 결과를 자동으로 파싱해 카메라 관련 토픽 출력
- **Edge Overlay 모드** (기본): RGB 영상 위에 Depth의 경계선(엣지)을 빨간색으로 그려서 **픽셀 정렬 오차를 육안으로 가장 쉽게 판단** 가능
- Blend 모드 (alpha 슬라이더)
- Side-by-side, Depth colormap 단독
- 캡처 기능 ('c' 키)
- rqt_reconfigure 사용법 + 추천 파라미터 상세 안내 내장

**실행**
```bash
python3 tb4_cam_align_tuner.py
# 또는 토픽 직접 지정
python3 tb4_cam_align_tuner.py \
  --rgb /oakd/rgb/preview/image_raw \
  --depth /oakd/stereo/depth
```

(네임스페이스가 있는 경우 예: `/turtlebot4_1/oakd/...`)

### 2. `tb4_camera_rqt_helper.sh`
rqt + 토픽 확인 + 안내를 한 번에 해주는 런처.

```bash
chmod +x tb4_camera_rqt_helper.sh
./tb4_camera_rqt_helper.sh
```

### 3. `my_oakd_rgbd_align.yaml`
RGB + Depth 를 활성화하고 **RGB 프레임에 depth 를 align** 하는 최소 설정 예시.

```bash
ros2 launch turtlebot4_bringup oakd.launch.py \
    params_file:=/home/ubuntu/my_oakd_rgbd_align.yaml
```

이 yaml 을 기반으로 해상도, stereo 필터, align 관련 값을 수정하면서 테스트하세요.

## rqt 에서 쉽게 튜닝하는 표준 워크플로우

1. **준비** (한 번만)
   - `tb4_camera_rqt_helper.sh` 또는 `tb4_cam_align_tuner.py` 실행해서 토픽 확인
   - 필요하면 `my_oakd_rgbd_align.yaml` 복사/수정 후 oakd 재런치

2. **rqt_reconfigure 열기** (파라미터 실시간/준실시간 조정)
   ```bash
   ros2 run rqt_reconfigure rqt_reconfigure
   ```
   - `oakd` 노드 선택 (네임스페이스 포함될 수 있음)

3. **이미지 뷰어들 열기**
   ```bash
   rqt_image_view
   ```
   추천 토픽:
   - `/oakd/rgb/preview/image_raw` (또는 `/oakd/rgb/image_raw`)
   - `/oakd/stereo/depth` (Colorize 모드 ON)
   - `/oakd/left/image_raw`, `/oakd/right/image_raw`

4. **Python 튜너로 정렬 품질 실시간 확인** (가장 강력)
   - `tb4_cam_align_tuner.py` 실행
   - 'e' 키 → Edge Overlay 모드
   - rqt_reconfigure 에서 값 바꾸면서 **빨간 엣지가 RGB 물체 경계와 정확히 일치하는지** 보면서 조정

5. **주요 조정 포인트**
   - `stereo.i_align_depth = true`
   - `stereo.i_board_socket_id = 0` (RGB 소켓)
   - RGB / Stereo 해상도 매칭 (i_width, i_height, isp scale)
   - stereo 필터들 (lr_check, subpixel, spatial/temporal filter 등)
   - RGB exposure / gain (특징점 풍부하게)

## 참고
- TurtleBot4 기본 launch 는 `i_pipeline_type: RGB` (depth 비활성) 입니다.
- Depth 를 쓰려면 `RGBD` 로 바꿔야 합니다.
- `i_` 접두사 파라미터는 재시작 필요, `r_` 접두사는 대부분 실시간 변경 가능.
- 공장 calibration 이 크게 틀어진 경우 수동 미세조정으로는 한계가 있을 수 있으며, depthai calibration tool 로 전체 재보정을 권장합니다.

작업 끝나면 스크립트들을 종료하고 안전하게 종료하세요.
