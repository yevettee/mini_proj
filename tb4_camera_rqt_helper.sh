#!/bin/bash
#
# TurtleBot4 OAK-D Camera Alignment - rqt Helper Launcher
#
# 사용법:
#   chmod +x tb4_camera_rqt_helper.sh
#   ./tb4_camera_rqt_helper.sh
#
# 이 스크립트는:
#   1. dock/undock 절대 금지 경고
#   2. 관련 토픽 빠른 확인
#   3. rqt_reconfigure + rqt_image_view 추천 런처
#   4. Python 튜너 스크립트 실행 안내
#

set -e

echo "==================================================================="
echo "🚨🚨🚨  TurtleBot4 Camera Config Tuning Helper  🚨🚨🚨"
echo "==================================================================="
echo ""
echo "⚠️  절대 DOCK / UNDOCK 하지 마세요!"
echo "   - 도킹 중 카메라 노드가 종료됩니다."
echo "   - 튜닝 작업 전체 동안 로봇을 자유 공간에 두세요."
echo ""
read -p "이해했으면 'yes' 입력: " confirm
if [ "$confirm" != "yes" ]; then
    echo "취소됨."
    exit 1
fi

echo ""
echo "=== [1] ros2 topic list (oak / rgb / depth / stereo 필터) ==="
ros2 topic list 2>/dev/null | grep -E 'oak|rgb|depth|stereo|left|right|camera' || echo "(필터된 토픽 없음 - bringup 상태 확인)"

echo ""
echo "=== [2] 추천 rqt 실행 (별도 터미널에서 실행하세요) ==="
echo ""
echo "터미널 A (파라미터 실시간 조정 - 가장 중요):"
echo "  ros2 run rqt_reconfigure rqt_reconfigure"
echo ""
echo "터미널 B (이미지 개별 확인):"
echo "  rqt_image_view"
echo "    또는"
echo "  ros2 run rqt_image_view rqt_image_view"
echo ""
echo "rqt_image_view 추천 토픽:"
echo "  /oakd/rgb/preview/image_raw"
echo "  /oakd/stereo/depth   (Colorize 옵션 켜기)"
echo "  /oakd/left/image_raw"
echo "  /oakd/right/image_raw"
echo ""

echo "=== [3] Python Alignment Visual Tuner (추천) ==="
echo "이 디렉토리의 tb4_cam_align_tuner.py 를 사용하면"
echo "RGB + Depth를 블렌드/엣지 오버레이로 동시에 보면서"
echo "rqt_reconfigure 로 값을 바꿔가며 픽셀 정렬을 육안 확인할 수 있습니다."
echo ""
echo "실행 예:"
echo "  python3 tb4_cam_align_tuner.py"
echo "  # 또는"
echo "  python3 tb4_cam_align_tuner.py --rgb /oakd/rgb/preview/image_raw --depth /oakd/stereo/depth"
echo ""

echo "=== [4] 수동 config 변경이 필요할 때 ==="
echo "1. oakd_pro.yaml (또는 oakd_lite.yaml) 복사"
echo "2. i_pipeline_type: RGBD 로 변경 + stereo 섹션 추가 (i_align_depth: true 등)"
echo "3. ros2 launch turtlebot4_bringup oakd.launch.py params_file:=/path/to/my_config.yaml"
echo ""
echo "자세한 예시는 tb4_cam_align_tuner.py 파일 상단 주석과 SAMPLE_YAML 참고."
echo ""

echo "준비되면 위 명령들을 복사해서 사용하세요."
echo "Python 튜너를 바로 실행하시겠습니까? (y/n)"
read -r answer
if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
    python3 "$(dirname "$0")/tb4_cam_align_tuner.py"
fi
