#!/usr/bin/env python3
"""
TurtleBot4 OAK-D RGB / Stereo-Depth Pixel Alignment Tuner

이 스크립트는 rqt + 수동 파라미터 튜닝을 쉽게 하기 위한 보조 도구입니다.
- ros2 topic list 로 관련 토픽 자동 탐색/출력
- RGB + Depth 실시간 오버레이/블렌드/엣지 비교 뷰어 제공 (픽셀 정렬 품질을 육안으로 빠르게 판단)
- rqt_reconfigure 실행법 + 추천 파라미터 안내
- 절대 dock/undock 금지 경고

사용법 (ROS2 환경 + depthai_ros_driver 실행 중이어야 함):
    python3 tb4_cam_align_tuner.py

또는 토픽 이름을 직접 지정:
    python3 tb4_cam_align_tuner.py \
        --rgb /oakd/rgb/preview/image_raw \
        --depth /oakd/stereo/depth

rqt에서 파라미터 조정하면서 이 뷰어로 실시간 확인하세요.
"""

import argparse
import subprocess
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[경고] rclpy / cv_bridge 를 찾을 수 없습니다. ROS2 환경에서 실행하세요.", file=sys.stderr)


# ==================== 경고 (절대 dock/undock 금지) ====================
DOCK_WARNING = """
================================================================================
🚨🚨🚨  절대 DOCK / UNDOCK 하지 마세요!  🚨🚨🚨
================================================================================
- TurtleBot4는 도킹 중에 전력 절약을 위해 OAK-D 카메라 노드가 자동 종료됩니다.
- 카메라 config 튜닝 중에 dock/undock 하면 노드가 죽고, 토픽이 사라지며,
  재시작 시 calibration 상태가 꼬일 수 있습니다.
- 튜닝 작업이 완전히 끝날 때까지 로봇을 **자유 공간에 두고** 작업하세요.
- 작업 후 종료할 때도 dock 버튼 누르지 마세요.
================================================================================
"""

# ==================== 추천 rqt 명령어 및 파라미터 ====================
RQT_INSTRUCTIONS = """
[ rqt 에서 쉽게 튜닝하는 방법 ]

1. 별도 터미널에서 rqt_reconfigure 실행:
   $ ros2 run rqt_reconfigure rqt_reconfigure

2. 노드 목록에서 'oakd' (또는 네임스페이스 포함 예: /turtlebot4_xx/oakd) 선택

3. 주요 튜닝 대상 파라미터 (depthai_ros_driver):

   [RGB 품질 / 특징점 관련 - 런타임 변경 가능(r_ 접두사)]
     - rgb.r_exposure, rgb.r_iso, rgb.r_set_man_exposure
     - rgb.r_brightness, rgb.r_contrast 등

   [Stereo / Depth 품질 및 정렬 관련]
     - stereo.i_align_depth          : true 로 설정 (RGB에 depth align)
     - stereo.i_board_socket_id      : 0  (RGB 카메라 소켓, OAK-D에서 보통 CAM_A=0)
     - stereo.i_lr_check             : true (Left-Right consistency check)
     - stereo.i_extended_disparity   : false 또는 true (가까운 거리)
     - stereo.i_subpixel             : true (더 부드러운 depth)
     - stereo.i_depth_preset         : HIGH_ACCURACY 또는 DEFAULT
     - stereo.r_spatial_filter_...   (런타임 필터들, depth 노이즈/엣지 개선)

   [해상도 매칭 (중요: RGB와 Depth 픽셀 크기 맞추기)]
     - rgb.i_width, rgb.i_height, rgb.i_isp_num/den (scale)
     - stereo.i_width, stereo.i_height
     - preview 관련: i_preview_size, i_keep_preview_aspect_ratio

   ⚠️  'i_' 로 시작하는 파라미터는 보통 초기화 시에만 적용됩니다.
       값을 바꾼 후에는 oakd 노드를 재시작해야 합니다.
       (서비스: /oakd/stop, /oakd/start 또는 launch 재실행)

4. rqt_image_view 로 별도 확인 추천:
   $ rqt_image_view
   (또는 ros2 run rqt_image_view rqt_image_view)

   추천 토픽:
     - /oakd/rgb/preview/image_raw   (또는 /oakd/rgb/image_raw)
     - /oakd/stereo/depth   또는  /oakd/stereo/depth/image_raw (Colorize 모드 추천)
     - /oakd/left/image_raw, /oakd/right/image_raw (stereo raw 확인용)

5. 추가로 RViz2에서 PointCloud2 + Image overlay 로 3D 정렬 확인 가능.
"""

# ==================== 샘플 config YAML (주석 참고) ====================
SAMPLE_YAML = """
# turtlebot4 RGBD + RGB-depth alignment 수동 튜닝용 최소 예시
# 사용법:
#   1. 이 내용을 my_oakd_rgbd.yaml 로 저장 (~/my_oakd_rgbd.yaml)
#   2. 기존 bringup 중지 (또는 서비스 중지)
#   3. ros2 launch turtlebot4_bringup oakd.launch.py params_file:=/home/ubuntu/my_oakd_rgbd.yaml
#
# 네임스페이스가 있는 경우 root_key 를 맞춰야 합니다 (launch가 자동 처리).

/oakd:
  ros__parameters:
    camera:
      i_enable_imu: false
      i_enable_ir: true
      i_floodlight_brightness: 0
      i_laser_dot_brightness: 100
      i_nn_type: none
      i_pipeline_type: RGBD          # ★ RGB + Stereo Depth 활성화 핵심
      i_usb_speed: SUPER_PLUS

    rgb:
      i_board_socket_id: 0
      i_fps: 15.0
      i_resolution: '720'            # 또는 '1080' 등. Depth 해상도와 맞추기 중요
      i_width: 1280
      i_height: 720
      i_enable_preview: true
      i_preview_size: 640            # 네트워크 부하에 따라 조절 (정렬 확인 시 크게)
      i_publish_topic: true          # /oakd/rgb/image_raw 도 보고 싶을 때
      i_low_bandwidth: true

    stereo:
      i_board_socket_id: 0           # ★ RGB(0)에 depth align
      i_align_depth: true            # ★ RGB 프레임에 맞춰 depth 생성 (픽셀 정렬 핵심)
      i_output_depth: true
      i_lr_check: true
      i_extended_disparity: false
      i_subpixel: true
      i_depth_preset: HIGH_ACCURACY
      i_publish_topic: true

    # 필요시 left/right raw 도 활성화
    # left:
    #   i_publish_topic: true
    # right:
    #   i_publish_topic: true
"""


def run_ros2_topic_list() -> str:
    """ros2 topic list 실행 결과를 문자열로 반환."""
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=8
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[에러] ros2 topic list 실행 실패: {e}"


def find_camera_topics(topic_list_output: str) -> list:
    """카메라 관련 토픽 후보 필터링."""
    keywords = ["oak", "rgb", "depth", "stereo", "left", "right", "camera"]
    candidates = []
    for line in topic_list_output.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if any(kw in lower for kw in keywords):
            candidates.append(line)
    return candidates


class CameraAlignViewer(Node):
    def __init__(self, rgb_topic: str, depth_topic: str):
        super().__init__("tb4_cam_align_viewer")
        self.bridge = CvBridge()
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth: Optional[np.ndarray] = None
        self.lock = threading.Lock()

        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, depth_topic, self.depth_callback, 10
        )

        self.get_logger().info(f"구독 시작: RGB={rgb_topic}, Depth={depth_topic}")

    def rgb_callback(self, msg: Image):
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.latest_rgb = cv_img.copy()
        except CvBridgeError as e:
            self.get_logger().warn(f"RGB cv_bridge 변환 실패: {e}")

    def depth_callback(self, msg: Image):
        try:
            # depth 는 보통 16UC1 (mm) 또는 32FC1
            if msg.encoding in ("16UC1", "mono16"):
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="16UC1")
            elif msg.encoding in ("32FC1", "mono32"):
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
                cv_img = (cv_img * 1000).astype(np.uint16)  # m -> mm 대략
            else:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.latest_depth = cv_img.copy()
        except CvBridgeError as e:
            self.get_logger().warn(f"Depth cv_bridge 변환 실패: {e}")

    def get_frames(self):
        with self.lock:
            return (
                self.latest_rgb.copy() if self.latest_rgb is not None else None,
                self.latest_depth.copy() if self.latest_depth is not None else None,
            )


def normalize_depth_for_display(depth: np.ndarray, max_range_mm: int = 10000) -> np.ndarray:
    """Depth를 시각화용 uint8 그레이스케일로 정규화."""
    if depth is None:
        return None
    d = depth.astype(np.float32)
    d = np.clip(d, 0, max_range_mm)
    d = (d / max_range_mm * 255).astype(np.uint8)
    return d


def make_colormap_depth(gray: np.ndarray) -> np.ndarray:
    """Jet colormap 적용."""
    if gray is None:
        return None
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored


def detect_edges(depth_gray: np.ndarray, canny_low: int = 30, canny_high: int = 80) -> np.ndarray:
    """Depth에서 Canny 엣지 검출 (정렬 확인용)."""
    if depth_gray is None:
        return None
    blurred = cv2.GaussianBlur(depth_gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    return edges


def run_viewer(rgb_topic: str, depth_topic: str):
    if not ROS_AVAILABLE:
        print("ROS2 환경이 아니므로 뷰어를 실행할 수 없습니다.", file=sys.stderr)
        return

    rclpy.init()
    node = CameraAlignViewer(rgb_topic, depth_topic)

    # rclpy spin을 별도 스레드로
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("\n[뷰어 단축키]")
    print("  'b' : Blend (RGB + Depth 알파 블렌딩)")
    print("  's' : Side-by-side")
    print("  'e' : Edge overlay (RGB 위에 Depth 엣지)  ← 픽셀 정렬 확인에 가장 유용!")
    print("  'd' : Depth colormap 단독")
    print("  'c' : 현재 화면 캡처 (img_align_check_*.png)")
    print("  'q' 또는 ESC : 종료")
    print("  트랙바: Edge overlay 시 Canny threshold 조절\n")

    # OpenCV 창 + 트랙바
    window_name = "TB4 RGB-Depth Alignment Tuner (Edge overlay 추천)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # ★ 중요: Qt 백엔드(OpenCV 4.x 일부 빌드)에서는 namedWindow 직후 createTrackbar가
    #   "NULL window handler" 에러를 일으킴. 반드시 먼저 imshow + waitKey(1)로
    #   윈도우를 실제로 생성한 후에 trackbar를 만들어야 함.
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initializing viewer...", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow(window_name, placeholder)
    cv2.waitKey(1)   # Qt 이벤트 루프에 윈도우 생성을 강제

    # 이제 안전하게 트랙바 생성
    try:
        cv2.createTrackbar("Canny Low", window_name, 30, 200, lambda x: None)
        cv2.createTrackbar("Canny High", window_name, 80, 300, lambda x: None)
        cv2.createTrackbar("Blend Alpha %", window_name, 50, 100, lambda x: None)
        cv2.createTrackbar("MaxDepth(mm)", window_name, 10000, 20000, lambda x: None)
    except cv2.error as e:
        print(f"[경고] 트랙바 생성 실패 (일부 OpenCV+Qt 환경에서 발생): {e}")
        print("       트랙바 없이도 키보드(b/s/e/d/c/q)로는 정상 동작합니다.")

    mode = "edge"  # 기본을 edge overlay 로
    capture_count = 0

    try:
        while True:
            rgb, depth = node.get_frames()

            if rgb is None and depth is None:
                msg = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(msg, "Waiting for frames...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(window_name, msg)
                key = cv2.waitKey(50) & 0xFF
                if key in (ord('q'), 27):
                    break
                continue

            # 트랙바 값 읽기 (트랙바 생성 실패한 환경 대비 안전하게)
            try:
                canny_low = max(1, cv2.getTrackbarPos("Canny Low", window_name))
                canny_high = max(canny_low + 1, cv2.getTrackbarPos("Canny High", window_name))
                alpha_pct = cv2.getTrackbarPos("Blend Alpha %", window_name)
                max_d = max(100, cv2.getTrackbarPos("MaxDepth(mm)", window_name))
            except cv2.error:
                # 트랙바가 없으면 기본값 사용
                canny_low, canny_high = 30, 80
                alpha_pct = 50
                max_d = 10000
            alpha = alpha_pct / 100.0

            display = None
            h, w = (rgb.shape[:2] if rgb is not None else
                    (depth.shape[:2] if depth is not None else (480, 640)))

            if mode == "edge":
                # RGB 위에 Depth 엣지 오버레이 (정렬 확인 최고)
                base = rgb.copy() if rgb is not None else np.zeros((h, w, 3), np.uint8)
                if depth is not None:
                    d_gray = normalize_depth_for_display(depth, max_d)
                    edges = detect_edges(d_gray, canny_low, canny_high)
                    if edges is not None:
                        # 빨간색 엣지
                        edge_color = np.zeros_like(base)
                        edge_color[edges > 0] = (0, 0, 255)
                        display = cv2.addWeighted(base, 0.85, edge_color, 0.9, 0)
                        cv2.putText(display, "EDGE OVERLAY (Red=Depth edges) - Best for alignment check",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        display = base
                else:
                    display = base

            elif mode == "blend":
                if rgb is not None and depth is not None:
                    d_gray = normalize_depth_for_display(depth, max_d)
                    d_color = make_colormap_depth(d_gray)
                    # Depth를 RGB 크기에 맞춤 (간단 resize, 실제로는 intrinsics로 warp 추천)
                    if d_color.shape[:2] != rgb.shape[:2]:
                        d_color = cv2.resize(d_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                    display = cv2.addWeighted(rgb, 1 - alpha, d_color, alpha, 0)
                    cv2.putText(display, f"BLEND alpha={alpha:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    display = rgb if rgb is not None else make_colormap_depth(normalize_depth_for_display(depth, max_d))

            elif mode == "side":
                # 나란히
                left = rgb if rgb is not None else np.zeros((h, w, 3), np.uint8)
                if depth is not None:
                    d_gray = normalize_depth_for_display(depth, max_d)
                    d_color = make_colormap_depth(d_gray)
                    if d_color.shape[:2] != left.shape[:2]:
                        d_color = cv2.resize(d_color, (left.shape[1], left.shape[0]), cv2.INTER_NEAREST)
                    right = d_color
                else:
                    right = np.zeros_like(left)
                display = np.hstack([left, right])
                cv2.putText(display, "RGB | DEPTH (colormap)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elif mode == "depth":
                if depth is not None:
                    d_gray = normalize_depth_for_display(depth, max_d)
                    display = make_colormap_depth(d_gray)
                    cv2.putText(display, "DEPTH (JET)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    display = np.zeros((h, w, 3), np.uint8)

            if display is None:
                display = np.zeros((480, 640, 3), dtype=np.uint8)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('b'):
                mode = "blend"
            elif key == ord('s'):
                mode = "side"
            elif key == ord('e'):
                mode = "edge"
            elif key == ord('d'):
                mode = "depth"
            elif key == ord('c'):
                fname = f"img_align_check_{capture_count:04d}.png"
                cv2.imwrite(fname, display)
                print(f"📷 캡처 저장: {fname}")
                capture_count += 1

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("뷰어 종료.")


def main():
    print(DOCK_WARNING)

    # OpenCV Qt 백엔드에서 자주 나오는 harmless warning 설명
    print("※ 아래에 'QFontDatabase: Cannot find font directory .../cv2/qt/fonts' 경고가")
    print("  여러 줄 나올 수 있습니다. 이는 OpenCV가 Qt로 빌드됐는데 폰트가 없는 환경에서")
    print("  발생하는 알려진 harmless warning이며, 프로그램 동작에는 전혀 지장이 없습니다.\n")

    parser = argparse.ArgumentParser(description="TurtleBot4 OAK-D RGB-Depth Alignment Visual Tuner")
    parser.add_argument("--rgb", type=str, default=None, help="RGB image topic (e.g. /oakd/rgb/preview/image_raw)")
    parser.add_argument("--depth", type=str, default=None, help="Depth image topic (e.g. /oakd/stereo/depth)")
    args = parser.parse_args()

    print("[1/3] ros2 topic list 로 카메라 토픽 확인 중...")
    topic_output = run_ros2_topic_list()
    print("--- ros2 topic list (camera 관련 필터) ---")
    candidates = find_camera_topics(topic_output)
    if candidates:
        for c in candidates:
            print(f"  {c}")
    else:
        print("  (카메라 관련 토픽을 찾지 못했습니다. 전체 출력은 아래)")
        print(topic_output[:2000] if topic_output else "  (출력 없음 - ROS_DOMAIN_ID / discovery 확인 필요)")

    print("\n[2/3] rqt 사용법 안내")
    print(RQT_INSTRUCTIONS)

    print("[3/3] 샘플 RGBD config (my_oakd_rgbd.yaml 로 저장해서 사용)")
    print(SAMPLE_YAML)

    # 토픽 결정
    rgb_topic = args.rgb
    depth_topic = args.depth

    if not rgb_topic or not depth_topic:
        print("\n토픽을 직접 지정하지 않았습니다.")
        print("자동 후보 중에서 선택하거나, --rgb / --depth 로 지정하세요.")
        # 간단 자동 선택 시도 (사용자가 보기 쉽게)
        if candidates:
            rgb_cand = [c for c in candidates if "rgb" in c.lower() and "info" not in c.lower()]
            depth_cand = [c for c in candidates if "depth" in c.lower() or "stereo" in c.lower()]
            if rgb_cand:
                print(f"\n추천 RGB 후보: {rgb_cand[0]}")
            if depth_cand:
                print(f"추천 Depth 후보: {depth_cand[0]}")
        print("\n예시 실행:")
        print("  python3 tb4_cam_align_tuner.py --rgb /oakd/rgb/preview/image_raw --depth /oakd/stereo/depth")
        # 대화형 입력 유도
        try:
            rgb_topic = input("\nRGB topic 입력 (Enter=skip): ").strip() or None
            depth_topic = input("Depth topic 입력 (Enter=skip): ").strip() or None
        except EOFError:
            pass

    if rgb_topic and depth_topic:
        print(f"\n✅ 뷰어 시작: RGB={rgb_topic}  |  Depth={depth_topic}")
        print("   (rqt_reconfigure 는 별도 터미널에서 띄워두고 함께 사용하세요)\n")
        run_viewer(rgb_topic, depth_topic)
    else:
        print("\n토픽이 지정되지 않아 뷰어를 시작하지 않습니다.")
        print("ros2 topic list 로 정확한 토픽 이름을 확인한 후 --rgb, --depth 로 실행하세요.")
        print("예: python3 tb4_cam_align_tuner.py --rgb /turtlebot4_1/oakd/rgb/preview/image_raw --depth /turtlebot4_1/oakd/stereo/depth")


if __name__ == "__main__":
    main()
