"""
depth_checker.py - YOLOv8 기반 BBox 거리 측정 노드

OAK-D (TurtleBot4) 의 RGB + Depth 이미지를 받아 YOLO로 객체를 검출하고,
각 바운딩박스 영역의 깊이값(median/min/center)을 이용해 실시간 거리를 측정/시각화.

실행 예시:
  # 기본 (preview + stereo/depth)
  ros2 run rokey_pjt depth_checker

  # 네임스페이스/다른 토픽 사용하는 경우
  ros2 run rokey_pjt depth_checker --ros-args \
    -p rgb_topic:=/robot6/oakd/rgb/preview/image_raw \
    -p depth_topic:=/robot6/oakd/stereo/depth

  # 모델 경로 변경 (커스텀 학습 모델 사용 시)
  ros2 run rokey_pjt depth_checker --ros-args \
    -p model_path:=/home/woody/mini_proj/models/runs/YOLO_Tournament/Round1_v8_Nano/weights/best.pt \
    -p conf_threshold:=0.45 \
    -p inference_hz:=6.0
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import numpy as np
import cv2
from pathlib import Path
from typing import Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# ============================================================
# 기본 토픽 (이 프로젝트의 TurtleBot4 OAK-D 추천값)
# ============================================================
# 이 프로젝트에서 가장 자주 맞는 기본값 (네임스페이스 없는 경우)
DEFAULT_RGB_TOPIC = '/oakd/rgb/preview/image_raw'
DEFAULT_DEPTH_TOPIC = '/oakd/stereo/image_raw'      # ← /stereo/depth 보다 이게 더 흔함
DEFAULT_INFO_TOPIC = '/oakd/stereo/camera_info'

# 시각화 설정
NORMALIZE_DEPTH_RANGE = 5.0   # depth colormap 정규화 최대값 (m)
VIZ_WINDOW = 'YOLO Depth Checker (q: quit, s: save)'


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """depth 이미지를 미터 단위 float32로 변환 (uint16 mm / float32 m 모두 지원)"""
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    if depth.dtype == np.float32:
        return depth
    # 기타 (uint8 등) 대비
    return depth.astype(np.float32)


def _compute_bbox_distance(
    depth_m: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    method: str = 'median'
) -> Optional[float]:
    """
    바운딩박스 영역의 거리 계산 (노이즈에 강건한 방법 추천)
    method: 'median' | 'min' | 'center'
    """
    h, w = depth_m.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_m[y1:y2, x1:x2]
    # 유효 범위 필터 (0.05m ~ 20m)
    valid = roi[(roi > 0.05) & (roi < 20.0)]
    if valid.size < 3:
        return None

    if method == 'min':
        return float(np.min(valid))
    if method == 'center':
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        val = depth_m[cy, cx]
        return float(val) if 0.05 < val < 20.0 else None
    # default: median (가장 robust)
    return float(np.median(valid))


class DepthChecker(Node):
    def __init__(self):
        super().__init__('depth_checker')

        # ---------------- Parameters ----------------
        self.declare_parameter('rgb_topic', DEFAULT_RGB_TOPIC)
        self.declare_parameter('depth_topic', DEFAULT_DEPTH_TOPIC)
        self.declare_parameter('camera_info_topic', DEFAULT_INFO_TOPIC)
        # 기본 모델: 사용자가 학습한 best.pt (Round1_v8_Nano) — models/ 아래로 정리된 위치
        self.declare_parameter(
            'model_path',
            str(Path.home() / 'mini_proj' / 'models' / 'final_v11n_augment' / 'best.pt')
        )
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('inference_hz', 6.0)           # YOLO 추론 주기 (너무 높이면 CPU/GPU 부하)
        self.declare_parameter('depth_method', 'median')      # median | min | center
        self.declare_parameter('show_depth_window', True)

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.model_path = self.get_parameter('model_path').value
        self.conf_th = float(self.get_parameter('conf_threshold').value)
        self.depth_method = self.get_parameter('depth_method').value
        self.show_depth_win = bool(self.get_parameter('show_depth_window').value)

        # ---------------- State (초기화) ----------------
        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None
        self.should_exit = False

        self.latest_rgb: Optional[np.ndarray] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_viz: Optional[np.ndarray] = None

        self.detections = []
        self.got_rgb = False
        self.got_depth = False
        self.model = None          # YOLO 모델 (아직 로드 전)
        self.yolo_ok = False       # YOLO 사용 가능 여부

        # ---------------- YOLO 모델 로딩 ----------------
        if YOLO is None:
            self.get_logger().error("ultralytics not installed. pip install ultralytics")
            self.model = None
        else:
            model_path = Path(self.model_path).expanduser()
            if not model_path.exists():
                self.get_logger().warn(f"Model not found at {model_path}. YOLO will fail to load.")
            else:
                self.get_logger().info(f"Loading YOLO model: {model_path}")
                self.model = YOLO(str(model_path))
                try:
                    import torch
                    device = 0 if torch.cuda.is_available() else 'cpu'
                    self.model.to(device)
                    self.get_logger().info(f"YOLO device: {device}")
                except Exception:
                    pass

                # 커스텀 모델일 경우 어떤 클래스를 인식하는지 명확히 출력
                self.get_logger().info(f"Model classes this model can detect: {list(self.model.names.values())}")

        self.yolo_ok = self.model is not None

        # ---------------- 로그 (이제 안전하게 출력 가능) ----------------
        self.get_logger().info(
            f"Topics -> RGB: {self.rgb_topic} | Depth: {self.depth_topic}"
        )
        self.get_logger().info(
            f"YOLO model status: {'OK' if self.yolo_ok else 'NOT LOADED (will show raw RGB only)'}"
        )

        # ---------------- Subscriptions ----------------
        self.rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self._rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, self.depth_topic, self._depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.info_topic, self._info_callback, 10
        )

        # ---------------- Timers ----------------
        inf_hz = max(1.0, float(self.get_parameter('inference_hz').value))
        self.create_timer(1.0 / inf_hz, self._inference_timer)
        self.create_timer(0.033, self._display_timer)  # ~30 Hz display

        # ---------------- OpenCV Window ----------------
        cv2.namedWindow(VIZ_WINDOW)
        cv2.setMouseCallback(VIZ_WINDOW, self._mouse_callback)

        # 첫 화면을 검은 화면으로 두지 않기 위해 즉시 placeholder 표시
        self._show_placeholder()

        self.get_logger().info(
            "DepthChecker(YOLO) ready. Press 'q' to quit, 's' to save snapshot."
        )
        self.get_logger().info(
            "TIP: ros2 topic list | grep -E 'oakd|rgb|depth' 로 실제 토픽을 확인하세요."
        )

    def _show_placeholder(self):
        """초기/대기 상태에서 검은 화면 대신 명확한 안내 화면을 보여줌"""
        h, w = 520, 680
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (25, 25, 25)

        yolo_status = "LOADED" if self.yolo_ok else "NOT LOADED (raw RGB only)"

        lines = [
            "YOLO Depth Checker",
            "",
            f"RGB Topic : {self.rgb_topic}",
            f"Depth Topic: {self.depth_topic}",
            f"YOLO      : {yolo_status}",
            "",
            "Waiting for camera images...",
            "",
            "Check actual topics:",
            "  ros2 topic list | grep -E 'oakd|rgb|depth|stereo'",
            "",
            "Namespace example (robot6):",
            "  ros2 run rokey_pjt depth_checker --ros-args \\",
            "    -p rgb_topic:=/robot6/oakd/rgb/preview/image_raw \\",
            "    -p depth_topic:=/robot6/oakd/stereo/image_raw",
            "",
            "Press 'q' to quit"
        ]

        y = 35
        for i, line in enumerate(lines):
            if i == 0:
                color = (0, 255, 255)
                scale = 0.85
            elif "Waiting" in line:
                color = (0, 255, 80)
                scale = 0.7
            else:
                color = (210, 210, 210)
                scale = 0.58
            cv2.putText(img, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
            y += 26

        cv2.imshow(VIZ_WINDOW, img)
        cv2.waitKey(1)

    # ---------------- Callbacks ----------------
    def _info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo: fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f} "
                f"cx={self.K[0,2]:.1f} cy={self.K[1,2]:.1f}"
            )

    def _rgb_callback(self, msg: Image):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if not self.got_rgb:
                self.got_rgb = True
                h, w = self.latest_rgb.shape[:2]
                self.get_logger().info(f"First RGB frame received from {self.rgb_topic}  →  size: {w}x{h}")
        except Exception as e:
            self.get_logger().warn(f"RGB convert error: {e}")

    def _depth_callback(self, msg: Image):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_m = _depth_to_meters(depth_raw)
            if not self.got_depth:
                self.got_depth = True
                h, w = self.latest_depth_m.shape[:2]
                self.get_logger().info(f"First Depth frame received from {self.depth_topic}  →  size: {w}x{h}")
        except Exception as e:
            self.get_logger().warn(f"Depth convert error: {e}")

    # ---------------- Inference (timer) ----------------
    def _inference_timer(self):
        if self.latest_rgb is None or self.latest_depth_m is None:
            return

        rgb = self.latest_rgb
        depth_m = self.latest_depth_m

        # === 중요한 진단: RGB와 Depth 해상도가 다르면 거리 측정이 완전히 틀어질 수 있음 ===
        rgb_h, rgb_w = rgb.shape[:2]
        depth_h, depth_w = depth_m.shape[:2]
        if (rgb_h, rgb_w) != (depth_h, depth_w):
            if not hasattr(self, '_shape_warned') or not self._shape_warned:
                self.get_logger().warn(
                    f"!!! RESOLUTION MISMATCH !!!  RGB={rgb_w}x{rgb_h}  vs  Depth={depth_w}x{depth_h}\n"
                    f"   → YOLO bbox 좌표가 Depth 이미지 픽셀과 맞지 않아서 거리가 완전히 틀릴 수 있습니다.\n"
                    f"   → oakd 설정에서 RGB와 Stereo/Depth 해상도를 동일하게 맞추거나 (i_width, i_height),\n"
                    f"     또는 stereo.i_align_depth=true + i_board_socket_id=0 설정을 확인하세요."
                )
                self._shape_warned = True

        # YOLO 모델이 없으면 그냥 RGB만 보여줌 (거리 측정은 안 됨)
        if self.model is None:
            self.latest_viz = rgb.copy()
            self.detections = []
            return

        # YOLO inference
        try:
            results = self.model(rgb, verbose=False, conf=self.conf_th)[0]
        except Exception as e:
            self.get_logger().warn(f"YOLO inference failed: {e}")
            self.latest_viz = rgb.copy()
            return

        new_dets = []
        viz = rgb.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                cls_name = results.names.get(cls_id, str(cls_id))

                x1, y1, x2, y2 = xyxy
                dist = _compute_bbox_distance(depth_m, x1, y1, x2, y2, self.depth_method)

                new_dets.append((xyxy, conf, cls_name, dist))

                color = (0, 200, 255) if dist is not None else (80, 80, 80)
                cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

                label = f"{cls_name} {conf:.2f}"
                if dist is not None:
                    label += f"  {dist:.2f}m"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(viz, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
                cv2.putText(viz, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        self.detections = new_dets
        self.latest_viz = viz

    # ---------------- Display (fast timer) ----------------
    def _display_timer(self):
        if self.should_exit:
            return

        # 항상 무언가 표시 (검은 화면 방지)
        if self.latest_viz is not None:
            display = self.latest_viz.copy()
        elif self.latest_rgb is not None:
            display = self.latest_rgb.copy()
            # YOLO 아직 동작 안 함 (depth 대기 중 등)
            status = "YOLO waiting for depth..." if self.got_depth else "Waiting for depth image..."
            cv2.putText(display, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
            cv2.putText(display, f"Detections: {len(self.detections)}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        else:
            # 아직 어떤 영상도 도착하지 않음 → placeholder 재표시 (또는 새로 생성)
            display = np.zeros((480, 640, 3), dtype=np.uint8)
            display[:] = (25, 25, 25)
            cv2.putText(display, "YOLO Depth Checker", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "No image received yet", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            cv2.putText(display, f"RGB  : {self.rgb_topic}", (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            cv2.putText(display, f"Depth: {self.depth_topic}", (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

            cv2.putText(display, "Waiting for camera topics...", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

            cv2.putText(display, "ros2 topic list | grep oakd", (20, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(display, "Press 'q' to quit", (20, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # depth window (optional)
        if self.show_depth_win and self.latest_depth_m is not None:
            dvis = np.clip(self.latest_depth_m, 0, NORMALIZE_DEPTH_RANGE)
            dvis = (dvis / NORMALIZE_DEPTH_RANGE * 255).astype(np.uint8)
            dcol = cv2.applyColorMap(dvis, cv2.COLORMAP_JET)

            for xyxy, conf, cls_name, dist in self.detections:
                x1, y1, x2, y2 = xyxy
                cv2.rectangle(dcol, (x1, y1), (x2, y2), (255, 255, 255), 1)
                if dist is not None:
                    cv2.putText(dcol, f"{dist:.2f}m", (x1, y1 + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Depth (YOLO overlay)', dcol)

        cv2.imshow(VIZ_WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.should_exit = True
        elif key == ord('s') and display is not None:
            fname = f"depth_yolo_{len(self.detections)}dets.png"
            cv2.imwrite(fname, display)
            self.get_logger().info(f"Saved snapshot: {fname}")

    # ---------------- Mouse (manual point query) ----------------
    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.latest_depth_m is None:
            return

        h, w = self.latest_depth_m.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            d = self.latest_depth_m[y, x]
            if 0.05 < d < 20.0:
                self.get_logger().info(f"Manual click @ ({x},{y}) -> {d:.2f} m")
            else:
                self.get_logger().info(f"Manual click @ ({x},{y}) -> invalid depth")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = DepthChecker()

    try:
        while rclpy.ok() and not node.should_exit:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

