"""
depth_checker_mouse.py - 단순 마우스 클릭 거리 측정 노드 (YOLO 없음)

가벼운 용도 또는 YOLO 없이 빠르게 테스트할 때 사용.
ROS 파라미터로 토픽을 쉽게 변경할 수 있음.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import numpy as np
import cv2
from typing import Optional


DEFAULT_DEPTH_TOPIC = '/oakd/stereo/image_raw'   # /stereo/depth 보다 image_raw이 더 흔함
DEFAULT_INFO_TOPIC = '/oakd/stereo/camera_info'

WINDOW_NAME = 'Depth (Click to measure) - q:quit'
NORMALIZE_DEPTH_RANGE = 5.0


def _depth_to_meters(depth: np.ndarray) -> np.ndarray:
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0
    if depth.dtype == np.float32:
        return depth
    return depth.astype(np.float32)


class DepthCheckerMouse(Node):
    def __init__(self):
        super().__init__('depth_checker_mouse')

        self.declare_parameter('depth_topic', DEFAULT_DEPTH_TOPIC)
        self.declare_parameter('camera_info_topic', DEFAULT_INFO_TOPIC)
        self.declare_parameter('normalize_range', NORMALIZE_DEPTH_RANGE)

        self.depth_topic = self.get_parameter('depth_topic').value
        self.info_topic = self.get_parameter('camera_info_topic').value
        self.norm_range = float(self.get_parameter('normalize_range').value)

        self.get_logger().info(f"Depth topic: {self.depth_topic}")

        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None
        self.should_exit = False

        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_viz: Optional[np.ndarray] = None

        self.create_subscription(Image, self.depth_topic, self._depth_callback, 10)
        self.create_subscription(CameraInfo, self.info_topic, self._info_callback, 10)

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._mouse_callback)

        # 가벼운 display timer
        self.create_timer(0.033, self._display_timer)

        self.get_logger().info("DepthCheckerMouse ready. Click on image to log distance.")

    def _info_callback(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.get_logger().info(
                f"CameraInfo: fx={self.K[0,0]:.1f} cx={self.K[0,2]:.1f}"
            )

    def _depth_callback(self, msg: Image):
        try:
            raw = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            self.latest_depth_m = _depth_to_meters(raw)
        except Exception as e:
            self.get_logger().warn(f"Depth error: {e}")

    def _display_timer(self):
        if self.should_exit or self.latest_depth_m is None:
            return

        d = np.clip(self.latest_depth_m, 0, self.norm_range)
        d8 = (d / self.norm_range * 255).astype(np.uint8)
        viz = cv2.applyColorMap(d8, cv2.COLORMAP_JET)

        h, w = viz.shape[:2]
        # center cross (from K or image center)
        cx = int(self.K[0, 2]) if self.K is not None else w // 2
        cy = int(self.K[1, 2]) if self.K is not None else h // 2
        cv2.circle(viz, (cx, cy), 5, (0, 0, 0), -1)
        cv2.line(viz, (0, cy), (w, cy), (0, 0, 0), 1)
        cv2.line(viz, (cx, 0), (cx, h), (0, 0, 0), 1)

        cv2.putText(viz, "Click anywhere for distance", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        self.latest_viz = viz
        cv2.imshow(WINDOW_NAME, viz)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.should_exit = True

    def _mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.latest_depth_m is None:
            return
        h, w = self.latest_depth_m.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            d = self.latest_depth_m[y, x]
            if 0.05 < d < 20.0:
                self.get_logger().info(f"Clicked ({x},{y}) -> {d:.2f} m")
            else:
                self.get_logger().info(f"Clicked ({x},{y}) -> invalid/out-of-range")

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main():
    rclpy.init()
    node = DepthCheckerMouse()

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

