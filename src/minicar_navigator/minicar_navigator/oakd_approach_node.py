#!/usr/bin/env python3
"""
oakd_approach_node.py
---------------------
OAK-D 카메라로 미니카를 탐색하고 접근하는 노드.

Phase 3: OAK-D RGB + YOLO → 미니카 탐색 (없으면 제자리 회전)
Phase 4: OAK-D Stereo Depth(16UC1, mm) → 거리 계산 → 접근 → 정지

State Machine:
  IDLE → (start_approach=True) → SEARCHING
  SEARCHING → (YOLO 감지) → APPROACHING
  SEARCHING → (타임아웃) → IDLE
  APPROACHING → (depth <= target_distance) → ARRIVED
  APPROACHING → (미니카 소실) → SEARCHING

Subscribes:
  /start_approach          (std_msgs/Bool)               - True: 시작, False: 중단
  <rgb_topic>              (sensor_msgs/CompressedImage) - OAK-D RGB compressed
  <depth_topic>            (sensor_msgs/CompressedImage) - OAK-D Stereo compressedDepth

Publishes:
  <approach_status_topic>  (std_msgs/String)    - IDLE/SEARCHING/APPROACHING/ARRIVED
  <cmd_vel_topic>          (geometry_msgs/Twist)
  <detection_image_topic>  (sensor_msgs/Image)  - 시각화

Parameters:
  model_path            (str)   : YOLO 모델 경로 (default: 'best.pt')
  confidence            (float) : 감지 신뢰도 (default: 0.5)
  target_class          (str)   : 감지 클래스 (default: 'car')
  target_distance       (float) : 목표 정지 거리 m (default: 0.5)
  max_linear_speed      (float) : 최대 직진 속도 m/s (default: 0.2)
  max_angular_speed     (float) : 최대 회전 속도 rad/s (default: 0.5)
  kp_linear             (float) : 직진 비례 게인 (default: 0.3)
  kp_angular            (float) : 회전 비례 게인 (default: 0.8)
  search_angular_speed  (float) : 탐색 회전 속도 rad/s (default: 0.3)
  search_timeout_sec    (float) : 탐색 타임아웃 초 (default: 30.0)
  rgb_topic             (str)   : RGB compressed 토픽
  depth_topic           (str)   : Depth compressedDepth 토픽
  cmd_vel_topic         (str)   : cmd_vel 토픽
  approach_status_topic (str)   : 상태 퍼블리시 토픽
  detection_image_topic (str)   : 시각화 이미지 토픽
"""

import threading
from enum import Enum

import cv2
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .utils import draw_box, resolve_model_path


class ApproachState(str, Enum):
    """OAK-D 접근 노드 내부 상태."""
    IDLE        = 'IDLE'
    SEARCHING   = 'SEARCHING'
    APPROACHING = 'APPROACHING'
    ARRIVED     = 'ARRIVED'


class OakdApproachNode(Node):
    def __init__(self):
        super().__init__('oakd_approach')

        # --- Parameters ---
        self.declare_parameter('model_path',           'best.pt')
        self.declare_parameter('confidence',           0.5)
        self.declare_parameter('target_class',         'car')
        self.declare_parameter('target_distance',      0.5)
        self.declare_parameter('max_linear_speed',     0.2)
        self.declare_parameter('max_angular_speed',    0.5)
        self.declare_parameter('kp_linear',            0.3)
        self.declare_parameter('kp_angular',           0.8)
        self.declare_parameter('search_angular_speed', 0.3)
        self.declare_parameter('search_timeout_sec',   30.0)

        ns = self.get_namespace().rstrip('/')
        default_rgb   = f'{ns}/oakd/rgb/image_raw/compressed' if ns else '/oakd/rgb/image_raw/compressed'
        default_depth = f'{ns}/oakd/stereo/image_raw'        if ns else '/oakd/stereo/image_raw'
        default_cmd   = f'{ns}/cmd_vel'                               if ns else '/robot6/cmd_vel'

        self.declare_parameter('rgb_topic',             default_rgb)
        self.declare_parameter('depth_topic',           default_depth)
        self.declare_parameter('cmd_vel_topic',         default_cmd)
        self.declare_parameter('approach_status_topic', '/approach_status')
        self.declare_parameter('detection_image_topic', '/oakd_detection_image')

        self.model_path           = resolve_model_path(self.get_parameter('model_path').value)
        self.confidence           = self.get_parameter('confidence').value
        self.target_class         = self.get_parameter('target_class').value.lower()
        self.target_distance      = self.get_parameter('target_distance').value
        self.max_linear_speed     = self.get_parameter('max_linear_speed').value
        self.max_angular_speed    = self.get_parameter('max_angular_speed').value
        self.kp_linear            = self.get_parameter('kp_linear').value
        self.kp_angular           = self.get_parameter('kp_angular').value
        self.search_angular_speed = self.get_parameter('search_angular_speed').value
        self.search_timeout_sec   = self.get_parameter('search_timeout_sec').value
        rgb_topic                 = self.get_parameter('rgb_topic').value
        depth_topic               = self.get_parameter('depth_topic').value
        cmd_vel_topic             = self.get_parameter('cmd_vel_topic').value
        approach_status_topic     = self.get_parameter('approach_status_topic').value
        detection_image_topic     = self.get_parameter('detection_image_topic').value

        # --- Internal State ---
        self.state              = ApproachState.IDLE
        self._latest_rgb        = None
        self._latest_depth      = None
        self._rgb_width         = 1
        self._rgb_height        = 1
        self._depth_width       = 1
        self._depth_height      = 1
        self._search_start_time = None
        self._depth_fail_count  = 0

        # YOLO 스레드가 채우는 최신 결과 — _tick이 읽어서 이미지 퍼블리시 + 제어에 사용
        self._latest_center     = None
        self._latest_depth_m    = None
        self._latest_yolo_boxes = []   # _publish_annotated용

        # One-time debug flags
        self.logged_rgb_shape   = False
        self.logged_depth_shape = False

        # Thread safety
        self.lock          = threading.Lock()
        self._yolo_active  = False  # YOLO 스레드 중복 실행 방지

        # --- YOLO ---
        if not YOLO_AVAILABLE:
            self.get_logger().error('ultralytics not installed. Run: pip install ultralytics')
            raise SystemExit(1)
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model      = YOLO(self.model_path)
        self.classNames = self.model.names if hasattr(self.model, 'names') else {}
        self.get_logger().info(f'YOLO model loaded | classes: {list(self.classNames.values())}')

        self.bridge = CvBridge()

        # --- Publishers ---
        self.pub_status  = self.create_publisher(String, approach_status_topic, 10)
        self.pub_cmd_vel = self.create_publisher(Twist,  cmd_vel_topic, 10)
        self.pub_image   = self.create_publisher(Image,  detection_image_topic, 10)

        # --- Subscribers ---
        self.create_subscription(Bool, '/start_approach', self._on_start, 10)

        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self._sub_rgb   = message_filters.Subscriber(self, CompressedImage, rgb_topic,   qos_profile=cam_qos)
        self._sub_depth = message_filters.Subscriber(self, Image,           depth_topic, qos_profile=cam_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._sub_rgb, self._sub_depth],
            queue_size=5,
            slop=0.1,   # 두 프레임의 타임스탬프 차이 허용 오차 (초)
        )
        self._sync.registerCallback(self._on_synced)

        self._shutdown = False
        self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
        self._yolo_thread.start()

        self.create_timer(0.05, self._tick)  # 20Hz — 이미지 퍼블리시 + 제어

        self._publish_status(ApproachState.IDLE)
        self.get_logger().info(
            f'OakdApproach ready | target="{self.target_class}" | '
            f'stop_dist={self.target_distance}m | '
            f'rgb={rgb_topic} | depth={depth_topic} | cmd_vel={cmd_vel_topic}'
        )

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _on_start(self, msg: Bool):
        if msg.data and self.state == ApproachState.IDLE:
            self.state = ApproachState.SEARCHING
            self._search_start_time = self.get_clock().now()
            self._publish_status(ApproachState.SEARCHING)
            self.get_logger().info('Start received → SEARCHING')
        elif not msg.data and self.state != ApproachState.IDLE:
            self.get_logger().info('Stop received → IDLE')
            self._enter_idle()

    def _on_synced(self, rgb_msg: CompressedImage, depth_msg: Image):
        """RGB + Depth 타임스탬프가 slop 이내로 매칭됐을 때만 호출됨."""
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            if frame is not None and frame.size > 0:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f'RGB image shape: {frame.shape}')
                    self.logged_rgb_shape = True
                with self.lock:
                    self._latest_rgb    = frame
                    self._rgb_height, self._rgb_width = frame.shape[:2]
        except Exception as e:
            self.get_logger().warn(f'RGB decode error: {e}')
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0:
                if not self.logged_depth_shape:
                    self.get_logger().info(f'Depth image shape: {depth.shape}, dtype: {depth.dtype}')
                    self.logged_depth_shape = True
                with self.lock:
                    self._latest_depth  = depth
                    self._depth_height, self._depth_width = depth.shape[:2]
            else:
                with self.lock:
                    self._latest_depth = None
        except Exception as e:
            with self.lock:
                self._latest_depth = None
            self.get_logger().warn(f'Depth decode error: {e}')

    # ------------------------------------------------------------------ #
    #  Tick (10Hz)                                                         #
    # ------------------------------------------------------------------ #

    def _yolo_loop(self):
        """YOLO 추론 전용 스레드 — 느린 추론이 이미지 퍼블리시를 막지 않도록 분리."""
        while not self._shutdown:
            if self._yolo_active:
                import time; time.sleep(0.01)
                continue
            with self.lock:
                rgb   = self._latest_rgb
                depth = self._latest_depth
                rgb_w = self._rgb_width
                rgb_h = self._rgb_height
            if rgb is None:
                import time; time.sleep(0.01)
                continue
            self._yolo_active = True
            try:
                results  = self.model(rgb, conf=self.confidence, verbose=False)
                center   = self._find_best_target(results)
                depth_m  = self._get_depth(depth, *center, rgb_w, rgb_h) if center else None
                boxes    = self._extract_boxes(results)
                with self.lock:
                    self._latest_center     = center
                    self._latest_depth_m    = depth_m
                    self._latest_yolo_boxes = boxes
            finally:
                self._yolo_active = False

    def _tick(self):
        """20Hz 타이머 — 이미지 퍼블리시 + 제어. YOLO 추론을 기다리지 않음."""
        with self.lock:
            rgb    = self._latest_rgb
            rgb_w  = self._rgb_width
            center = self._latest_center
            depth_m = self._latest_depth_m
            boxes  = self._latest_yolo_boxes

        if rgb is None:
            return

        self._publish_annotated(rgb, boxes, center, depth_m)

        if self.state == ApproachState.SEARCHING:
            self._tick_searching(center)
        elif self.state == ApproachState.APPROACHING:
            self._tick_approaching(center, depth_m, rgb_w)

    def _tick_searching(self, center: tuple[int, int] | None):
        """미니카가 감지될 때까지 제자리에서 대기."""
        if center is not None:
            self.state = ApproachState.APPROACHING
            self._depth_fail_count = 0
            self._publish_status(ApproachState.APPROACHING)
            self.get_logger().info('Minicar found → APPROACHING')
            return

        elapsed = (self.get_clock().now() - self._search_start_time).nanoseconds / 1e9
        if elapsed > self.search_timeout_sec:
            self.get_logger().warn(f'Search timeout ({self.search_timeout_sec:.0f}s) → IDLE')
            self._enter_idle()
            return

    def _tick_approaching(self, center: tuple[int, int] | None,
                          depth_m: float | None, rgb_w: int):
        """미니카 방향으로 회전하면서 depth 기반으로 직진."""
        if center is None:
            self.get_logger().warn('Minicar lost → back to SEARCHING')
            self.state = ApproachState.SEARCHING
            self._publish_status(ApproachState.SEARCHING)
            self._stop_movement()
            return

        if depth_m is None:
            self._depth_fail_count += 1
            if self._depth_fail_count >= 10:
                self.get_logger().warn('Depth unavailable (1s) → back to SEARCHING')
                self._depth_fail_count = 0
                self.state = ApproachState.SEARCHING
                self._publish_status(ApproachState.SEARCHING)
                self._stop_movement()
            return

        self._depth_fail_count = 0
        self.get_logger().info(f'Distance: {depth_m:.3f}m | center_x: {center[0]} / {rgb_w}')

        if depth_m <= self.target_distance:
            self.get_logger().info(f'ARRIVED (dist={depth_m:.2f}m)')
            self._stop_movement()
            self.state = ApproachState.ARRIVED
            self._publish_status(ApproachState.ARRIVED)
            return

        # 2단계 제어: 정렬 후 접근
        cx, _ = center
        error_x = (cx - rgb_w / 2.0) / (rgb_w / 2.0)
        twist = Twist()

        if abs(error_x) > 0.25:
            # 차가 화면 중앙에서 25% 이상 벗어남 → 회전만, 전진 없음
            twist.angular.z = max(-self.max_angular_speed,
                                   min(self.max_angular_speed, -self.kp_angular * error_x))
        else:
            # 차가 중앙 근처 → 전진 + 미세 각도 보정
            twist.linear.x = min(self.max_linear_speed,
                                  self.kp_linear * (depth_m - self.target_distance))
            if abs(error_x) > 0.1:
                twist.angular.z = max(-self.max_angular_speed,
                                       min(self.max_angular_speed, -self.kp_angular * error_x))

        self.pub_cmd_vel.publish(twist)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _find_best_target(self, yolo_results) -> tuple[int, int] | None:
        """YOLO 결과에서 confidence 가장 높은 target의 중심 좌표를 반환."""
        best_conf   = -1.0
        best_center = None

        for result in yolo_results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.classNames.get(cls_id, f'class_{cls_id}').lower()
                if cls_name == self.target_class:
                    conf_val = float(box.conf[0])
                    if conf_val > best_conf:
                        best_conf = conf_val
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        best_center = ((x1 + x2) // 2, (y1 + y2) // 2)

        return best_center

    def _extract_boxes(self, yolo_results) -> list:
        """YOLO 결과에서 (x1,y1,x2,y2,cls_name,conf,is_target) 리스트 추출."""
        boxes = []
        for result in yolo_results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.classNames.get(cls_id, f'class_{cls_id}').lower()
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, cls_name, conf_val, cls_name == self.target_class))
        return boxes

    def _publish_annotated(self, frame, boxes: list,
                           center: tuple[int, int] | None,
                           depth_m: float | None) -> None:
        """최신 YOLO boxes로 이미지 시각화 후 퍼블리시."""
        annotated = frame.copy()

        for x1, y1, x2, y2, cls_name, conf_val, is_target in boxes:
            color = (0, 255, 0) if is_target else (0, 165, 255)
            draw_box(annotated, x1, y1, x2, y2, f'{cls_name} {conf_val:.2f}', color)

        if center is not None:
            cx, cy    = center
            depth_str = f'{depth_m:.2f}m' if depth_m is not None else '--'
            cv2.putText(annotated, f'dist: {depth_str}', (cx - 30, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        label = f'[{self.state.value}]' + (' DETECTED' if center else ' searching...')
        color = (0, 255, 0) if center else (0, 165, 255)
        cv2.putText(annotated, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_image.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f'Image publish error: {e}')

    def _get_depth(self, depth, rgb_cx: int, rgb_cy: int,
                   rgb_w: int, rgb_h: int) -> float | None:
        """RGB 픽셀 기준 depth(m) 반환. 3x3 패치 평균으로 노이즈 감소."""
        with self.lock:
            dep_w = self._depth_width
            dep_h = self._depth_height

        if depth is None or dep_w <= 1:
            return None

        dx = int(rgb_cx / rgb_w * dep_w)
        dy = int(rgb_cy / rgb_h * dep_h)
        dx = max(0, min(dx, dep_w - 1))
        dy = max(0, min(dy, dep_h - 1))

        patch = depth[
            max(0, dy - 1):min(dep_h, dy + 2),
            max(0, dx - 1):min(dep_w, dx + 2)
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None

        raw_m = float(np.mean(valid)) / 1000.0  # mm → m
        # 선형 보정: real = 0.8426 * measured + 0.1793 (실측 캘리브레이션)
        return 0.8426 * raw_m + 0.1793

    def _enter_idle(self):
        self._stop_movement()
        self.state = ApproachState.IDLE
        self._publish_status(ApproachState.IDLE)

    def _stop_movement(self):
        self.pub_cmd_vel.publish(Twist())

    def _publish_status(self, status: ApproachState):
        msg = String()
        msg.data = status.value
        self.pub_status.publish(msg)
        self.get_logger().info(f'ApproachStatus → {status.value}')

    def destroy_node(self):
        self._shutdown = True
        self._yolo_thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OakdApproachNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
