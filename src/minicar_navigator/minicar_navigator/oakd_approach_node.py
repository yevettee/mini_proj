#!/usr/bin/env python3
"""
oakd_approach_node.py  —  Hybrid Nav2 + cmd_vel 접근/추종 노드

State Machine:
  IDLE       → (start_approach=True) → SEARCHING
  SEARCHING  → (감지, dist > target_distance)  → NAVIGATING
  SEARCHING  → (감지, dist ≤ target_distance)  → FOLLOWING
  NAVIGATING → (dist ≤ target_distance)        → FOLLOWING
  NAVIGATING → (차 소실)                       → SEARCHING
  FOLLOWING  → (dist > follow_hysteresis)      → NAVIGATING
  FOLLOWING  → (차 소실)                       → SEARCHING
  Any        → (stop signal)                   → IDLE

NAVIGATING : Nav2 NavigateToPose  — 차 위치에서 target_distance 앞 지점으로 경로 계획
FOLLOWING  : cmd_vel 비례 제어   — target_distance 거리 유지하며 차 추종

Parameters:
  model_path            (str)   : YOLO 모델 경로 (default: 'best.pt')
  confidence            (float) : 감지 신뢰도 (default: 0.5)
  target_class          (str)   : 감지 클래스 (default: 'car')
  target_distance       (float) : FOLLOWING 전환 / 유지 거리 m (default: 0.5)
  follow_hysteresis     (float) : FOLLOWING→NAVIGATING 재전환 거리 m (default: 1.5)
  nav_update_interval   (float) : Nav2 goal 재전송 주기 s (default: 1.0)
  max_linear_speed      (float) : cmd_vel 최대 직진 속도 (default: 0.2)
  max_angular_speed     (float) : cmd_vel 최대 회전 속도 (default: 0.5)
  kp_linear             (float) : 직진 비례 게인 (default: 0.3)
  kp_angular            (float) : 회전 비례 게인 (default: 0.8)
  search_angular_speed  (float) : 탐색 회전 속도 (default: 0.3)
  search_timeout_sec    (float) : 탐색 타임아웃 s (default: 30.0)
  rgb_topic             (str)   : OAK-D RGB compressed 토픽
  depth_topic           (str)   : OAK-D stereo depth 토픽
  camera_info_topic     (str)   : OAK-D camera_info 토픽
  cmd_vel_topic         (str)   : cmd_vel 퍼블리시 토픽
  action_server_name    (str)   : Nav2 액션 서버 이름
  map_frame             (str)   : map TF 프레임 이름 (default: 'map')
  approach_status_topic (str)   : 상태 퍼블리시 토픽
  detection_image_topic (str)   : 시각화 이미지 토픽
"""

import math
import time
import threading
from enum import Enum

import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
from rclpy.time import Time

from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Bool, String
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import Twist, PoseStamped, Quaternion, PointStamped
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # PointStamped 변환 타입 등록 (필수)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

from .utils import draw_box, resolve_model_path


class ApproachState(str, Enum):
    IDLE       = 'IDLE'
    SEARCHING  = 'SEARCHING'
    NAVIGATING = 'NAVIGATING'   # Nav2 — 차까지 거리 > target_distance
    FOLLOWING  = 'FOLLOWING'    # cmd_vel — 차까지 거리 ≤ target_distance, 실시간 추종


class OakdApproachNode(Node):
    def __init__(self):
        super().__init__('oakd_approach')

        # ------------------------------------------------------------------ #
        #  Parameters                                                          #
        # ------------------------------------------------------------------ #
        self.declare_parameter('model_path',           'best.pt')
        self.declare_parameter('confidence',           0.5)
        self.declare_parameter('target_class',         'car')
        self.declare_parameter('target_distance',      0.5)
        self.declare_parameter('follow_hysteresis',    1.5)
        self.declare_parameter('nav_update_interval',  1.0)
        self.declare_parameter('max_linear_speed',     0.2)
        self.declare_parameter('max_angular_speed',    0.5)
        self.declare_parameter('kp_linear',            0.3)
        self.declare_parameter('kp_angular',           0.8)
        self.declare_parameter('search_angular_speed', 0.3)
        self.declare_parameter('search_timeout_sec',   30.0)

        ns = self.get_namespace().rstrip('/')
        default_rgb   = f'{ns}/oakd/rgb/image_raw/compressed' if ns else '/oakd/rgb/image_raw/compressed'
        default_depth = f'{ns}/oakd/stereo/image_raw'         if ns else '/oakd/stereo/image_raw'
        default_info  = f'{ns}/oakd/rgb/camera_info'          if ns else '/oakd/rgb/camera_info'
        default_cmd   = f'{ns}/cmd_vel'                        if ns else '/robot6/cmd_vel'

        self.declare_parameter('rgb_topic',             default_rgb)
        self.declare_parameter('depth_topic',           default_depth)
        self.declare_parameter('camera_info_topic',     default_info)
        self.declare_parameter('cmd_vel_topic',         default_cmd)
        self.declare_parameter('action_server_name',    '/robot6/navigate_to_pose')
        self.declare_parameter('map_frame',             'map')
        self.declare_parameter('approach_status_topic', '/approach_status')
        self.declare_parameter('detection_image_topic', '/oakd_detection_image')

        self.model_path          = resolve_model_path(self.get_parameter('model_path').value)
        self.confidence          = self.get_parameter('confidence').value
        self.target_class        = self.get_parameter('target_class').value.lower()
        self.target_distance     = self.get_parameter('target_distance').value
        self.follow_hysteresis   = self.get_parameter('follow_hysteresis').value
        self.nav_update_interval = self.get_parameter('nav_update_interval').value
        self.max_linear_speed    = self.get_parameter('max_linear_speed').value
        self.max_angular_speed   = self.get_parameter('max_angular_speed').value
        self.kp_linear           = self.get_parameter('kp_linear').value
        self.kp_angular          = self.get_parameter('kp_angular').value
        self.search_angular_speed = self.get_parameter('search_angular_speed').value
        self.search_timeout_sec  = self.get_parameter('search_timeout_sec').value

        rgb_topic             = self.get_parameter('rgb_topic').value
        depth_topic           = self.get_parameter('depth_topic').value
        camera_info_topic     = self.get_parameter('camera_info_topic').value
        cmd_vel_topic         = self.get_parameter('cmd_vel_topic').value
        action_server_name    = self.get_parameter('action_server_name').value
        self.map_frame        = self.get_parameter('map_frame').value
        approach_status_topic = self.get_parameter('approach_status_topic').value
        detection_image_topic = self.get_parameter('detection_image_topic').value

        # ------------------------------------------------------------------ #
        #  Internal State                                                      #
        # ------------------------------------------------------------------ #
        self.state               = ApproachState.IDLE
        self._latest_rgb         = None
        self._latest_depth       = None
        self._rgb_stamp          = None
        self._depth_stamp        = None
        self._depth_max_age      = 0.5
        self._rgb_width          = 1
        self._rgb_height         = 1
        self._depth_width        = 1
        self._depth_height       = 1
        self._latest_center      = None
        self._latest_depth_m     = None
        self._latest_yolo_boxes  = []
        self._latest_cam_3d      = None   # (X, Y, Z, frame_id) 카메라 좌표계

        self.K            = None
        self.camera_frame = None

        self._search_start_time  = None
        self._last_nav_goal_time = None
        self._nav_goal_handle    = None
        self._lost_count         = 0
        self._lost_threshold     = 5

        self.logged_rgb_shape    = False
        self.logged_depth_shape  = False
        self.logged_intrinsics   = False

        self.lock          = threading.Lock()
        self._yolo_active  = False
        self._shutdown     = False

        # ------------------------------------------------------------------ #
        #  TF2  (런치 파일에서 /tf → /robot6/tf 리매핑 적용)                  #
        # ------------------------------------------------------------------ #
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------------------------------------------------------------------ #
        #  Nav2 Action Client                                                  #
        # ------------------------------------------------------------------ #
        self._cb_group = ReentrantCallbackGroup()
        self._nav_client = ActionClient(
            self, NavigateToPose, action_server_name,
            callback_group=self._cb_group,
        )

        # ------------------------------------------------------------------ #
        #  YOLO                                                                #
        # ------------------------------------------------------------------ #
        if not YOLO_AVAILABLE:
            self.get_logger().error('ultralytics not installed. Run: pip install ultralytics')
            raise SystemExit(1)
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model      = YOLO(self.model_path)
        self.classNames = self.model.names if hasattr(self.model, 'names') else {}
        self.get_logger().info(f'YOLO model loaded | classes: {list(self.classNames.values())}')

        self.bridge = CvBridge()

        # ------------------------------------------------------------------ #
        #  Publishers                                                          #
        # ------------------------------------------------------------------ #
        self.pub_status  = self.create_publisher(String, approach_status_topic, 10)
        self.pub_cmd_vel = self.create_publisher(Twist,  cmd_vel_topic, 10)
        self.pub_image   = self.create_publisher(Image,  detection_image_topic, 10)

        # ------------------------------------------------------------------ #
        #  Subscribers                                                         #
        # ------------------------------------------------------------------ #
        self.create_subscription(Bool,       '/start_approach',  self._on_start,       10)
        self.create_subscription(CameraInfo, camera_info_topic,  self._on_camera_info, 1)

        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=5,
        )
        self.create_subscription(CompressedImage, rgb_topic,   self._on_rgb,   cam_qos)
        self.create_subscription(Image,           depth_topic, self._on_depth, cam_qos)

        # ------------------------------------------------------------------ #
        #  Threads & Timers                                                    #
        # ------------------------------------------------------------------ #
        self._yolo_thread = threading.Thread(target=self._yolo_loop, daemon=True)
        self._yolo_thread.start()

        self.create_timer(0.05, self._tick)  # 20 Hz

        self._publish_status(ApproachState.IDLE)
        self.get_logger().info(
            f'OakdApproach (Hybrid) ready | '
            f'target="{self.target_class}" | '
            f'target_dist={self.target_distance}m | '
            f'hysteresis={self.follow_hysteresis}m | '
            f'rgb={rgb_topic} | depth={depth_topic}'
        )

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _on_camera_info(self, msg: CameraInfo):
        with self.lock:
            if self.K is None:
                self.K = np.array(msg.k).reshape(3, 3)
                if not self.logged_intrinsics:
                    self.get_logger().info(
                        f'Camera intrinsics: '
                        f'fx={self.K[0,0]:.2f}, fy={self.K[1,1]:.2f}, '
                        f'cx={self.K[0,2]:.2f}, cy={self.K[1,2]:.2f}'
                    )
                    self.logged_intrinsics = True

    def _on_start(self, msg: Bool):
        if msg.data and self.state == ApproachState.IDLE:
            self.state = ApproachState.SEARCHING
            self._search_start_time = self.get_clock().now()
            self._publish_status(ApproachState.SEARCHING)
            self.get_logger().info('Start → SEARCHING')
        elif not msg.data and self.state != ApproachState.IDLE:
            self.get_logger().info('Stop → IDLE')
            self._enter_idle()

    def _on_rgb(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if frame is not None and frame.size > 0:
                if not self.logged_rgb_shape:
                    self.get_logger().info(f'RGB shape: {frame.shape}')
                    self.logged_rgb_shape = True
                with self.lock:
                    self._latest_rgb  = frame
                    self._rgb_stamp   = time.monotonic()
                    self._rgb_height, self._rgb_width = frame.shape[:2]
        except Exception as e:
            self.get_logger().warn(f'RGB decode error: {e}')

    def _on_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth is not None and depth.size > 0:
                if not self.logged_depth_shape:
                    self.get_logger().info(f'Depth shape: {depth.shape}, dtype: {depth.dtype}')
                    self.logged_depth_shape = True
                with self.lock:
                    self._latest_depth  = depth
                    self._depth_stamp   = time.monotonic()
                    self._depth_height, self._depth_width = depth.shape[:2]
                    self.camera_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')

    # ------------------------------------------------------------------ #
    #  YOLO Background Thread                                             #
    # ------------------------------------------------------------------ #

    def _yolo_loop(self):
        while not self._shutdown:
            if self._yolo_active:
                time.sleep(0.01)
                continue

            with self.lock:
                rgb         = self._latest_rgb
                depth       = self._latest_depth
                rgb_stamp   = self._rgb_stamp
                depth_stamp = self._depth_stamp
                rgb_w    = self._rgb_width
                rgb_h    = self._rgb_height
                dep_w    = self._depth_width
                dep_h    = self._depth_height
                K        = self.K.copy() if self.K is not None else None
                frame_id = self.camera_frame

            if rgb is None:
                time.sleep(0.01)
                continue

            if rgb_stamp is not None and depth_stamp is not None:
                depth_age   = abs(rgb_stamp - depth_stamp)
                depth_valid = (depth is not None) and (depth_age <= self._depth_max_age)
            else:
                depth_valid = False

            self._yolo_active = True
            try:
                results = self.model(rgb, conf=self.confidence, verbose=False)
                center  = self._find_best_target(results)
                boxes   = self._extract_boxes(results)
                depth_m = None
                cam_3d  = None

                if center is not None:
                    if depth_valid:
                        depth_m = self._get_depth(depth, center[0], center[1],
                                                  rgb_w, rgb_h, dep_w, dep_h)
                    if depth_m is not None and K is not None and frame_id:
                        cam_3d = self._compute_cam_3d(
                            center[0], center[1], depth_m,
                            K, rgb_w, rgb_h, dep_w, dep_h, frame_id
                        )

                with self.lock:
                    self._latest_center     = center
                    self._latest_depth_m    = depth_m
                    self._latest_yolo_boxes = boxes
                    self._latest_cam_3d     = cam_3d
            finally:
                self._yolo_active = False

    # ------------------------------------------------------------------ #
    #  20Hz Tick                                                          #
    # ------------------------------------------------------------------ #

    def _tick(self):
        with self.lock:
            rgb     = self._latest_rgb
            rgb_w   = self._rgb_width
            center  = self._latest_center
            depth_m = self._latest_depth_m
            boxes   = self._latest_yolo_boxes
            cam_3d  = self._latest_cam_3d

        if rgb is None:
            return

        self._publish_annotated(rgb, boxes, center, depth_m)

        if self.state == ApproachState.SEARCHING:
            self._tick_searching(center, depth_m)
        elif self.state == ApproachState.NAVIGATING:
            self._tick_navigating(center, depth_m, cam_3d)
        elif self.state == ApproachState.FOLLOWING:
            self._tick_following(center, depth_m, rgb_w)

    # ------------------------------------------------------------------ #
    #  State Handlers                                                      #
    # ------------------------------------------------------------------ #

    def _tick_searching(self, center, depth_m):
        if center is not None and depth_m is not None:
            if depth_m <= self.target_distance:
                self.get_logger().info(f'Car found (close {depth_m:.2f}m) → FOLLOWING')
                self.state = ApproachState.FOLLOWING
                self._publish_status(ApproachState.FOLLOWING)
            else:
                self.get_logger().info(f'Car found (far {depth_m:.2f}m) → NAVIGATING')
                self.state = ApproachState.NAVIGATING
                self._last_nav_goal_time = None
                self._publish_status(ApproachState.NAVIGATING)
            return

        if center is not None and depth_m is None:
            self.get_logger().warn('Car detected but depth invalid — check OAK-D stereo calibration')

        elapsed = (self.get_clock().now() - self._search_start_time).nanoseconds / 1e9
        if elapsed > self.search_timeout_sec:
            self.get_logger().warn(f'Search timeout ({self.search_timeout_sec:.0f}s) → IDLE')
            self._enter_idle()
            return

        self._stop_movement()

    def _tick_navigating(self, center, depth_m, cam_3d):
        if center is None:
            self._lost_count += 1
            if self._lost_count >= self._lost_threshold:
                self.get_logger().warn(f'Car lost {self._lost_count} frames → SEARCHING')
                self._lost_count = 0
                self._cancel_nav_goal()
                self._stop_movement()
                self.state = ApproachState.SEARCHING
                self._search_start_time = self.get_clock().now()
                self._publish_status(ApproachState.SEARCHING)
            return
        self._lost_count = 0

        if depth_m is not None and depth_m <= self.target_distance:
            self.get_logger().info(f'Reached {depth_m:.2f}m → FOLLOWING')
            self._cancel_nav_goal()
            self._stop_movement()
            self.state = ApproachState.FOLLOWING
            self._publish_status(ApproachState.FOLLOWING)
            return

        now = self.get_clock().now()
        elapsed = (
            (now - self._last_nav_goal_time).nanoseconds / 1e9
            if self._last_nav_goal_time else float('inf')
        )
        if elapsed >= self.nav_update_interval and cam_3d is not None:
            self._send_nav_goal(cam_3d)
            self._last_nav_goal_time = now

    def _tick_following(self, center, depth_m, rgb_w):
        if center is None:
            self._lost_count += 1
            if self._lost_count >= self._lost_threshold:
                self.get_logger().warn(f'Car lost {self._lost_count} frames → SEARCHING')
                self._lost_count = 0
                self._stop_movement()
                self.state = ApproachState.SEARCHING
                self._search_start_time = self.get_clock().now()
                self._publish_status(ApproachState.SEARCHING)
            return
        self._lost_count = 0

        if depth_m is not None and depth_m > self.follow_hysteresis:
            self.get_logger().info(f'Car moved far ({depth_m:.2f}m > {self.follow_hysteresis}m) → NAVIGATING')
            self.state = ApproachState.NAVIGATING
            self._last_nav_goal_time = None
            self._publish_status(ApproachState.NAVIGATING)
            return

        dist_str = f'{depth_m:.2f}m' if depth_m is not None else 'no depth'
        self.get_logger().debug(f'FOLLOWING | dist={dist_str}')

        cx, _   = center
        error_x = (cx - rgb_w / 2.0) / (rgb_w / 2.0)  # -1 ~ +1
        twist   = Twist()

        if abs(error_x) > 0.25:
            twist.angular.z = float(np.clip(-self.kp_angular * error_x,
                                            -self.max_angular_speed, self.max_angular_speed))
        else:
            if depth_m is not None:
                dist_error = depth_m - self.target_distance
                if abs(dist_error) > 0.05:
                    twist.linear.x = float(np.clip(self.kp_linear * dist_error,
                                                   -self.max_linear_speed, self.max_linear_speed))
            if abs(error_x) > 0.1:
                twist.angular.z = float(np.clip(-self.kp_angular * error_x,
                                                -self.max_angular_speed, self.max_angular_speed))

        self.pub_cmd_vel.publish(twist)

    # ------------------------------------------------------------------ #
    #  Nav2                                                                #
    # ------------------------------------------------------------------ #

    def _send_nav_goal(self, cam_3d: tuple):
        """차의 카메라 3D 좌표를 map으로 변환 → target_distance 앞 지점에 Nav2 goal 전송."""
        if not self.tf_buffer.can_transform(self.map_frame, cam_3d[3], Time()):
            self.get_logger().warn(
                f'map frame "{self.map_frame}" not yet available — waiting for localization',
                throttle_duration_sec=2.0,
            )
            return

        car_map = self._cam_3d_to_map(cam_3d)
        if car_map is None:
            return
        car_x, car_y = car_map

        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, 'base_link', Time(), timeout=Duration(seconds=0.3)
            )
            robot_x = tf.transform.translation.x
            robot_y = tf.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f'TF base_link→map failed: {e}')
            return

        dx   = robot_x - car_x
        dy   = robot_y - car_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return

        goal_x = car_x + (dx / dist) * self.target_distance
        goal_y = car_y + (dy / dist) * self.target_distance

        yaw = math.atan2(car_y - goal_y, car_x - goal_x)
        qz  = math.sin(yaw / 2.0)
        qw  = math.cos(yaw / 2.0)

        pose = PoseStamped()
        pose.header.frame_id    = self.map_frame
        pose.header.stamp       = self.get_clock().now().to_msg()
        pose.pose.position.x    = goal_x
        pose.pose.position.y    = goal_y
        pose.pose.position.z    = 0.0
        pose.pose.orientation   = Quaternion(x=0.0, y=0.0, z=qz, w=qw)

        if not self._nav_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn('Nav2 action server not ready — skipping goal')
            return

        self._cancel_nav_goal()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        future = self._nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self._on_nav_goal_accepted)
        self.get_logger().info(
            f'Nav2 goal → ({goal_x:.2f}, {goal_y:.2f}), '
            f'yaw={math.degrees(yaw):.1f}° | car=({car_x:.2f},{car_y:.2f})'
        )

    def _on_nav_goal_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn('Nav2 goal rejected')
            return
        with self.lock:
            self._nav_goal_handle = handle

    def _cancel_nav_goal(self):
        with self.lock:
            handle = self._nav_goal_handle
            self._nav_goal_handle = None
        if handle is not None:
            handle.cancel_goal_async()

    # ------------------------------------------------------------------ #
    #  Geometry Helpers                                                    #
    # ------------------------------------------------------------------ #

    def _compute_cam_3d(self, px: int, py: int, depth_m: float,
                        K: np.ndarray,
                        rgb_w: int, rgb_h: int,
                        dep_w: int, dep_h: int,
                        frame_id: str) -> tuple:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X = (px - cx) * depth_m / fx
        Y = (py - cy) * depth_m / fy
        Z = depth_m
        return (X, Y, Z, frame_id)

    def _cam_3d_to_map(self, cam_3d: tuple):
        """카메라 좌표계 3D → map 좌표 (x, y). 실패 시 None 반환."""
        X, Y, Z, frame_id = cam_3d
        pt = PointStamped()
        pt.header.stamp    = Time().to_msg()
        pt.header.frame_id = frame_id
        pt.point.x = X
        pt.point.y = Y
        pt.point.z = Z
        try:
            pt_map = self.tf_buffer.transform(pt, self.map_frame, timeout=Duration(seconds=0.3))
            return (pt_map.point.x, pt_map.point.y)
        except Exception as e:
            self.get_logger().warn(f'TF camera→map failed: {e}',
                                   throttle_duration_sec=2.0)
            return None

    def _get_depth(self, depth, rgb_cx: int, rgb_cy: int,
                   rgb_w: int, rgb_h: int,
                   dep_w: int, dep_h: int) -> float | None:
        if depth is None or dep_w <= 1:
            return None

        dx = int(np.clip(rgb_cx * dep_w / rgb_w, 0, dep_w - 1))
        dy = int(np.clip(rgb_cy * dep_h / rgb_h, 0, dep_h - 1))

        patch = depth[
            max(0, dy - 2):min(dep_h, dy + 3),
            max(0, dx - 2):min(dep_w, dx + 3)
        ]
        valid = patch[patch > 0]
        if valid.size == 0:
            return None
        return float(np.median(valid)) / 1000.0

    def _find_best_target(self, yolo_results):
        best_conf, best_center = -1.0, None
        for result in yolo_results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.classNames.get(cls_id, f'class_{cls_id}').lower()
                if cls_name == self.target_class:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        best_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        return best_center

    def _extract_boxes(self, yolo_results) -> list:
        boxes = []
        for result in yolo_results:
            for box in result.boxes:
                cls_id   = int(box.cls[0])
                cls_name = self.classNames.get(cls_id, f'class_{cls_id}').lower()
                conf     = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, cls_name, conf, cls_name == self.target_class))
        return boxes

    # ------------------------------------------------------------------ #
    #  Visualization                                                       #
    # ------------------------------------------------------------------ #

    def _publish_annotated(self, frame, boxes, center, depth_m):
        annotated = frame.copy()

        for x1, y1, x2, y2, cls_name, conf, is_target in boxes:
            color = (0, 255, 0) if is_target else (0, 165, 255)
            draw_box(annotated, x1, y1, x2, y2, f'{cls_name} {conf:.2f}', color)

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

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def _enter_idle(self):
        self._cancel_nav_goal()
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
        self._cancel_nav_goal()
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
