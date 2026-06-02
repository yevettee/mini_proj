#!/usr/bin/env python3
"""
minicar_nav_manager.py
----------------------
YOLO 감지 결과를 받아 Nav2 네비게이션과 OAK-D 접근을 조율하는 매니저 노드.

State Machine:
  IDLE       →(미니카 N회 연속 감지)→  NAVIGATING
  NAVIGATING →(완료)→  SEARCHING
  NAVIGATING →(실패)→  COOLDOWN
  SEARCHING  →(OAK-D ARRIVED)→  COOLDOWN
  SEARCHING  →(OAK-D IDLE/타임아웃)→  COOLDOWN
  COOLDOWN   →(쿨다운 완료)→  IDLE

Subscribes:
  /minicar_detected  (std_msgs/Bool)   - YOLO 감지 결과
  /nav_status        (std_msgs/String) - Nav2 상태
  /approach_status   (std_msgs/String) - OAK-D 접근 상태

Publishes:
  /navigate_to_goal  (std_msgs/Bool)   - Nav2 Controller 트리거
  /start_approach    (std_msgs/Bool)   - OAK-D 접근 노드 트리거
  /manager_state     (std_msgs/String) - 현재 매니저 상태

Parameters:
  detection_count_threshold (int)   : 네비게이션 트리거 전 연속 감지 횟수 (default: 5)
  cooldown_sec              (float) : 쿨다운 초 (default: 10.0)
  check_rate                (float) : 상태 확인 주기 Hz (default: 5.0)
"""

import threading
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Bool, String


class ManagerState(str, Enum):
    """매니저 노드 상태. str 상속으로 ROS2 토픽 퍼블리시에 바로 사용 가능."""
    IDLE       = 'IDLE'
    NAVIGATING = 'NAVIGATING'
    SEARCHING  = 'SEARCHING'   # Nav2 도착 후 OAK-D로 미니카 탐색/접근 중
    COOLDOWN   = 'COOLDOWN'


class MinicarNavManagerNode(Node):
    def __init__(self):
        super().__init__('minicar_nav_manager')

        # --- Parameters ---
        self.declare_parameter('detection_count_threshold', 5)
        self.declare_parameter('cooldown_sec',             10.0)
        self.declare_parameter('check_rate',                5.0)

        self.det_threshold = self.get_parameter('detection_count_threshold').value
        self.cooldown_sec  = self.get_parameter('cooldown_sec').value
        self.check_rate    = self.get_parameter('check_rate').value

        # --- Internal State ---
        self.state            = ManagerState.IDLE
        self.det_count        = 0      # 연속 감지 카운터
        self.cooldown_ticks   = 0      # 쿨다운 남은 틱 수
        self._last_detected   = False
        self._approach_active = False  # start_approach=True 를 실제로 보냈는지 추적 (#1)

        # Thread safety — callback과 timer가 공유 상태를 동시에 접근할 수 있음
        self.lock = threading.Lock()

        # --- Publishers ---
        self.pub_nav_trigger    = self.create_publisher(Bool,   '/navigate_to_goal', 10)
        self.pub_start_approach = self.create_publisher(Bool,   '/start_approach',   10)
        self.pub_state          = self.create_publisher(String, '/manager_state',    10)

        # --- Subscribers ---
        self.sub_detected   = self.create_subscription(
            Bool, '/minicar_detected', self._on_detected, 10
        )
        self.sub_nav_status = self.create_subscription(
            String, '/nav_status', self._on_nav_status, 10
        )
        self.sub_approach   = self.create_subscription(
            String, '/approach_status', self._on_approach_status, 10
        )

        # --- Timer ---
        period = 1.0 / self.check_rate
        self.timer = self.create_timer(period, self._tick)

        self._publish_state()
        self.get_logger().info(
            f'MinicarNavManager started | '
            f'threshold={self.det_threshold} | cooldown={self.cooldown_sec}s'
        )

    # ------------------------------------------------------------------ #
    #  Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _on_detected(self, msg: Bool):
        with self.lock:
            self._last_detected = msg.data

    def _on_nav_status(self, msg: String):
        status = msg.data
        if self.state == ManagerState.NAVIGATING:
            if status == 'SUCCEEDED':
                self.get_logger().info('Navigation succeeded → entering SEARCHING')
                self._enter_searching()
            elif status == 'FAILED':
                self.get_logger().warn('Navigation failed → entering COOLDOWN')
                self._enter_cooldown()

    def _on_approach_status(self, msg: String):
        # #1: _approach_active 플래그로 startup IDLE 메시지와 실제 종료 구분
        if not self._approach_active or self.state != ManagerState.SEARCHING:
            return
        status = msg.data
        if status == 'ARRIVED':
            self.get_logger().info('Approach ARRIVED → entering COOLDOWN')
            self._stop_approach()
            self._enter_cooldown()
        elif status == 'IDLE':
            self.get_logger().warn('Approach ended (timeout) → entering COOLDOWN')
            self._enter_cooldown()
        # SEARCHING / APPROACHING 은 중간 상태 — 무시

    # ------------------------------------------------------------------ #
    #  State Machine Tick                                                  #
    # ------------------------------------------------------------------ #

    def _tick(self):
        if self.state == ManagerState.IDLE:
            self._tick_idle()
        elif self.state == ManagerState.COOLDOWN:
            self._tick_cooldown()
        # NAVIGATING: wait for nav_status callback

    def _tick_idle(self):
        with self.lock:
            detected = self._last_detected

        if detected:
            self.det_count += 1
            self.get_logger().debug(
                f'Detection count: {self.det_count}/{self.det_threshold}'
            )
        else:
            if self.det_count > 0:
                self.get_logger().debug('Detection lost → resetting counter')
            self.det_count = 0

        if self.det_count >= self.det_threshold:
            self.get_logger().info(
                f'Minicar detected {self.det_count} times in a row → triggering navigation!'
            )
            self._enter_navigating()

    def _tick_cooldown(self):
        self.cooldown_ticks -= 1
        if self.cooldown_ticks <= 0:
            self.get_logger().info('Cooldown finished → back to IDLE')
            self._enter_idle()

    # ------------------------------------------------------------------ #
    #  State Transitions                                                   #
    # ------------------------------------------------------------------ #

    def _enter_navigating(self):
        self.state     = ManagerState.NAVIGATING
        self.det_count = 0
        self._publish_state()

        trigger = Bool()
        trigger.data = True
        self.pub_nav_trigger.publish(trigger)
        self.get_logger().info('Published navigate_to_goal=True')

    def _enter_searching(self):
        self.state            = ManagerState.SEARCHING
        self.det_count        = 0
        self._approach_active = True
        self._publish_state()
        trigger = Bool()
        trigger.data = True
        self.pub_start_approach.publish(trigger)
        self.get_logger().info('Published start_approach=True')

    def _stop_approach(self):
        self._approach_active = False
        trigger = Bool()
        trigger.data = False
        self.pub_start_approach.publish(trigger)

    def _enter_cooldown(self):
        if self.state == ManagerState.SEARCHING:
            self._stop_approach()
        self.state          = ManagerState.COOLDOWN
        self.cooldown_ticks = int(self.cooldown_sec * self.check_rate)
        self.det_count      = 0
        self._publish_state()
        self.get_logger().info(
            f'Cooldown started ({self.cooldown_sec}s / {self.cooldown_ticks} ticks)'
        )

    def _enter_idle(self):
        self.state            = ManagerState.IDLE
        self.det_count        = 0
        self.cooldown_ticks   = 0
        self._approach_active = False
        self._publish_state()

    def _publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.pub_state.publish(msg)
        self.get_logger().info(f'Manager state → {self.state.value}')


def main(args=None):
    rclpy.init(args=args)
    node = MinicarNavManagerNode()
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
