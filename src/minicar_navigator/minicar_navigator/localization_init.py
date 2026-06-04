#!/usr/bin/env python3
"""
localization_init.py
--------------------
도킹 상태 확인 후 자동 언도킹만 수행하는 노드.
초기 위치(2D Pose Estimate)는 사용자가 RViz에서 직접 설정.

사용 순서:
  터미널1: ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot6 map:=...
  터미널2: ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot6
  터미널3: RViz에서 2D Pose Estimate 설정
  터미널4: ros2 launch minicar_navigator minicar_nav.launch.py  ← 언도킹 자동 수행
"""

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus


class LocalizationInitNode(Node):
    def __init__(self):
        super().__init__('localization_init')

        self.declare_parameter('namespace', 'robot6')
        ns = self.get_parameter('namespace').value

        self._is_docked    = False
        self._dock_received = False

        self.create_subscription(
            DockStatus,
            f'/{ns}/dock_status',
            self._dock_callback,
            10,
        )

        self._dock_client   = ActionClient(self, Dock,   f'/{ns}/dock')
        self._undock_client = ActionClient(self, Undock, f'/{ns}/undock')

        self.get_logger().info(f'localization_init 시작 | namespace=/{ns}')

    def _dock_callback(self, msg: DockStatus):
        self._is_docked     = msg.is_docked
        self._dock_received = True

    def run(self):
        # dock_status 수신 대기 (최대 10초)
        self.get_logger().info('dock_status 확인 중...')
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=10)

        while rclpy.ok() and not self._dock_received:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.get_clock().now() >= deadline:
                self.get_logger().error('dock_status 수신 실패 (10초 타임아웃)')
                return

        self.get_logger().info(f'도크 상태: {"도킹됨" if self._is_docked else "도킹 안 됨"}')

        if not self._is_docked:
            self.get_logger().info('언도킹 상태 → 도킹 후 언도킹합니다.')
            self._do_dock()

        self._do_undock()

    def _do_dock(self):
        self.get_logger().info('Dock 액션 서버 연결 대기 중...')
        if not self._dock_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Dock 서버에 연결할 수 없습니다.')
            return

        self.get_logger().info('도킹 시작...')
        future = self._dock_client.send_goal_async(Dock.Goal())
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Dock goal이 거부되었습니다.')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('도킹 완료.')

    def _do_undock(self):
        self.get_logger().info('Undock 액션 서버 연결 대기 중...')
        if not self._undock_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Undock 서버에 연결할 수 없습니다.')
            return

        self.get_logger().info('언도킹 시작...')
        future = self._undock_client.send_goal_async(Undock.Goal())
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Undock goal이 거부되었습니다.')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info('언도킹 완료.')


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationInitNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
