#!/usr/bin/env python3
"""
localization_init.py
--------------------
RViz의 '2D Pose Estimate' 수동 클릭을 대체하는 초기화 노드.

사용 순서:
  터미널 1: ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot6 map:=...
  터미널 2: ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot6
  터미널 3: ros2 run minicar_navigator localization_init  ← 이 스크립트

동작:
  1. /robot6/dock_status 확인
  2. /robot6/initialpose 퍼블리시  (AMCL 초기 위치 자동 설정)
  3. /robot6/undock 액션 호출       (자동 언도킹)

Parameters (minicar_nav_params.yaml):
  namespace : 로봇 네임스페이스     (default: 'robot6')
  dock_x    : 도크의 map x 좌표     (default: 0.0)
  dock_y    : 도크의 map y 좌표     (default: 0.0)
  dock_yaw  : 도크 방향각 [rad]     (default: 0.0)
"""

import math

import rclpy
import rclpy.duration
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import PoseWithCovarianceStamped
from irobot_create_msgs.action import Undock
from irobot_create_msgs.msg import DockStatus


def _yaw_to_quat(yaw: float):
    h = yaw / 2.0
    return 0.0, 0.0, math.sin(h), math.cos(h)


class LocalizationInitNode(Node):
    def __init__(self):
        super().__init__('localization_init')

        self.declare_parameter('namespace', 'robot6')
        self.declare_parameter('dock_x',    0.0)
        self.declare_parameter('dock_y',    0.0)
        self.declare_parameter('dock_yaw',  0.0)

        ns            = self.get_parameter('namespace').value
        self.dock_x   = self.get_parameter('dock_x').value
        self.dock_y   = self.get_parameter('dock_y').value
        self.dock_yaw = self.get_parameter('dock_yaw').value

        self._is_docked         = False
        self._dock_received      = False
        self._amcl_pose_received = False

        # /robot6/dock_status
        self.create_subscription(
            DockStatus,
            f'/{ns}/dock_status',
            self._dock_callback,
            10,
        )

        # /robot6/amcl_pose — AMCL이 초기 위치를 수락했는지 확인용
        self.create_subscription(
            PoseWithCovarianceStamped,
            f'/{ns}/amcl_pose',
            self._amcl_pose_callback,
            10,
        )

        # /robot6/initialpose  (AMCL 초기 위치 토픽)
        self._pub_pose = self.create_publisher(
            PoseWithCovarianceStamped,
            f'/{ns}/initialpose',
            10,
        )

        # /robot6/undock
        self._undock_client = ActionClient(self, Undock, f'/{ns}/undock')

        self.get_logger().info(
            f'namespace=/{ns} | '
            f'dock=({self.dock_x:.3f}, {self.dock_y:.3f}, yaw={self.dock_yaw:.3f}rad)'
        )

    def _dock_callback(self, msg: DockStatus):
        self._is_docked     = msg.is_docked
        self._dock_received = True

    def _amcl_pose_callback(self, msg: PoseWithCovarianceStamped):
        self._amcl_pose_received = True

    def run(self):
        # Step 0: AMCL 포즈가 이미 설정돼 있는지 확인 (2초 대기)
        self.get_logger().info('AMCL 포즈 상태 확인 중...')
        rclpy.spin_once(self, timeout_sec=2.0)

        if self._amcl_pose_received:
            self.get_logger().info('AMCL 초기 위치가 이미 설정되어 있습니다. 초기화를 건너뜁니다.')
            return

        self.get_logger().info('AMCL 초기 위치 미설정. 자동 초기화를 시작합니다.')

        # Step 1: dock_status 수신 대기 (최대 10초)
        self.get_logger().info('dock_status 수신 대기 중...')
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=10)

        while rclpy.ok() and not self._dock_received:
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.get_clock().now() >= deadline:
                self.get_logger().error(
                    'dock_status 수신 실패 (10초 타임아웃). '
                    '토픽 이름이나 네임스페이스를 확인하세요.'
                )
                break

        if self._dock_received:
            self.get_logger().info(
                f'도크 상태: {"도킹됨" if self._is_docked else "도킹 안 됨"}'
            )

        # Step 2: AMCL 초기 위치 자동 설정
        self._publish_initial_pose()

        # Step 3: 도킹 상태면 자동 언도킹
        if self._is_docked:
            self._do_undock()
        else:
            self.get_logger().warn(
                '도크에 없거나 상태 확인 실패. 언도킹을 건너뜁니다.'
            )

        self.get_logger().info('초기화 완료.')

    # ------------------------------------------------------------------ #

    def _build_initial_pose_msg(self) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = self.dock_x
        msg.pose.pose.position.y = self.dock_y
        msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = _yaw_to_quat(self.dock_yaw)
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.0685
        return msg

    def _publish_initial_pose(self):
        # AMCL은 TF 타이밍 문제로 첫 메시지를 놓칠 수 있으므로
        # amcl_pose 응답이 올 때까지 1초 간격으로 반복 발행 (최대 15초)
        self.get_logger().info(
            f'초기 위치 발행 시작: '
            f'x={self.dock_x:.3f}  y={self.dock_y:.3f}  yaw={self.dock_yaw:.3f}rad'
        )

        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=15)
        attempt  = 0

        while rclpy.ok() and not self._amcl_pose_received:
            if self.get_clock().now() >= deadline:
                self.get_logger().warn(
                    'AMCL 응답 없음 (15초 타임아웃). '
                    'AMCL이 실행 중인지, 토픽 네임스페이스를 확인하세요.'
                )
                return

            msg = self._build_initial_pose_msg()
            msg.header.stamp = self.get_clock().now().to_msg()
            self._pub_pose.publish(msg)

            attempt += 1
            self.get_logger().info(f'initialpose 발행 #{attempt} — AMCL 응답 대기 중...')
            rclpy.spin_once(self, timeout_sec=1.0)

        self.get_logger().info('AMCL 초기 위치 설정 완료!')

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
