#!/usr/bin/env python3
"""
nav2_controller.py
------------------
Nav2 NavigateToPose 액션 클라이언트 노드.
/navigate_to_goal (std_msgs/Bool) 토픽을 수신하면 설정된 목표 지점으로 TurtleBot4를 이동시킵니다.

Subscribes:
  /navigate_to_goal  (std_msgs/Bool) - True 수신 시 네비게이션 시작

Publishes:
  /nav_status        (std_msgs/String) - 네비게이션 상태 ('IDLE'|'NAVIGATING'|'SUCCEEDED'|'FAILED')

Parameters:
  goal_x        (float) : 목표 x 좌표 (map frame, default: 1.0)
  goal_y        (float) : 목표 y 좌표 (map frame, default: 1.0)
  goal_yaw      (float) : 목표 방향각 radian (default: 0.0)
  goal_frame_id (str)   : 좌표 프레임 (default: 'map')
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus


def euler_to_quaternion(yaw: float):
    """yaw(라디안) → quaternion (x, y, z, w) 변환."""
    half = yaw / 2.0
    return 0.0, 0.0, math.sin(half), math.cos(half)


class Nav2ControllerNode(Node):
    def __init__(self):
        super().__init__('nav2_controller')

        # --- Parameters ---
        self.declare_parameter('goal_x',            1.0)
        self.declare_parameter('goal_y',            1.0)
        self.declare_parameter('goal_yaw',          0.0)
        self.declare_parameter('goal_frame_id',     'map')
        self.declare_parameter('action_server_name', 'navigate_to_pose')

        self.goal_x             = self.get_parameter('goal_x').value
        self.goal_y             = self.get_parameter('goal_y').value
        self.goal_yaw           = self.get_parameter('goal_yaw').value
        self.goal_frame_id      = self.get_parameter('goal_frame_id').value
        action_server_name      = self.get_parameter('action_server_name').value

        # --- State ---
        self.is_navigating = False
        self.cb_group = ReentrantCallbackGroup()

        # --- Action Client ---
        self._action_client = ActionClient(
            self,
            NavigateToPose,
            action_server_name,
            callback_group=self.cb_group,
        )

        # --- Subscriber ---
        self.sub_goal = self.create_subscription(
            Bool,
            '/navigate_to_goal',
            self.goal_callback,
            10,
            callback_group=self.cb_group,
        )

        # --- Publisher ---
        self.pub_status = self.create_publisher(String, '/nav_status', 10)
        self._publish_status('IDLE')

        self.get_logger().info(
            f'Nav2Controller ready | goal=({self.goal_x}, {self.goal_y}) '
            f'yaw={self.goal_yaw:.2f}rad frame={self.goal_frame_id} '
            f'action={action_server_name}'
        )

    def goal_callback(self, msg: Bool):
        if not msg.data:
            return
        if self.is_navigating:
            self.get_logger().warn('Already navigating. Ignoring new goal request.')
            return
        self.get_logger().info('Navigation trigger received. Sending goal to Nav2...')
        self._send_goal()

    def _send_goal(self):
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('NavigateToPose action server not available!')
            self._publish_status('FAILED')
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self._build_pose_stamped()

        self.is_navigating = True
        self._publish_status('NAVIGATING')
        self.get_logger().info(
            f'Sending goal: x={self.goal_x}, y={self.goal_y}, yaw={self.goal_yaw:.2f}'
        )

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback,
        )
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        try:
            goal_handle = future.result()
        except Exception as e:
            self.get_logger().error(f'Goal request failed: {e}')
            self.is_navigating = False
            self._publish_status('FAILED')
            return

        if not goal_handle.accepted:
            self.get_logger().error('Goal was REJECTED by Nav2!')
            self.is_navigating = False
            self._publish_status('FAILED')
            return

        self.get_logger().info('Goal ACCEPTED by Nav2. Waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result()
        status = result.status

        self.is_navigating = False

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Navigation SUCCEEDED! TurtleBot4 reached the goal.')
            self._publish_status('SUCCEEDED')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn('Navigation was CANCELED.')
            self._publish_status('IDLE')
        else:
            self.get_logger().error(f'Navigation FAILED with status: {status}')
            self._publish_status('FAILED')

    def _feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        remaining = feedback.distance_remaining
        self.get_logger().debug(f'Distance remaining: {remaining:.2f} m')

    def _build_pose_stamped(self) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.header.frame_id = self.goal_frame_id

        pose.pose.position.x = self.goal_x
        pose.pose.position.y = self.goal_y
        pose.pose.position.z = 0.0

        qx, qy, qz, qw = euler_to_quaternion(self.goal_yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def _publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.pub_status.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Nav2ControllerNode()
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
