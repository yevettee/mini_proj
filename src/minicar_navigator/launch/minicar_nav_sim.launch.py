"""
minicar_nav_sim.launch.py
-------------------------
Gazebo(Ignition) 시뮬용 런치 (B안: 웹캠 yolo_detector 제외).
3개 노드만 실행: nav2_controller, minicar_nav_manager, oakd_approach_node.
파라미터는 config/minicar_nav_params_sim.yaml 에서 로드, 모든 노드에 use_sim_time=true.

선행 조건 (별도 터미널):
  sim
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py \
      world:=<world> nav2:=true localization:=true rviz:=true   (+ 맵에 RC car 모델 스폰)
  # 언도크 + RViz '2D Pose Estimate'로 초기 위치 지정

사용 예:
  ros2 launch minicar_navigator minicar_nav_sim.launch.py \
      goal_x:=1.5 goal_y:=0.3 goal_yaw:=0.0

시작 트리거 (웹캠 대신 수동, 다른 터미널에서 한 번):
  ros2 topic pub --once /minicar_detected std_msgs/msg/Bool "{data: true}"
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    config = os.path.join(
        get_package_share_directory('minicar_navigator'),
        'config',
        'minicar_nav_params_sim.yaml'
    )

    sim_time = {'use_sim_time': True}

    args = [
        DeclareLaunchArgument('goal_x',   default_value='0.0',
                              description='Nav2 목표 x (map, m) — RC car가 보이는 위치'),
        DeclareLaunchArgument('goal_y',   default_value='0.0',
                              description='Nav2 목표 y (map, m)'),
        DeclareLaunchArgument('goal_yaw', default_value='0.0',
                              description='Nav2 목표 yaw (rad)'),
        DeclareLaunchArgument('target_distance', default_value='0.5',
                              description='RC car 접근 정지 거리 (m)'),
    ]

    nav2_controller = Node(
        package='minicar_navigator',
        executable='nav2_controller',
        name='nav2_controller',
        output='screen',
        parameters=[
            config,
            sim_time,
            {'goal_x':   LaunchConfiguration('goal_x')},
            {'goal_y':   LaunchConfiguration('goal_y')},
            {'goal_yaw': LaunchConfiguration('goal_yaw')},
        ],
    )

    manager = Node(
        package='minicar_navigator',
        executable='minicar_nav_manager',
        name='minicar_nav_manager',
        output='screen',
        parameters=[config, sim_time],
    )

    oakd_approach = Node(
        package='minicar_navigator',
        executable='oakd_approach_node',
        name='oakd_approach_node',
        output='screen',
        parameters=[
            config,
            sim_time,
            {'target_distance': LaunchConfiguration('target_distance')},
        ],
    )

    # B안: yolo_detector(웹캠) 미실행. 시작은 /minicar_detected 수동 발행으로.
    return LaunchDescription(args + [nav2_controller, manager, oakd_approach])
