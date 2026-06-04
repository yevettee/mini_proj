"""
minicar_nav.launch.py
---------------------
전체 파이프라인을 한 번에 실행하는 런치 파일.
모델 경로 등 기본값은 config/minicar_nav_params.yaml 에서 관리.

사용 예:
  ros2 launch minicar_navigator minicar_nav.launch.py
  ros2 launch minicar_navigator minicar_nav.launch.py target_class:=bottle

실행 순서 (사전 조건):
  터미널1: ros2 launch turtlebot4_navigation localization.launch.py namespace:=/robot6 map:=...
  터미널2: ros2 launch turtlebot4_navigation nav2.launch.py namespace:=/robot6
  터미널3: ros2 launch minicar_navigator minicar_nav.launch.py  ← 이 파일
           (AMCL 초기 위치 미설정 시 localization_init 이 자동으로 언도킹까지 처리)
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
        'minicar_nav_params.yaml'
    )

    # ------------------------------------------------------------------ #
    #  Launch Arguments — 자주 바꾸는 값만 노출                            #
    # ------------------------------------------------------------------ #
    args = [
        DeclareLaunchArgument('target_distance', default_value='0.5',
                              description='미니카 접근 정지 거리 (m)'),
    ]

    # ------------------------------------------------------------------ #
    #  Nodes — 모델 경로는 config에서 관리                                 #
    # ------------------------------------------------------------------ #

    # AMCL 초기 위치 미설정 시 자동으로 initialpose 발행 + 언도킹
    # 이미 설정돼 있으면 2초 확인 후 자동 종료
    localization_init = Node(
        package='minicar_navigator',
        executable='localization_init',
        name='localization_init',
        output='screen',
        parameters=[config],
    )

    yolo_detector = Node(
        package='minicar_navigator',
        executable='yolo_detector',
        name='yolo_detector',
        output='screen',
        parameters=[config],
    )

    nav2_controller = Node(
        package='minicar_navigator',
        executable='nav2_controller',
        name='nav2_controller',
        output='screen',
        parameters=[config],
    )

    manager = Node(
        package='minicar_navigator',
        executable='minicar_nav_manager',
        name='minicar_nav_manager',
        output='screen',
        parameters=[config],
    )

    oakd_approach = Node(
        package='minicar_navigator',
        executable='oakd_approach_node',
        name='oakd_approach_node',
        output='screen',
        parameters=[
            config,
            {'target_distance': LaunchConfiguration('target_distance')},
        ],
        remappings=[
            ('/tf',        '/robot6/tf'),
            ('/tf_static', '/robot6/tf_static'),
        ],
    )

    return LaunchDescription(args + [
        localization_init,
        yolo_detector,
        nav2_controller,
        manager,
        oakd_approach,
    ])
