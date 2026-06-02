# gazebo_setup — turtlebot4_ws 오버라이드 파일

minicar_room Gazebo 시뮬을 재현하기 위해 **turtlebot4_ws 패키지에 덮어써야 하는 파일**들입니다.
(이 파일들은 `~/mini_proj` 가 아니라 `~/turtlebot4_ws/src` 안의 turtlebot4 / turtlebot4_simulator 패키지 소속이라 별도로 모았습니다.)

## 적용 방법

```bash
cd ~/mini_proj/gazebo_setup
bash apply_overrides.sh                # 기본 ~/turtlebot4_ws
# 또는: bash apply_overrides.sh /your/ws
```

`--symlink-install` 워크스페이스면 재빌드 없이 적용됩니다. 아니면:
```bash
cd ~/turtlebot4_ws
colcon build --packages-select turtlebot4_navigation turtlebot4_ignition_bringup
```

## 파일별 설명

| 파일 | 대상 경로 (turtlebot4_ws/src/…) | 내용 |
|------|------|------|
| `turtlebot4_navigation/config/slam.yaml` | `turtlebot4/turtlebot4_navigation/config/` | 소형 대칭방용 SLAM 튜닝: `max_laser_range 12→5`, `do_loop_closing false`, `minimum_travel_distance/heading 0→0.2` (포즈 발산/방 밖 매핑 방지) |
| `turtlebot4_navigation/config/nav2.yaml` | `turtlebot4/turtlebot4_navigation/config/` | costmap inflation 축소: `inflation_radius 0.45→0.25`, `cost_scaling_factor 4→6` (좁은 방에서 통로 막힘 방지) |
| `turtlebot4_ignition_bringup/worlds/minicar_room.sdf` | `turtlebot4_simulator/turtlebot4_ignition_bringup/worlds/` | 방(외벽+divider) + RC car(19×8cm 검정, 원점기준 x=1.0 y=-1.8 yaw=0.6) |
| `turtlebot4_ignition_bringup/launch/turtlebot4_spawn.launch.py` | `turtlebot4_simulator/turtlebot4_ignition_bringup/launch/` | `map` 인자 추가 + localization에 전달 (기본값 `~/mini_proj/map/minicar_sim_map.yaml`) |

## 맵
- 저장 맵: `~/mini_proj/map/minicar_sim_map.{pgm,yaml}` (이 저장소에 포함)
- launch 기본 맵 경로가 `~/mini_proj/map/minicar_sim_map.yaml` 이므로 **이 저장소를 `~/mini_proj` 에 클론**하면 자동 인식됩니다.
- 다른 경로면 `minisim` 대신 `... map:=/경로/minicar_sim_map.yaml` 로 실행하세요.

## world 재생성 (선택)
`~/mini_proj/map/room_measured.py` 가 minicar_room.sdf 생성기입니다 (방 치수/ RC car 위치 상수 포함). 수정 후:
```bash
python3 ~/mini_proj/map/room_measured.py   # minicar_room.sdf 재생성
```
