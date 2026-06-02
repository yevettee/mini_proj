#!/usr/bin/env bash
# apply_overrides.sh
# ------------------
# minicar_room Gazebo 세팅을 위해 turtlebot4_ws 패키지 파일들을 덮어씁니다.
# 팀원이 turtlebot4 / turtlebot4_simulator 를 클론해 둔 워크스페이스에 적용하세요.
#
# 사용법:
#   bash apply_overrides.sh                 # 기본 워크스페이스 ~/turtlebot4_ws
#   bash apply_overrides.sh /path/to/ws     # 다른 경로 지정
#
# 적용 파일:
#   turtlebot4_navigation/config/slam.yaml          (소형 대칭방 SLAM 튜닝)
#   turtlebot4_navigation/config/nav2.yaml          (costmap inflation 축소)
#   turtlebot4_ignition_bringup/worlds/minicar_room.sdf   (방 + RC car)
#   turtlebot4_ignition_bringup/launch/turtlebot4_spawn.launch.py  (map 인자 plumbing)
set -e

WS="${1:-$HOME/turtlebot4_ws}"
SRC="$WS/src"
HERE="$(cd "$(dirname "$0")" && pwd)"

NAV="$SRC/turtlebot4/turtlebot4_navigation"
IGN="$SRC/turtlebot4_simulator/turtlebot4_ignition_bringup"

[ -d "$NAV" ] || { echo "❌ $NAV 없음 — turtlebot4 패키지를 먼저 클론하세요"; exit 1; }
[ -d "$IGN" ] || { echo "❌ $IGN 없음 — turtlebot4_simulator 패키지를 먼저 클론하세요"; exit 1; }

echo "워크스페이스: $WS"
cp -v "$HERE/turtlebot4_navigation/config/slam.yaml"        "$NAV/config/slam.yaml"
cp -v "$HERE/turtlebot4_navigation/config/nav2.yaml"        "$NAV/config/nav2.yaml"
cp -v "$HERE/turtlebot4_ignition_bringup/worlds/minicar_room.sdf"        "$IGN/worlds/minicar_room.sdf"
cp -v "$HERE/turtlebot4_ignition_bringup/launch/turtlebot4_spawn.launch.py" "$IGN/launch/turtlebot4_spawn.launch.py"

echo
echo "✅ 적용 완료. --symlink-install 워크스페이스면 재빌드 없이 적용됩니다."
echo "   아니라면: cd $WS && colcon build --packages-select turtlebot4_navigation turtlebot4_ignition_bringup"
echo
echo "참고: localization 기본 맵 경로 = ~/mini_proj/map/minicar_sim_map.yaml"
echo "      (이 저장소를 ~/mini_proj 에 클론해야 자동 인식. 아니면 minisim에 map:=<경로> 추가)"
