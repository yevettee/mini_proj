#!/usr/bin/env python3
"""
room_measured.py — 실측값 기반 미니카 방 Gazebo 월드 생성 (직선 벽).
좌표: 방 좌하단 = 월드 원점(0,0), x→오른쪽, y→위, 단위 m.

실측(cm→m):
  외곽 3.60 x 2.70
  내부 가로 칸막이: 왼쪽 벽(x=0)에서 x=2.00 까지, 높이 y=DIV_Y
  웹캠 알코브(칸막이 위, 왼쪽): 가로 0.60 x 세로 0.40
사용: python3 room_measured.py [--preview] [--div-y 1.05]
"""
import sys, numpy as np
MAZE='/home/rokey/turtlebot4_ws/src/turtlebot4_simulator/turtlebot4_ignition_bringup/worlds/maze.sdf'
OUT='/home/rokey/mini_proj/map/minicar_room.sdf'

# ---- 실측 파라미터 (m) ----
W, H   = 3.60, 2.70      # 외곽
DIV_LEN = 2.00           # 칸막이 길이(왼쪽 벽에서)
ALC_W, ALC_H = 0.60, 0.40  # 웹캠 알코브 (가로x세로)
DIV_Y  = 1.05            # 칸막이 높이(바닥에서) — ※측정값에 없어 도면 비율로 추정, 조정 가능
T, HGT = 0.10, 0.50      # 벽 두께, 높이(0.5m)
DKX, DKY = 0.30, 2.00    # (코너기준) 도킹스테이션 위치 → 이 점을 월드 원점(0,0)으로 평행이동

def main():
    div_y = DIV_Y
    if '--div-y' in sys.argv:
        div_y = float(sys.argv[sys.argv.index('--div-y')+1])
    preview = '--preview' in sys.argv

    # 벽 = (x1,y1,x2,y2)  중심선
    h = T / 2.0   # 안쪽 면이 정확히 W×H(=360×270, 안쪽치수)가 되도록 외곽 중심선을 바깥으로
    walls = [
        # 외곽 4벽 (안쪽 치수 = 360×270, 안쪽 면이 x:0~W, y:0~H)
        (-h, -h, W+h, -h), (-h, H+h, W+h, H+h), (-h, -h, -h, H+h), (W+h, -h, W+h, H+h),
        # 내부 가로 칸막이 (왼쪽 안쪽벽 x=0 → x=DIV_LEN)
        (0, div_y, DIV_LEN, div_y),
        # 웹캠 알코브 (칸막이 위, 왼쪽): 오른벽 + 윗벽 (왼쪽=방벽, 아래=칸막이)
        (ALC_W, div_y, ALC_W, div_y + ALC_H),
        (0, div_y + ALC_H, ALC_W, div_y + ALC_H),
    ]
    # 도킹스테이션(코너기준 DKX,DKY)을 월드 원점(0,0)으로 — 전체 평행이동
    walls = [(x1 - DKX, y1 - DKY, x2 - DKX, y2 - DKY) for (x1, y1, x2, y2) in walls]

    boxes = []
    for x1, y1, x2, y2 in walls:
        dx, dy = x2 - x1, y2 - y1
        L = (dx*dx + dy*dy) ** 0.5
        boxes.append(((x1+x2)/2, (y1+y2)/2, L + T, T, np.arctan2(dy, dx)))

    print(f"방 {W}x{H}m | 칸막이 길이 {DIV_LEN}m @ y={div_y}m | 알코브 {ALC_W}x{ALC_H}m | 벽 {len(boxes)}개")
    print(f"추천: 로봇 spawn(=도킹스테이션=원점) x:=0.0 y:=0.0 yaw:=3.14159 (undock 후 +X 향함) | RC car ~ (1.2, -1.8)")

    if preview:
        S = 18; pw, ph = int(W*S)+4, int(H*S)+4
        c = np.full((ph, pw), ' ')
        for cx, cy, sx, sy, yaw in boxes:
            for t in np.linspace(-sx/2, sx/2, int(sx*S)+2):
                px = int((cx + t*np.cos(yaw))*S)+2
                py = int((H - (cy + t*np.sin(yaw)))*S)+2
                if 0 <= py < ph and 0 <= px < pw: c[py, px] = '#'
        for r in range(0, ph, max(1, ph//34)):
            print("".join(c[r, ::max(1, pw//72)]))
        return

    maze = open(MAZE).read()
    gp = maze.index("<model name='ground_plane'>"); gpe = maze.index('</model>', gp)+len('</model>')
    header = maze[:gpe].replace("<world name='maze'>", "<world name='minicar_room'>")
    # maze.sdf와 동일하게: 벽마다 별도 <link>, link pose로 위치, box는 중심정렬(visual/collision pose 없음)
    parts = ["        <model name='walls'>", "            <static>true</static>"]
    for i, (cx, cy, sx, sy, yaw) in enumerate(boxes):
        size = f"{sx:.4f} {sy:.4f} {HGT:.4f}"
        parts.append(f"            <link name='wall{i}'>")
        parts.append(f"                <pose>{cx:.4f} {cy:.4f} {HGT/2:.4f} 0 0 {yaw:.5f}</pose>")
        parts.append(f"                <collision name='col'><geometry><box><size>{size}</size></box></geometry></collision>")
        parts.append(f"                <visual name='vis'><geometry><box><size>{size}</size></box></geometry>"
                     f"<material><ambient>0.7 0.7 0.72 1</ambient><diffuse>0.7 0.7 0.72 1</diffuse><specular>0.2 0.2 0.2 1</specular></material></visual>")
        parts.append(f"            </link>")
    parts += ["        </model>"]

    # 원점 좌표축 마커 (시각 전용=collision 없음 → 라이다/SLAM에 안 잡힘)
    # 빨강=+X, 초록=+Y, 파랑=+Z, 각 0.5m
    if '--no-axes' not in sys.argv:
        # 코너 벽에 묻히지 않게 원점에서 (0.4,0.4) 안쪽 열린 바닥으로 이동, 막대 굵게/길게
        # 원점(0,0)=도크=로봇 위치에 표시 (로봇이 일부 가려도 빨강+X/파랑+Z는 보임)
        ax = [
            ("x_axis", "0.20 0 0.08 0 0 0", "0.40 0.04 0.04", "1 0 0 1"),
            ("y_axis", "0 0.18 0.08 0 0 0", "0.04 0.36 0.04", "0 1 0 1"),
            ("z_axis", "0 0 0.25 0 0 0",    "0.04 0.04 0.50", "0 0 1 1"),
        ]
        parts += ["        <model name='origin_axes'>", "            <static>true</static>"]
        for nm, pose, size, col in ax:
            parts += [f"            <link name='{nm}'>",
                      f"                <pose>{pose}</pose>",
                      f"                <visual name='v'><geometry><box><size>{size}</size></box></geometry>"
                      f"<material><ambient>{col}</ambient><diffuse>{col}</diffuse><emissive>{col}</emissive></material></visual>",
                      "            </link>"]
        parts += ["        </model>"]

    # RC car (원점=도크 기준 x=1.0 y=-1.8, divider 아래 빈 공간). 실측 19×8cm, 검정 차체+캐빈+바퀴4 분리 link.
    if '--no-car' not in sys.argv:
        CAR_X, CAR_Y, CAR_YAW = 1.0, -1.8, 0.6
        parts += ["        <model name='rc_car'>",
                  "            <static>true</static>",
                  f"            <pose>{CAR_X:.4f} {CAR_Y:.4f} 0 0 0 {CAR_YAW:.5f}</pose>",
                  "            <link name='body'>",
                  "                <pose>0 0 0.0350 0 0 0</pose>",
                  "                <collision name='col'><geometry><box><size>0.1900 0.0800 0.0400</size></box></geometry></collision>",
                  "                <visual name='vis'><geometry><box><size>0.1900 0.0800 0.0400</size></box></geometry><material><ambient>0.05 0.05 0.05 1</ambient><diffuse>0.05 0.05 0.05 1</diffuse><specular>0.1 0.1 0.1 1</specular></material></visual>",
                  "            </link>",
                  "            <link name='cabin'>",
                  "                <pose>-0.0100 0 0.0675 0 0 0</pose>",
                  "                <collision name='col'><geometry><box><size>0.0800 0.0600 0.0250</size></box></geometry></collision>",
                  "                <visual name='vis'><geometry><box><size>0.0800 0.0600 0.0250</size></box></geometry><material><ambient>0.15 0.15 0.17 1</ambient><diffuse>0.15 0.15 0.17 1</diffuse><specular>0.2 0.2 0.2 1</specular></material></visual>",
                  "            </link>"]
        for nm, wx, wy in [("wheel_fl",0.06,0.045),("wheel_fr",0.06,-0.045),("wheel_rl",-0.06,0.045),("wheel_rr",-0.06,-0.045)]:
            parts += [f"            <link name='{nm}'>",
                      f"                <pose>{wx:.4f} {wy:.4f} 0.0220 1.5708 0 0</pose>",
                      "                <visual name='vis'><geometry><cylinder><radius>0.0220</radius><length>0.0180</length></cylinder></geometry><material><ambient>0.02 0.02 0.02 1</ambient><diffuse>0.02 0.02 0.02 1</diffuse></material></visual>",
                      "            </link>"]
        parts += ["        </model>"]

    open(OUT, 'w').write(header+"\n"+"\n".join(parts)+"\n    </world>\n</sdf>\n")
    print(f"✅ {OUT}  (원점축: 빨강=+X 초록=+Y 파랑=+Z, --no-axes로 끄기 가능)")

if __name__ == '__main__':
    main()
