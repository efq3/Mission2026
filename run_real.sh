#!/bin/bash

echo "Cleaning up existing processes..."
killall -9 MicroXRCEAgent 2>/dev/null || true
pkill -9 -f "ros2 run vtail_control" 2>/dev/null || true
sleep 1

echo "Starting Real Drone Environment..."

ROS_SETUP="source /opt/ros/humble/setup.bash 2>/dev/null || true && source ~/Competition_ws/Mission2026/install/setup.bash"

# 1. Agent (Micro XRCE-DDS Agent)
# 실제 라즈베리파이 환경에서는 Pixhawk와 UART/시리얼 연결을 주로 사용합니다.
# 장치 이름(/dev/ttyAMA0 등)과 보드레이트는 하드웨어 설정에 맞게 변경하세요.
# 만약 기존처럼 UDP 통신을 그대로 사용한다면: MicroXRCEAgent udp4 -p 8888 로 변경하세요.
gnome-terminal --tab --title="Agent" -- bash -c "$ROS_SETUP && MicroXRCEAgent serial --dev /dev/ttyAMA0 -b 921600; exec bash"

# 2. ROS2 Nodes (새로 작성된 launch 파일 실행)
gnome-terminal --tab --title="ROS2 Nodes" -- bash -c "$ROS_SETUP && ros2 launch vtail_control real_mission.launch.py; exec bash"
