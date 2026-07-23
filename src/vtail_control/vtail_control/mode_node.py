import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time

from px4_msgs.msg import MissionResult, VehicleCommand, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import Point

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')
        
        # ROS 2 파라미터: 타겟 WP 번호 (기본값 3)
        # 미션 목록: 1번 이륙, 2번 WP1, 3번 WP2 -> target_wp = 3
        self.declare_parameter('target_wp', 3)
        self.target_wp = self.get_parameter('target_wp').value

        # PX4 micro-XRCE-DDS 통신 완벽 호환 QoS 설정
        self.px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 1. PX4 오도메트리 수신 (/fmu/out/vehicle_odometry)
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            self.px4_qos
        )

        # 2. 미션 진행 상태 수신 (/fmu/out/mission_result)
        self.mission_sub = self.create_subscription(
            MissionResult, 
            '/fmu/out/mission_result', 
            self.mission_callback, 
            self.px4_qos
        )

        # 3. 기체 네비게이션 상태 수신 (/fmu/out/vehicle_status)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            self.px4_qos
        )

        # 4. YOLO 타겟 위치 수신 (/yolo/target_position)
        self.yolo_sub = self.create_subscription(
            Point, 
            '/yolo/target_position', 
            self.yolo_callback, 
            10
        )

        # 5. PX4 제어 명령 퍼블리셔 (/fmu/in/vehicle_command)
        self.command_pub = self.create_publisher(
            VehicleCommand, 
            '/fmu/in/vehicle_command', 
            self.px4_qos
        )

        # 6. 상태 확인 및 모드 제어 타이머 (10Hz)
        self.timer = self.create_timer(0.1, self.check_status_loop)

        # 내부 상태 변수
        self.current_mode = "MISSION"
        self.actual_nav_state = 0
        self.last_yolo_time = time.time()
        self.yolo_detected_ever = False
        self.seq_current = -1
        self.seq_reached = -1
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0

        self.get_logger().info(
            f"🚀 MissionControlNode 시작! (목표: 미션 {self.target_wp}번 항목 접근 시 OFFBOARD 전환)"
        )

    def odom_callback(self, msg):
        """ PX4 실시간 위치/고도 업데이트 """
        self.local_x = float(msg.position[0])
        self.local_y = float(msg.position[1])
        self.local_z = float(-msg.position[2])

    def status_callback(self, msg):
        """ PX4 실제 nav_state 수신 """
        self.actual_nav_state = msg.nav_state

    def trigger_offboard(self, reason):
        """ 자동 OFFBOARD 모드 전환 트리거 """
        if self.current_mode == "MISSION":
            self.get_logger().info(f"⚡⚡⚡ [{reason}] -> OFFBOARD 모드로 전환합니다!")
            self.current_mode = "OFFBOARD"

    def is_target_wp_approaching(self):
        """
        목표 WP(예: 3번 WP2) 접근 여부 확인
        - seq_current가 target_wp일 때 (WP1 도달 후 WP2로 향하는 비행 중)
        - 또는 seq_reached가 target_wp - 1일 때 (WP1 도착 직후)
        """
        if self.seq_current < 0 and self.seq_reached < 0:
            return False
            
        # 정확히 target_wp(예: 3)로 향하고 있거나, 바로 전 웨이포인트(예: 2)에 도착했을 때
        if self.seq_current == self.target_wp or self.seq_reached == (self.target_wp - 1):
            return True
            
        return False

    def yolo_callback(self, msg):
        """ YOLO 타겟 수신 시 시간 갱신 """
        self.last_yolo_time = time.time()
        self.yolo_detected_ever = True
        
        # WP1을 지나 WP2로 향할 때만 YOLO 감지로 OFFBOARD 전환
        if self.is_target_wp_approaching():
            self.trigger_offboard("WP2 향하는 중 YOLO 타겟 카메라 포착")

    def mission_callback(self, msg):
        """ PX4 MissionResult 수신 시 WP 체크 """
        self.seq_current = msg.seq_current
        self.seq_reached = msg.seq_reached
        
        if self.is_target_wp_approaching():
            self.trigger_offboard(f"목표 WP{self.target_wp} 미션 순번 도달 감지(seq_current={self.seq_current})")

    def check_status_loop(self):
        """ 1초마다 터미널에 실시간 상태 출력 및 모드 관리 """
        
        # 1. 터미널 실시간 로그 (1초 주기)
        nav_state_str = "OFFBOARD(14)" if self.actual_nav_state == 14 else f"NAV_{self.actual_nav_state}"
        self.get_logger().info(
            f"📍 [SITL 실시간] X={self.local_x:.1f}m, Y={self.local_y:.1f}m, 고도={self.local_z:.1f}m | 현재향하는WP={self.seq_current} (목표={self.target_wp}) | PX4모드={nav_state_str} | 노드상태={self.current_mode}",
            throttle_duration_sec=1.0
        )

        # 2. OFFBOARD 상태 유지 및 타겟 상실 복귀 관리
        if self.current_mode == "OFFBOARD":
            self.send_offboard_command()

            # 실제 YOLO 타겟을 감지한 적이 있는 경우 3초 이상 상실 시 MISSION으로 복귀
            if self.yolo_detected_ever:
                time_since_last_target = time.time() - self.last_yolo_time
                if time_since_last_target > 3.0:
                    self.get_logger().info("⚠️ 타겟 상실 3초 경과 -> MISSION 모드로 복귀합니다.")
                    self.current_mode = "MISSION_RETURN"

        elif self.current_mode == "MISSION_RETURN":
            self.send_mission_command()

    def send_offboard_command(self):
        """ OFFBOARD 모드 전환 명령 전송 """
        msg1 = VehicleCommand()
        msg1.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg1.param1 = 1.0
        msg1.param2 = 6.0 # OFFBOARD
        msg1.target_system = 1
        msg1.target_component = 1
        msg1.source_system = 1
        msg1.source_component = 1
        msg1.from_external = True
        msg1.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg1)

        msg2 = VehicleCommand()
        msg2.command = 100001 # VEHICLE_CMD_SET_NAV_STATE
        msg2.param1 = 14.0 # NAVIGATION_STATE_OFFBOARD = 14
        msg2.target_system = 1
        msg2.target_component = 1
        msg2.source_system = 1
        msg2.source_component = 1
        msg2.from_external = True
        msg2.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg2)

    def send_mission_command(self):
        """ AUTO.MISSION 복귀 명령 전송 """
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0
        msg.param2 = 4.0 # AUTO.MISSION
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MissionControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()