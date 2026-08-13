import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleCommand
from std_msgs.msg import Float32, String
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import MissionResult
import time

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')
        
        # 기본 상태 초기화
        self.local_x = 0.0
        self.local_y = 0.0
        self.local_z = 0.0
        
        # PX4 micro-XRCE-DDS 통신 완벽 호환 QoS 설정
        self.px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 1. 기체 오도메트리 수신 (/fmu/out/vehicle_odometry)
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            self.px4_qos
        )
        
        # 2. YOLO 타겟 정보 수신 (YOLO Node로부터)
        self.target_sub = self.create_subscription(
            Float32,
            '/yolo/target_position',
            self.target_callback,
            10
        )
        
        # 3. 기체 네비게이션 상태 수신 (/fmu/out/vehicle_status)
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.status_callback,
            self.px4_qos
        )

        # 4. 미션 결과 수신 (/fmu/out/mission_result)
        self.mission_sub = self.create_subscription(
            MissionResult,
            '/fmu/out/mission_result',
            self.mission_result_callback,
            self.px4_qos
        )
        
        # 5. 제어 명령 퍼블리셔 (/fmu/in/vehicle_command)
        self.command_pub = self.create_publisher(
            VehicleCommand,
            '/fmu/in/vehicle_command',
            10
        )
        
        # 6. Offboard 모드 유지를 위한 Heartbeat 퍼블리셔
        self.offboard_ctrl_mode_pub = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            10
        )
        
        # 7. 이동 목표(Setpoint) 퍼블리셔
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            '/fmu/in/trajectory_setpoint',
            10
        )
        
        # 내부 상태 변수
        self.target_x = 0.0
        self.target_y = 600.0
        self.target_detected = False
        self.yolo_detected_ever = False
        self.last_yolo_time = 0.0
        self.actual_nav_state = 0
        self.arming_state = 1 # 기본적으로 DISARMED
        
        self.current_mode = "MISSION"
        
        self.target_wp = 3
        self.return_wp = 4
        
        self.seq_current = 0
        self.seq_reached = 0

        self.get_logger().info(
            f"🚀 MissionControlNode 시작! (목표: 미션 {self.target_wp}번 항목 접근 시 OFFBOARD 전환)"
        )

    def odom_callback(self, msg):
        """ PX4 실시간 위치/고도 업데이트 """
        self.local_x = float(msg.position[0])
        self.local_y = float(msg.position[1])
        self.local_z = float(-msg.position[2])

    def status_callback(self, msg):
        """ PX4 실제 nav_state 및 arming_state 수신 """
        self.get_logger().info(f"DEBUG: VehicleStatus received! arming={msg.arming_state}, nav={msg.nav_state}")
        self.actual_nav_state = msg.nav_state
        self.arming_state = msg.arming_state

    def target_callback(self, msg):
        """ YOLO 타겟 좌표 수신 (x오차율) """
        self.target_x = msg.data
        self.target_detected = True
        self.yolo_detected_ever = True
        self.last_yolo_time = time.time()
        
    def trigger_offboard(self, reason):
        """ 미션 -> OFFBOARD 전환 """
        self.get_logger().info(f"🔥 OFFBOARD 조건 달성: {reason} -> OFFBOARD 모드로 전환합니다.")
        self.current_mode = "OFFBOARD"

    def trigger_mission_return(self, reason):
        """ OFFBOARD -> 미션 복귀 """
        self.get_logger().info(f"🔙 AUTO.MISSION 복귀 조건 달성: {reason} -> MISSION 모드로 복귀합니다.")
        self.current_mode = "MISSION_RETURN"

    def is_target_wp_approaching(self):
        """ 현재 비행 중인 WP가 목표 WP인지 확인 """
        if self.seq_current == self.target_wp:
            return True
        return False

    def mission_result_callback(self, msg):
        """ PX4 실제 미션 수행 상태 업데이트 """
        self.seq_current = msg.seq_current
        self.seq_reached = msg.seq_reached
        
        # 시동이 걸려있지 않은(Disarmed) 대기 상태에서는 미션 WP 전환 무시 (고도로 보완)
        is_armed = (getattr(self, 'arming_state', 1) == 2) or (getattr(self, 'local_z', 0.0) > 0.5)
        if not is_armed:
            return
        
        # 1. OFFBOARD 상태인 경우: 복귀 WP(예: 4번) 이상에 도달 시 MISSION 모드로 복귀
        if self.current_mode == "OFFBOARD":
            if (self.seq_current >= self.return_wp and self.seq_current > 0) or (self.seq_reached >= self.return_wp and self.seq_reached > 0):
                self.trigger_mission_return(f"복귀 WP{self.return_wp} 도달 감지(seq_current={self.seq_current}, seq_reached={self.seq_reached})")
                
        # 2. MISSION 상태인 경우: 타겟 WP(예: 3번) 접근 시 OFFBOARD로 전환
        elif self.current_mode == "MISSION":
            if self.is_target_wp_approaching():
                self.trigger_offboard(f"목표 WP{self.target_wp} 미션 순번 도달 감지(seq_current={self.seq_current})")

    def check_status_loop(self):
        """ 1초마다 터미널에 실시간 상태 출력 및 모드 관리 """
        
        # 시동이 걸려있지 않은(Disarmed) 대기 상태일 때는 모드 명령을 보내지 않고 수동/대기 유지
        is_armed = (getattr(self, 'arming_state', 1) == 2) or (getattr(self, 'local_z', 0.0) > 0.5)
        if not is_armed:
            self.current_mode = "MISSION"
            arm_str = "DISARMED"
            self.get_logger().info(
                f"📍 [SITL 대기중] X={self.local_x:.1f}m, Y={self.local_y:.1f}m, 고도={self.local_z:.1f}m | 상태={arm_str} | 미션이륙 대기",
                throttle_duration_sec=2.0
            )
            return

        # 1. 터미널 실시간 로그 (1초 주기)
        nav_state_str = "OFFBOARD(14)" if self.actual_nav_state == 14 else f"NAV_{self.actual_nav_state}"
        self.get_logger().info(
            f"📍 [SITL 비행중] X={self.local_x:.1f}m, Y={self.local_y:.1f}m, 고도={self.local_z:.1f}m | 현재향하는WP={self.seq_current} (목표={self.target_wp}, 복귀={self.return_wp}) | PX4모드={nav_state_str} | 노드상태={self.current_mode}",
            throttle_duration_sec=1.0
        )

        # 2. OFFBOARD 상태 유지 및 타겟 상실 복귀 관리
        if self.current_mode == "OFFBOARD":
            # 실제 PX4 상태가 OFFBOARD(14)가 아니면 모드 변경 커맨드 계속 시도
            if getattr(self, 'actual_nav_state', 0) != 14:
                self.set_offboard_mode()
                
            arm_str = "OFFBOARD (제어중)"
            self.get_logger().info(
                f"🛸 [OFFBOARD 제어중] Y={self.local_y:.1f}m | 목표 Y={self.target_y:.1f}m | 상태={arm_str}",
                throttle_duration_sec=1.0
            )
            
            # 영상 처리 기반 요(Yaw) 제어 로직
            # 화면 중심(0) 기준으로 에러 계산
            error_x = self.target_x
            
            # P 제어 (비례 제어)
            p_gain = 0.005 # 민감도 조절
            yaw_rate = -error_x * p_gain
            
            # Yaw Setpoint 계산 (현재 Yaw + Yaw Rate)
            yaw_setpoint = yaw_rate # 임시로 rate 자체를 setpoint로 사용 (개선 필요)
            
            # 목표 Y 좌표로 이동 (전진)
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint(self.local_x, self.target_y, self.local_z, yaw_setpoint)

            # 실제 YOLO 타겟을 감지한 적이 있는 경우 3초 이상 상실 시 MISSION으로 복귀
            if getattr(self, 'yolo_detected_ever', False):
                time_since_last_target = time.time() - getattr(self, 'last_yolo_time', time.time())
                if time_since_last_target > 3.0:
                    self.get_logger().info("⚠️ 타겟 상실 3초 경과 -> MISSION 모드로 복귀합니다.")
                    self.current_mode = "MISSION_RETURN"

        elif self.current_mode == "MISSION_RETURN":
            self.send_mission_command()
            self.current_mode = "MISSION"

    def set_offboard_mode(self):
        """ OFFBOARD 모드 전환 명령 전송 (fallback용) """
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

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_ctrl_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z, yaw=0.0):
        msg = TrajectorySetpoint()
        msg.position = [x, y, -z] # ENU -> NED 변환 (-z)
        msg.yaw = yaw
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MissionControlNode()
    
    # 1초 주기로 check_status_loop 실행
    timer = node.create_timer(1.0, node.check_status_loop)
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
