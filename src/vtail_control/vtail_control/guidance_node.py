import rclpy
from rclpy.node import Node
import time

# ⭐️ MissionResult 추가됨
from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand, MissionResult
from geometry_msgs.msg import Point 

class GuidanceNode(Node):
    def __init__(self):
        super().__init__('vtail_guidance_node')

        # 1. 제어 명령 발행 (Publish)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.rates_pub = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', 10)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

        # 2. 데이터 수신 (Subscribe)
        self.target_sub = self.create_subscription(Point, '/yolo/target_position', self.target_callback, 10)
        
        # ⭐️ 3. PX4 미션 진행 상태 수신 (MissionResult) 추가
        self.mission_sub = self.create_subscription(MissionResult, '/fmu/out/mission_result', self.mission_callback, 10)

        # 4. 제어 루프 타이머 (50Hz = 0.02초 주기)
        self.timer = self.create_timer(0.02, self.control_loop)

        # 상태 변수 초기화
        self.current_mode = "MISSION"  # ⭐️ 현재 기체 모드 상태 저장용
        self.target_y_rel = 0.0  
        self.target_z_dist = 0.0 
        self.last_rx_time = time.time()
        
        # 비행 파라미터 
        self.v_cruise = 17.0     
        self.v_y = 0.0           
        
        # Low-Pass Filter 변수
        self.alpha = 0.15        
        self.filtered_yaw_rate = 0.0
        
        self.get_logger().info("🚀 유도 제어(Guidance) 및 임무 제어 노드 시작됨. (대기 중)")

    def mission_callback(self, msg):
        """ ⭐️ WP 도달 상태를 확인하고 모드를 전환하는 함수 """
        reached_wp = msg.seq_reached
        
        # 2번 WP(작전 구역)에 도착했고, 현재 MISSION 모드라면? -> 오프보드 탈취!
        if self.current_mode == "MISSION" and reached_wp == 2:
            self.get_logger().info("🎯 작전 구역(WP2) 도달 감지! -> OFFBOARD 제어권을 탈취합니다!")
            self.set_mode(6.0) # 6.0 = OFFBOARD
            self.current_mode = "OFFBOARD"

    def target_callback(self, msg):
        """ YOLO 데이터 수신 시 호출되는 함수 """
        self.target_z_dist = msg.x  
        self.target_y_rel = msg.y   
        self.last_rx_time = time.time()

    def control_loop(self):
        """ 50Hz마다 실행되는 메인 제어 루프 """
        
        # ⭐️ 중요: MISSION 모드일 때도 계속 보내둬야, 나중에 OFFBOARD 전환 시 PX4가 거부하지 않음!
        self.publish_offboard_control_mode()

        current_time = time.time()
        yaw_rate_cmd = 0.0
        time_since_last_target = current_time - self.last_rx_time

        # ---------------------------------------------------------
        # ⭐️ 오프보드 종료 조건 (미션 복귀)
        # 오프보드 모드인데, 타겟을 3초 이상 놓쳤다면 작전 종료로 간주하고 복귀
        # ---------------------------------------------------------
        if self.current_mode == "OFFBOARD" and time_since_last_target > 3.0:
            self.get_logger().info("🏁 타겟 상실 (3초 경과) 또는 임무 완료 -> 원래 미션으로 복귀합니다!")
            self.set_mode(4.0) # 4.0 = AUTO.MISSION
            self.current_mode = "MISSION"

        # ---------------------------------------------------------
        # [유도 제어 계산 파트]
        # ---------------------------------------------------------
        if time_since_last_target > 1.0:
            # 타겟 데이터를 1초 이상 못 받은 경우: 직진 유지
            yaw_rate_cmd = 0.0
        
        elif self.target_z_dist > 1.0:  
            t_go = self.target_z_dist / self.v_cruise
            a_y = 2.0 * (self.target_y_rel - (self.v_y * t_go)) / (t_go ** 2)
            yaw_rate_cmd = a_y / self.v_cruise

            max_rate = 0.785
            yaw_rate_cmd = max(min(yaw_rate_cmd, max_rate), -max_rate)

        # 필터링 및 실제 명령 전송 (이 명령은 기체가 OFFBOARD일 때만 실제로 기체를 움직입니다)
        self.filtered_yaw_rate = (self.alpha * yaw_rate_cmd) + ((1.0 - self.alpha) * self.filtered_yaw_rate)
        self.publish_vehicle_rates(self.filtered_yaw_rate)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True  
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_vehicle_rates(self, yaw_rate):
        msg = VehicleRatesSetpoint()
        msg.roll = yaw_rate * 0.5
        msg.pitch = 1.0
        msg.yaw = yaw_rate
        msg.thrust_body = [0.5, 0.0, 0.0]  
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_pub.publish(msg)
    
    def set_mode(self, mode_param):
        """ ⭐️ 파라미터를 받아 MISSION(4.0) 또는 OFFBOARD(6.0)로 전환 """
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1     # ⭐️ 추가됨 (안전한 명령 전송을 위해 시스템 ID 명시)
        msg.target_component = 1  # ⭐️ 추가됨
        msg.source_system = 1     # ⭐️ 추가됨
        msg.source_component = 1  # ⭐️ 추가됨
        msg.from_external = True  # ⭐️ 추가됨
        msg.param1 = 1.0 
        msg.param2 = mode_param 
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.command_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GuidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()