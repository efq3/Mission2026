import rclpy
from rclpy.node import Node
import time

from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint
from geometry_msgs.msg import Point 

class GuidanceNode(Node):
    def __init__(self):
        super().__init__('vtail_guidance_node')

        # 1. 제어 명령 발행 (Publish)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.rates_pub = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', 10)

        # 2. 데이터 수신 (Subscribe)
        self.target_sub = self.create_subscription(Point, '/yolo/target_position', self.target_callback, 10)
        
        # 3. 제어 루프 타이머 (50Hz = 0.02초 주기)
        self.timer = self.create_timer(0.02, self.control_loop)

        # 상태 변수 초기화
        self.target_y_rel = 0.0  
        self.target_z_dist = 0.0 
        self.last_rx_time = time.time()
        
        # 비행 파라미터 
        self.v_cruise = 17.0     
        self.v_y = 0.0           
        
        # Low-Pass Filter 변수
        self.alpha = 0.15        
        self.filtered_yaw_rate = 0.0
        
        self.get_logger().info("🚀 순수 유도 제어(Guidance) 노드 시작됨. (명령 대기 중)")

    def target_callback(self, msg):
        """ YOLO 데이터 수신 시 호출되는 함수 """
        self.target_z_dist = msg.x  
        self.target_y_rel = msg.y   
        self.last_rx_time = time.time()

    def control_loop(self):
        """ 50Hz마다 실행되는 메인 제어 루프 """
        
        # 중요: MISSION 모드일 때도 계속 보내둬야, Mode Node가 OFFBOARD로 전환 시 PX4가 거부하지 않음!
        self.publish_offboard_control_mode()

        current_time = time.time()
        yaw_rate_cmd = 0.0
        time_since_last_target = current_time - self.last_rx_time

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

def main(args=None):
    rclpy.init(args=args)
    node = GuidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()