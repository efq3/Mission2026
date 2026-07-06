import rclpy
from rclpy.node import Node
import time

from px4_msgs.msg import OffboardControlMode, VehicleRatesSetpoint, VehicleCommand
from geometry_msgs.msg import Point 

class GuidanceNode(Node):
    def __init__(self):
        super().__init__('vtail_guidance_node')

        # 1. 제어 명령 발행 (Publish)
        # PX4에 오프보드 모드임을 알리는 토픽
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        # PX4에 목표 각속도를 전달하는 토픽
        self.rates_pub = self.create_publisher(VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', 10)

        # 2. 데이터 수신 (Subscribe)
        # YOLO Pose를 처리한 노드로부터 타겟의 3D 상대 위치를 받아옴
        self.target_sub = self.create_subscription(Point, '/yolo/target_position', self.target_callback, 10)

        # 3. 제어 루프 타이머 (50Hz = 0.02초 주기)
        self.timer = self.create_timer(0.02, self.control_loop)

        # 상태 변수 초기화
        self.target_y_rel = 0.0  # 타겟의 측면 오차 (미터)
        self.target_z_dist = 0.0 # 타겟까지의 전방 거리 (미터)
        self.last_rx_time = time.time()
        
        # 비행 파라미터 (기체에 맞게 튜닝 필요)
        self.v_cruise = 17.0     # 순항 속도 (m/s)
        self.v_y = 0.0           # 기체의 현재 측면 속도 (단순화를 위해 일단 0으로 세팅)
        
        # Low-Pass Filter (노이즈 제거용) 변수
        self.alpha = 0.15        # 필터 계수 (0~1. 작을수록 덜덜거림이 줄지만 반응이 느려짐)
        self.filtered_yaw_rate = 0.0

        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)

    def target_callback(self, msg):
        """ YOLO 데이터 수신 시 호출되는 함수 """
        self.target_z_dist = msg.x  # Z: 전방 거리
        self.target_y_rel = msg.y   # Y: 측면 오차
        self.last_rx_time = time.time()

    def control_loop(self):
        """ 50Hz마다 실행되는 메인 제어 루프 """
        # PX4가 오프보드 모드를 풀지 않도록 지속적으로 신호(하트비트)를 보내야 함
        self.publish_offboard_control_mode()

        current_time = time.time()
        yaw_rate_cmd = 0.0

        # [예외 처리] 타겟 데이터를 1초 이상 못 받은 경우 (Target Lost)
        if (current_time - self.last_rx_time) > 1.0:
            # 타겟을 놓쳤으므로 회전을 멈추고 직진 유지
            yaw_rate_cmd = 0.0
        
        # [유도 제어] 칠판 수식 계산
        elif self.target_z_dist > 1.0:  # 너무 가까워서 0으로 나누는 에러 방지
            # 1. 남은 시간 (t_go) 계산
            t_go = self.target_z_dist / self.v_cruise
            
            # 2. 측면 가속도 (a_y) 계산: a_y = 2 * (y_rel - v_y * t_go) / t_go^2
            a_y = 2.0 * (self.target_y_rel - (self.v_y * t_go)) / (t_go ** 2)
            
            # 3. 목표 요 각속도 계산: rate = a_y / V_cruise
            yaw_rate_cmd = a_y / self.v_cruise

            # 기체가 감당할 수 있는 최대 회전 속도 제한 (예: 최대 45 deg/s = 0.785 rad/s)
            max_rate = 0.785
            yaw_rate_cmd = max(min(yaw_rate_cmd, max_rate), -max_rate)

        # [필터링] YOLO의 덜덜거리는 노이즈를 부드럽게 깎아줌
        self.filtered_yaw_rate = (self.alpha * yaw_rate_cmd) + ((1.0 - self.alpha) * self.filtered_yaw_rate)

        # 계산된 최종 요 각속도를 PX4로 전송
        self.publish_vehicle_rates(self.filtered_yaw_rate)

    def publish_offboard_control_mode(self):
        """ PX4에 Body Rate(각속도) 제어를 하겠다고 알림 """
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True  # Rate 제어 활성화
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_vehicle_rates(self, yaw_rate):
        """ 실제 서보모터를 움직일 각속도 명령 전송 """
        msg = VehicleRatesSetpoint()
        msg.roll = yaw_rate * 0.5
        msg.pitch = 1.0
        msg.yaw = yaw_rate
        
        # ⭐️ PX4 필수 요구사항: 추력은 [x, y, z] 배열입니다. 고정익 전진 추력은 x(0번째)입니다.
        msg.thrust_body = [0.5, 0.0, 0.0]  # 0.5 = 엔진 50% 파워 
        
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.rates_pub.publish(msg)
    
    def set_offboard_mode(self):
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = 1.0 # custom mode
        msg.param2 = 6.0 # offboard: 6.0, Auto: 4.0
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