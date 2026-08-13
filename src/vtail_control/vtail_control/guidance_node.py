import rclpy
from rclpy.node import Node
import time
import math

from px4_msgs.msg import OffboardControlMode, VehicleAttitudeSetpoint, VehicleAttitude
from geometry_msgs.msg import Point 
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

class GuidanceNode(Node):
    def __init__(self):
        super().__init__('vtail_guidance_node')

        self.px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 1. 제어 명령 발행 (Publish)
        self.offboard_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.att_sp_pub = self.create_publisher(VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', 10)

        # 2. 데이터 수신 (Subscribe)
        self.target_sub = self.create_subscription(Point, '/yolo/target_position', self.target_callback, 10)
        self.att_sub = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_callback, self.px4_qos)
        
        # 3. 제어 루프 타이머 (50Hz = 0.02초 주기)
        self.timer = self.create_timer(0.02, self.control_loop)

        # 상태 변수 초기화
        self.target_y_rel = 0.0  
        self.target_z_dist = 0.0 
        self.last_rx_time = time.time()
        self.current_yaw = 0.0
        
        # 비행 파라미터
        self.v_cruise = 17.0     
        self.v_y = 0.0           
        self.g = 9.81
        
        # Low-Pass Filter 변수
        self.alpha = 0.15        
        self.filtered_phi_cmd = 0.0
        
        self.get_logger().info("🚀 BTT(Bank-To-Turn) 유도 제어(Guidance) 노드 시작됨. (명령 대기 중)")

    def att_callback(self, msg):
        """ 현재 기체 자세(Yaw) 수신 """
        q = msg.q
        # 쿼터니언을 오일러 Yaw로 변환
        self.current_yaw = math.atan2(2.0 * (q[0]*q[3] + q[1]*q[2]), 1.0 - 2.0 * (q[2]*q[2] + q[3]*q[3]))

    def target_callback(self, msg):
        """ YOLO 데이터 수신 시 호출되는 함수 """
        self.target_z_dist = msg.x  
        self.target_y_rel = msg.y   
        self.last_rx_time = time.time()

    def euler_to_quaternion(self, roll, pitch, yaw):
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        q = [0.0]*4
        q[0] = cr * cp * cy + sr * sp * sy
        q[1] = sr * cp * cy - cr * sp * sy
        q[2] = cr * sp * cy + sr * cp * sy
        q[3] = cr * cp * sy - sr * sp * cy
        return q

    def control_loop(self):
        """ 50Hz마다 실행되는 메인 제어 루프 """
        self.publish_offboard_control_mode()

        current_time = time.time()
        phi_cmd = 0.0
        time_since_last_target = current_time - self.last_rx_time

        # ---------------------------------------------------------
        # [PNG 기반 BTT 유도 제어 계산]
        # ---------------------------------------------------------
        if time_since_last_target > 1.0:
            # 타겟 데이터를 1초 이상 못 받은 경우: 수평(Roll=0) 유지
            phi_cmd = 0.0
        elif self.target_z_dist > 1.0:  
            # 1. 충돌 예상 시간 t_go 및 횡방향 상대오차 Y_ref - Y_actual을 바탕으로 필요한 조향가속도 계산
            t_go = self.target_z_dist / self.v_cruise
            a_y = 2.0 * (self.target_y_rel - (self.v_y * t_go)) / (t_go ** 2)

            # 2. Bank-to-Turn (BTT) 변환: 요구 가속도(ay,cmd) -> 목표 롤 각도(phi_cmd)
            phi_cmd = math.atan(a_y / self.g)

            # 롤 각도 제한 (예: 최대 45도)
            max_roll = 0.785
            phi_cmd = max(min(phi_cmd, max_roll), -max_roll)

        # LPF 필터링
        self.filtered_phi_cmd = (self.alpha * phi_cmd) + ((1.0 - self.alpha) * self.filtered_phi_cmd)

        # 3. 픽스호크 PID 제어를 위한 자세 명령 발행
        # 픽스호크의 fw_att_control이 목표 롤(phi_cmd)을 추종하며 에일러론을 조작하고,
        # 동시에 측미끄러짐 각도 베타를 0으로 유지하는 Coordinated Turn(러더 워시아웃 필터 제어)을 병행함.
        self.publish_vehicle_attitude_setpoint(self.filtered_phi_cmd, 0.0, self.current_yaw)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = True      # 자세 제어 모드 활성화 (Roll 각도 명령용)
        msg.body_rate = False    
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_vehicle_attitude_setpoint(self, roll, pitch, yaw):
        msg = VehicleAttitudeSetpoint()
        msg.q_d = self.euler_to_quaternion(roll, pitch, yaw)
        msg.yaw_sp_move_rate = 0.0
        
        # 고정익의 Offboard 자세 제어에서는 thrust_body[0]에 추력(스로틀)을 넣음
        msg.thrust_body = [0.5, 0.0, 0.0]  
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.att_sp_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = GuidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
