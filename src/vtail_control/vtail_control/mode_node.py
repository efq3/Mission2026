import rclpy
from rclpy.node import Node
import time

from px4_msgs.msg import MissionResult, VehicleCommand
from geometry_msgs.msg import Point  # ⭐️ YOLO 데이터를 받기 위해 추가

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')
        
        # 1. 미션 진행 상태 수신
        self.mission_sub = self.create_subscription(
            MissionResult, 
            '/fmu/out/mission_result', 
            self.mission_callback, 
            10
        )
        
        # 2. 사령관도 복귀 타이밍을 재기 위해 YOLO 데이터를 듣습니다.
        self.yolo_sub = self.create_subscription(
            Point, 
            '/yolo/target_position', 
            self.yolo_callback, 
            10
        )
        
        # 명령 퍼블리셔
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        
        # 3. 상태 감시용 타이머 (1초에 10번씩 상태를 체크합니다)
        self.timer = self.create_timer(0.1, self.check_status_loop)
        
        # 변수 초기화
        self.current_mode = "MISSION" # 초기값을 MISSION으로 변경
        self.last_yolo_time = time.time()
        
        self.get_logger().info("Mode 노드 시작됨. WP2 도달 대기 중...")

    def yolo_callback(self, msg):
        """ YOLO 데이터가 들어올 때마다 시간 갱신 """
        self.last_yolo_time = time.time()

    def mission_callback(self, msg):
        """ WP 진입 확인 후 OFFBOARD 탈취 """
        reached_wp = msg.seq_reached
        
        if self.current_mode == "MISSION" and reached_wp == 2:
            self.get_logger().info(f"WP2 도달 -> Guidance에게 제어권을 넘깁니다 (OFFBOARD).")
            self.set_mode(6.0) # OFFBOARD
            self.current_mode = "OFFBOARD"

    def check_status_loop(self):
        """ 주기적으로 상태를 검사하여 복귀(Return) 여부를 결정 """
        # 오프보드 모드인데, YOLO 데이터를 못 받은 지 3초가 넘었다면?
        if self.current_mode == "OFFBOARD":
            time_since_last_target = time.time() - self.last_yolo_time
            
            if time_since_last_target > 3.0:
                self.get_logger().info(f"타겟 상실 3초 경과 작전 종료 -> MISSION으로 복귀시킵니다.")
                self.set_mode(4.0) # AUTO.MISSION
                self.current_mode = "MISSION_RETURN" # 중복 명령 방지

    def set_mode(self, mode_param):
        """ 모드 변경 명령 전송 """
        msg = VehicleCommand()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.param1 = 1.0  
        msg.param2 = mode_param
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