import rclpy
from rclpy.node import Node
from px4_msgs.msg import MissionResult, VehicleCommand

class MissionControlNode(Node):
    def __init__(self):
        super().__init__('mission_control_node')
        
        # 1. PX4의 미션 진행 상태 수신 (몇 번 WP에 도달했는지)
        self.mission_sub = self.create_subscription(
            MissionResult, 
            '/fmu/out/mission_result', 
            self.mission_callback, 
            10
        )
        
        # 모드 변경 명령 퍼블리셔
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        
        # 초기 모드 (이륙 및 미션 시작은 QGC나 수동으로 했다고 가정)
        self.current_mode = "MISSION" 
        
        self.get_logger().info("🚀 MissionResult 기반 제어 노드 시작됨.")

    def mission_callback(self, msg):
        # msg.seq_reached: 가장 최근에 도달한 Waypoint의 번호 (QGC 기준)
        # 참고: 이륙(Takeoff)이 0번이거나 1번일 수 있으므로, 
        # 실제 QGC 미션 리스트의 번호를 확인하고 숫자를 맞추셔야 합니다.
        reached_wp = msg.seq_reached
        
        # -----------------------------------------------------
        # 시나리오: WP2 도달 시 -> OFFBOARD 모드 탈취
        # -----------------------------------------------------
        if self.current_mode == "MISSION" and reached_wp == 2:
            self.get_logger().info(f"🎯 작전 구역(WP2) 도달 확인! -> OFFBOARD 모드로 전환합니다.")
            
            # ⚠️ 여기서 OFFBOARD로 넘어가기 전, 타겟 추적 명령(Setpoint)을 먼저 쏴야 합니다!
            self.set_mode(6.0) # OFFBOARD
            self.current_mode = "OFFBOARD"

        # -----------------------------------------------------
        # 시나리오: OFFBOARD 종료 및 미션 복귀
        # (미션이 정지되었으므로 seq_reached 대신 다른 조건 필요)
        # -----------------------------------------------------
        elif self.current_mode == "OFFBOARD":
            # 예시: YOLO 타겟을 폭하했거나, 추적이 끝났다는 변수(is_target_cleared)가 True일 때
            is_target_cleared = False # 실제로는 YOLO 처리 로직에서 받아와야 합니다.
            
            if is_target_cleared:
                self.get_logger().info(f"🏁 임무 완료! -> 원래 경로(MISSION)로 복귀합니다.")
                self.set_mode(4.0) # AUTO.MISSION
                self.current_mode = "MISSION_RETURN"

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