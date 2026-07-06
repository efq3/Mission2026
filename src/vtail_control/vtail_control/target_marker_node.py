import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

class TargetMarkerNode(Node):
    def __init__(self):
        super().__init__('target_marker_node')
        
        # 1. 타겟 좌표 수신 (YOLO에서 오는 데이터)
        self.sub = self.create_subscription(
            Point, 
            '/yolo/target_position', 
            self.target_callback, 
            10
        )
        
        # 2. RViz2용 마커 발행
        self.marker_pub = self.create_publisher(
            Marker, 
            '/visualization_marker', 
            10
        )

    def target_callback(self, msg):
        marker = Marker()
        
        # 기준 좌표계 설정 (RViz2의 Fixed Frame 이름과 일치해야 함)
        # 보통 PX4 글로벌 환경에서는 "map" 또는 "odom"을 사용합니다.
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        
        # 마커의 네임스페이스와 ID (여러 개의 마커를 띄울 때 구분용)
        marker.ns = "yolo_target"
        marker.id = 0
        
        # 마커 모양 (SPHERE: 구, CUBE: 정육면체, CYLINDER: 원기둥 등)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # 마커 위치 (Point 토픽에서 받은 x, y, z 대입)
        marker.pose.position.x = msg.x
        marker.pose.position.y = msg.y
        marker.pose.position.z = msg.z
        
        # 마커 회전값 (기본 수평 상태)
        marker.pose.orientation.w = 1.0
        
        # 마커 크기 (x, y, z 각각 2.0 미터 크기의 거대한 공)
        marker.scale.x = 1.0
        marker.scale.y = 3.0
        marker.scale.z = 2.5
        
        # 마커 색상 (r: 빨강, g: 초록, b: 파랑 / 0.0 ~ 1.0 사이 값)
        # a는 투명도 (1.0 = 불투명)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0 
        
        # 영원히 띄워둠 (특정 시간 뒤에 사라지게 하려면 sec 값을 줌)
        marker.lifetime.sec = 0
        
        # 퍼블리시!
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = TargetMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()