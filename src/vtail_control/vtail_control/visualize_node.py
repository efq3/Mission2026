import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from px4_msgs.msg import VehicleOdometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class Px4ToRviz(Node):
    def __init__(self):
        super().__init__('px4_to_rviz')
        self.sub = self.create_subscription(
            VehicleOdometry, 
            '/fmu/out/vehicle_odometry', 
            self.odom_cb, 
            qos_profile_sensor_data)
        self.pub = self.create_publisher(Path, '/drone_path', 10)
        self.path = Path()

    def odom_cb(self, msg):
        self.path.header.frame_id = 'map'
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        
        pose.pose.position.x = float(msg.position[1])
        pose.pose.position.y = float(msg.position[0])
        pose.pose.position.z = float(-msg.position[2])
        
        self.path.poses.append(pose)
        self.pub.publish(self.path)

rclpy.init()
rclpy.spin(Px4ToRviz())
