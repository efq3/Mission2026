#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class QGCStreamer(Node):
    def __init__(self):
        super().__init__('qgc_streamer')
        
        # YOLO 인식 결과(바운딩 박스 포함)가 들어오는 ROS 2 토픽 구독 (QoS 호환성 보장)
        self.sub = self.create_subscription(Image, '/yolo/image_raw', self.image_callback, qos_profile_sensor_data)
        self.sub_rel = self.create_subscription(Image, '/yolo/image_raw', self.image_callback, 10)
        self.bridge = CvBridge()
        
        # ★여기를 반드시 실제 맥북 IP로 변경하세요!★
        mac_ip = "100.110.78.107".strip() 
        port = 5600 # QGC 기본 비디오 포트
        
        # GStreamer H.264 UDP 스트리밍 파이프라인 (지연시간 최소화 세팅)
        pipeline = (
            f"appsrc ! videoconvert ! video/x-raw, format=I420 ! "
            f"x264enc tune=zerolatency bitrate=1000 speed-preset=ultrafast ! "
            f"rtph264pay config-interval=1 pt=96 ! "
            f"udpsink host={mac_ip} port={port} sync=false"
        )
        
        self.get_logger().info(f"맥북({mac_ip}:{port})으로 QGC 영상 스트리밍 준비 중...")
        
        # 해상도 640x480, 30fps로 전송 (네트워크 부하 방지)
        self.out = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (640, 480))
        if not self.out.isOpened():
            self.get_logger().error(f"⚠️ GStreamer 비디오 라이터 생성 실패! 맥북 IP({mac_ip}) 및 GStreamer 환경을 확인하세요.")
        else:
            self.get_logger().info(f"✅ 맥북({mac_ip}:{port}) QGC 비디오 송출이 성공적으로 연결되었습니다.")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # VideoWriter 설정과 해상도를 무조건 일치시켜야 에러가 안 납니다.
            cv_image = cv2.resize(cv_image, (640, 480))
            self.out.write(cv_image)
        except Exception as e:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = QGCStreamer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
