import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import math
import os
from ultralytics import YOLO

# ==========================================
# 기존 커스텀 필터 및 그리기 함수 유지
# ==========================================
def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0: return
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), 
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), 
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes))
        cv2.line(img, start, end, color, thickness)

class ShapeAndSizeFilter:
    def __init__(self, max_lost_frames=5):
        self.max_lost_frames = max_lost_frames
        self.last_box = None
        self.last_cls = None
        self.last_conf = None
        self.lost_frames = 0

    def _is_valid_shape(self, box, frame_width, frame_height):
        w = box[2] - box[0]
        h = box[3] - box[1]
        if h <= 0: return False
        if w >= (frame_width * 0.4) or h >= (frame_height * 0.4): return False
        ratio = w / h
        return 0.8 <= ratio <= 2.7

    def process(self, detected_box, frame_width, frame_height, detected_cls=None, detected_conf=None):
        if detected_box is not None:
            if not self._is_valid_shape(detected_box, frame_width, frame_height):
                detected_box = None 

        if detected_box is None:
            self.lost_frames += 1
            if self.lost_frames >= self.max_lost_frames:
                self.last_box = None; self.last_cls = None; self.last_conf = None
            return self.last_box, self.last_cls, self.last_conf

        self.last_box = detected_box
        self.last_cls = detected_cls
        self.last_conf = detected_conf
        self.lost_frames = 0
        return detected_box, detected_cls, detected_conf

# ==========================================
# ROS 2 YOLO 노드 메인 클래스 (Gazebo 카메라 지원)
# ==========================================
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_target_node')
        
        # 1. 조종사(Guidance Node) 및 QGC 스트리머에게 보낼 퍼블리셔 생성
        self.target_pub = self.create_publisher(Point, '/yolo/target_position', 10)
        self.image_pub = self.create_publisher(Image, '/yolo/image_raw', 10)
        
        # 2. 파라미터 선언 (Gazebo 이미지 토픽 및 웹캠 모드 선택)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('use_webcam', False)
        self.image_topic = self.get_parameter('image_topic').value
        self.use_webcam = self.get_parameter('use_webcam').value
        
        self.bridge = CvBridge()
        self.latest_frame = None

        # 3. YOLO 가중치 파일 안전 경로 탐색 (여러 후보 경로 동적 탐색)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_paths = [
            os.path.join(curr_dir, "best.pt"),
            "/home/nahj183/Competition_ws/Mission2026/src/vtail_control/vtail_control/best.pt",
            os.path.expanduser("~/Competition_ws/Mission2026/src/vtail_control/vtail_control/best.pt")
        ]
        
        model_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                model_path = p
                break
                
        if model_path is None:
            model_path = "/home/nahj183/Competition_ws/Mission2026/src/vtail_control/vtail_control/best.pt"
            
        self.get_logger().info(f"YOLO 모델 불러오는 중: {model_path}")
        self.model = YOLO(model_path)
        self.tracker_filter = ShapeAndSizeFilter(max_lost_frames=5)
        
        # 4. 이미지 입력 원천 설정 (Gazebo ROS 2 토픽 vs 실제 웹캠)
        if not self.use_webcam:
            self.get_logger().info(f"📷 [Gazebo 모드] 카메라 토픽 구독 시작: {self.image_topic} (SensorDataQoS 호환 적용)")
            
            # Gazebo image_bridge는 SensorDataQoS (BEST_EFFORT)로 전송하므로 QoS 호환성 적용 필수
            self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
            self.image_sub_gz1 = self.create_subscription(Image, '/world/competition/model/rc_cessna_mono_cam_0/link/camera_link/sensor/camera/image', self.image_callback, qos_profile_sensor_data)
            self.image_sub_gz2 = self.create_subscription(Image, '/world/competition/model/gz_rc_cessna_mono_cam/link/camera_link/sensor/camera/image', self.image_callback, qos_profile_sensor_data)
            self.image_sub_gz3 = self.create_subscription(Image, '/world/competition/model/rc_cessna_mono_cam/link/camera_link/sensor/camera/image', self.image_callback, qos_profile_sensor_data)
            self.image_sub_gz4 = self.create_subscription(Image, '/world/competition/model/rc_cessna/link/camera_link/sensor/camera/image', self.image_callback, qos_profile_sensor_data)
            self.image_sub_fb1 = self.create_subscription(Image, '/camera', self.image_callback, qos_profile_sensor_data)
            self.image_sub_fb2 = self.create_subscription(Image, '/camera/image', self.image_callback, qos_profile_sensor_data)
            self.image_sub_fb3 = self.create_subscription(Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)

            # 일반 노드 호환용 (Reliable QoS 폴백)
            self.image_sub_rel1 = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
            self.image_sub_rel2 = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
            self.image_sub_rel3 = self.create_subscription(Image, '/cam1/image_raw', self.image_callback, 10)
            self.cap = None
        else:
            self.get_logger().info("📷 [웹캠 모드] /dev/video0 연결 중...")
            self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)

        self.frame_count = 0
        self.last_received_topic = None

        # 5. 약 30Hz로 프레임 처리
        self.timer = self.create_timer(0.033, self.process_frame)

    def image_callback(self, msg):
        """ Gazebo 센서 플러그인 이미지 수신 시 OpenCV 프레임으로 변환 """
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frame_count += 1
        except Exception as e:
            self.get_logger().error(f"Gazebo 이미지 변환 오류: {e}", throttle_duration_sec=2.0)

    def process_frame(self):
        # 1) Gazebo 토픽 모드일 때
        if not self.use_webcam:
            if self.latest_frame is None:
                self.get_logger().warn(
                    f"⚠️ [YOLO] 카메라 영상 토픽 수신 대기 중... (구독 토픽: {self.image_topic}, /camera, /world/...)",
                    throttle_duration_sec=3.0
                )
                return
            frame = self.latest_frame.copy()
        # 2) 웹캠 모드일 때
        else:
            if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
                return
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return

        frame_height, frame_width = frame.shape[:2]
        cam_cx = frame_width // 2
        cam_cy = frame_height // 2

        cv2.circle(frame, (cam_cx, cam_cy), 5, (255, 0, 0), -1)

        # YOLO 추론 진행
        results = self.model.predict(source=frame, conf=0.01, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            current_box = None
            current_cls = None
            current_conf = None
            
            if len(boxes) > 0:
                current_box = boxes[0].xyxy[0].cpu().numpy()
                current_cls = int(boxes[0].cls[0].item())
                current_conf = float(boxes[0].conf[0].item())
            
            filtered_box, filtered_cls, filtered_conf = self.tracker_filter.process(
                current_box, frame_width, frame_height, current_cls, current_conf
            )
            
            if filtered_box is not None:
                x1, y1, x2, y2 = map(int, filtered_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
                
                # 오차 계산
                offset_x = cx - cam_cx
                offset_y = cy - cam_cy
                
                draw_dashed_line(frame, (cam_cx, cam_cy), (cx, cy), (255, 200, 0), 2, dash_length=15)
                cv2.putText(frame, f"({offset_x}, {offset_y})", (cx + 10, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 조종사(Guidance Node)에게 좌표 전송
                msg = Point()
                msg.x = float(offset_y)  
                msg.y = float(offset_x)  
                msg.z = 0.0
                
                self.target_pub.publish(msg)

        # 터미널 실시간 동작 확인용 로그 (2초 간격)
        is_detected = ('filtered_box' in locals() and filtered_box is not None)
        status_str = f"🎯 타겟 감지 성공!" if is_detected else "👀 카메라 영상 수신 중 (타겟 탐색 중...)"
        self.get_logger().info(f"🟢 [YOLO] {status_str} (누적 수신 프레임: {self.frame_count})", throttle_duration_sec=2.0)

        # YOLO 처리 결과 프레임을 /yolo/image_raw (QGC 스트리머용)로 퍼블리시
        try:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.image_pub.publish(out_msg)
        except Exception:
            pass

        try:
            cv2.imshow('YOLO Target Tracker (Gazebo)', frame)
            cv2.waitKey(1)
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    
    if hasattr(node, 'cap') and node.cap:
        node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()