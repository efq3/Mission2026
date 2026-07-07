import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import cv2
import math
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
# ROS 2 YOLO 노드 메인 클래스
# ==========================================
class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_target_node')
        
        # 1. 조종사(Guidance Node)에게 보낼 퍼블리셔 생성
        self.target_pub = self.create_publisher(Point, '/yolo/target_position', 10)
        
        # 2. YOLO 및 카메라 세팅
        self.model = YOLO("Competition_ws/src/vtail_control/vtail_control/best.pt")
        self.tracker_filter = ShapeAndSizeFilter(max_lost_frames=5)
        
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다")
            return
            
        self.get_logger().info("YOLO 노드 가동 시작, 타겟 탐색 중...")

        # 3. while 루프 대신 ROS 타이머 사용 (약 30Hz로 카메라 프레임 읽기)
        self.timer = self.create_timer(0.033, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 흑백화 후 BGR 재변환
        frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        frame_height, frame_width = frame.shape[:2]
        cam_cx = frame_width // 2
        cam_cy = frame_height // 2

        cv2.circle(frame, (cam_cx, cam_cy), 5, (255, 0, 0), -1)

        # YOLO 추론
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

                # ==========================================
                # 조종사(Guidance Node)에게 좌표 쏘기!
                # ==========================================
                msg = Point()
                # Guidance 노드의 변수 매칭:
                # msg.x -> self.target_z_dist (전방 거리: 임시로 y축 오차 사용)
                # msg.y -> self.target_y_rel (좌우 오차: x축 오차 사용)
                msg.x = float(offset_y)  
                msg.y = float(offset_x)  
                msg.z = 0.0
                
                self.target_pub.publish(msg)

        cv2.imshow('YOLO Target Tracker', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    
    # 노드 종료 시 카메라 자원 해제
    node.cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()