import cv2
import math
from ultralytics import YOLO

from geometry_msgs.msg import Point 



# ==========================================
# 추가된 함수: 점선 그리기
# ==========================================
def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes), 
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes), 
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes))
        cv2.line(img, start, end, color, thickness)

# ==========================================
# 커스텀 필터 클래스 (비율 및 크기 기반)
# ==========================================
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
        
        if h <= 0:
            return False
            
        # 화면 가로 또는 세로 크기의 40% 이상 시 차단
        if w >= (frame_width * 0.4) or h >= (frame_height * 0.4):
            return False

        ratio = w / h
        
        if 0.8 <= ratio <= 2.7:
            return True
        return False

    def process(self, detected_box, frame_width, frame_height, detected_cls=None, detected_conf=None):
        if detected_box is not None:
            if not self._is_valid_shape(detected_box, frame_width, frame_height):
                detected_box = None 

        if detected_box is None:
            self.lost_frames += 1
            if self.lost_frames >= self.max_lost_frames:
                self.last_box = None
                self.last_cls = None
                self.last_conf = None
            return self.last_box, self.last_cls, self.last_conf

        self.last_box = detected_box
        self.last_cls = detected_cls
        self.last_conf = detected_conf
        self.lost_frames = 0
        
        return detected_box, detected_cls, detected_conf


# ==========================================
# 1. 모델 로드 및 필터 초기화 (라즈베리파이 환경)
# ==========================================
# 📌 라즈베리파이 내부의 실제 모델 경로로 수정해주세요 (예시 경로로 작성됨)
model = YOLO("Competition_ws/src/vtail_control/vtail_control/best.pt")
tracker_filter = ShapeAndSizeFilter(max_lost_frames=5)

# 📌 라즈베리파이 카메라 안정성을 위해 V4L2 백엔드 사용
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 📌 라즈베리파이 연산 부하를 줄이기 위해 해상도 고정 (필수 권장)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

print("YOLO 객체 인식 및 거리 측정 시작. (종료하려면 화면을 클릭하고 'q'를 누르세요)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 웹캠 영상을 받자마자 흑백화한 뒤, YOLO 입력이 가능하도록 3채널(BGR) 구조로 재변환하여 frame에 덮어씁니다.
    frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    # 현재 프레임의 가로, 세로 길이 추출 및 카메라 중심점 계산
    frame_height, frame_width = frame.shape[:2]
    cam_cx = frame_width // 2
    cam_cy = frame_height // 2

    # 화면에 카메라 중심점(0,0 기준점) 표시 (선택사항, 파란색 점)
    cv2.circle(frame, (cam_cx, cam_cy), 5, (255, 0, 0), -1)

    # 추론 속도를 높이려면 conf 값을 조절하거나 verbose를 False로 유지하세요
    results = model.predict(source=frame, conf=0.01, stream=True, verbose=False)

    for r in results:
        boxes = r.boxes
        current_box = None
        current_cls = None
        current_conf = None
        
        if len(boxes) > 0:
            current_box = boxes[0].xyxy[0].cpu().numpy()
            current_cls = int(boxes[0].cls[0].item())
            current_conf = float(boxes[0].conf[0].item())
        
        # 커스텀 필터 통과 (비율/크기 검증 -> 프레임 유지)
        filtered_box, filtered_cls, filtered_conf = tracker_filter.process(
            current_box, 
            frame_width, 
            frame_height, 
            current_cls, 
            current_conf
        )
        
        # ==========================================
        # 5. 최종 시각화
        # ==========================================
        if filtered_box is not None:
            x1, y1, x2, y2 = map(int, filtered_box)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
            
            # --- 추가된 부분: 카메라 중심점과 객체 중심점 점선 연결 및 상대 좌표 표시 ---
            # 거리(상대 좌표) 계산: 카메라 중심을 (0,0)으로 가정 (우측 +x, 하단 +y)
            offset_x = cx - cam_cx
            offset_y = cy - cam_cy
            
            # 점선 그리기 (파란색 계열)
            draw_dashed_line(frame, (cam_cx, cam_cy), (cx, cy), (255, 200, 0), 2, dash_length=15)
            
            # (x, y) 좌표 텍스트 띄우기 (객체 빨간 점 약간 우측 하단에 표시)
            coord_text = f"({offset_x}, {offset_y})"
            cv2.putText(frame, coord_text, (cx + 10, cy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            # -------------------------------------------------------------

            class_name = model.names[filtered_cls] if filtered_cls is not None else "Unknown"
            conf_val = f"{filtered_conf:.2f}" if filtered_conf is not None else ""
            label = f"{class_name} {conf_val}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('YOLO Filtered WebCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()