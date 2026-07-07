import cv2
import numpy as np
import os

# ==========================================
# 1. ChArUco 보드 규격 설정
# ==========================================
squares_x = 11
squares_y = 8
square_length = 0.02
marker_length = 0.015

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, aruco_dict)

detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# ==========================================
# 2. 저장 경로 설정 (라즈베리파이 리눅스 환경에 맞춤)
# ==========================================
# 홈 디렉토리(~) 아래에 폴더 생성
target_dir = os.path.expanduser("~/cali_result")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"[안내] 저장 폴더를 생성했습니다: {target_dir}")

# ==========================================
# 3. 라즈베리파이 카메라 연결 및 데이터 초기화
# ==========================================
# 라즈베리파이 카메라 모듈 인식을 위해 V4L2 백엔드 사용
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 라즈베리파이에서 연산 부하 및 딜레이 방지를 위해 해상도 고정 (필요에 따라 변경 가능)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("에러: 카메라를 열 수 없습니다. 카메라 연결 및 권한을 확인하세요.")
    exit()

all_charuco_corners = []
all_charuco_ids = []
image_size = None

print("=== 실시간 ChArUco 캘리브레이션 ===")
print(f"📌 모든 결과물은 다음 경로에 저장됩니다:\n   {target_dir}")
print("-" * 40)
print("1. [Space바] : 현재 화면의 코너 데이터 수집 + 이미지 파일 저장")
print("2. [Enter]   : 데이터 수집 종료 및 캘리브레이션 연산 시작")
print("3. [ESC]     : 프로그램 종료 (저장 안 됨)")
print("====================================")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1] # (width, height)

    # 실시간 마커 탐지
    corners, ids, rejected = detector.detectMarkers(gray)
    display_frame = frame.copy()

    charuco_corners, charuco_ids = None, None
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
        
        # ChArUco 코너 유추
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
            cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids, (0, 255, 0))

    # 화면에 현재 캡처된 장수 표시
    cv2.putText(display_frame, f"Saved: {len(all_charuco_corners)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Raspberry Pi Camera Calibration", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Space바: 데이터 누적 및 이미지 파일 저장
    if key == ord(' '):
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            
            # 캡처된 원본 이미지를 지정 폴더에 저장
            img_name = f"cap_{len(all_charuco_corners)-1}.jpg"
            img_save_path = os.path.join(target_dir, img_name)
            cv2.imwrite(img_save_path, frame)
            
            print(f"[성공] {img_name} 저장 및 데이터 수집 완료 (현재 총 {len(all_charuco_corners)}장)")
        else:
            print("[실패] 보드가 제대로 인식되지 않았습니다. 각도를 조절해 주세요.")
            
    # Enter: 수집 종료 및 연산 시작
    elif key == 13:
        break
        
    # ESC: 취소 후 종료
    elif key == 27:
        print("캘리브레이션을 취소합니다.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 4. 카메라 캘리브레이션 수행
# ==========================================
if len(all_charuco_corners) < 5:
    print("\n[에러] 데이터가 너무 부족합니다. 최소 5장 이상 캡처해야 합니다. (10~20장 권장)")
    exit()

print("\n캘리브레이션 연산 중... 잠시만 기다려주세요.")

retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    charucoCorners=all_charuco_corners,
    charucoIds=all_charuco_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None
)

print(f"\n[완료] 재투영 오차(RMS Error): {retval:.4f} px")

# ==========================================
# 5. 지정 폴더에 YAML 파일 저장
# ==========================================
save_path = os.path.join(target_dir, "calibration_result.yaml")
cv_file = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)

cv_file.write("camera_matrix", camera_matrix)
cv_file.write("dist_coeffs", dist_coeffs)
cv_file.write("image_width", image_size[0])
cv_file.write("image_height", image_size[1])
cv_file.release()

print(f"\n✅ 캘리브레이션 결과가 성공적으로 저장되었습니다.")
print(f"📂 저장 경로: {save_path}")
print("\n=== Camera Matrix ===")
print(camera_matrix)
print("=== Distortion Coefficients ===")
print(dist_coeffs)