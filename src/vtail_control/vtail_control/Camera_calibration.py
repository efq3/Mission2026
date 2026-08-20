import cv2
import numpy as np
import os
import sys
import time
import select
import termios
import tty

# ============================================================
# 1. ChArUco 보드 규격
# ============================================================
squares_x = 11
squares_y = 8
square_length = 0.02
marker_length = 0.015

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

board = cv2.aruco.CharucoBoard(
    (squares_x, squares_y),
    square_length,
    marker_length,
    aruco_dict
)
board.setLegacyPattern(True)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ============================================================
# 2. 결과 저장 경로
#    이 파일이 있는 폴더 기준으로 cali_result 생성
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(script_dir, "cali_result")
os.makedirs(target_dir, exist_ok=True)

preview_path = os.path.join(target_dir, "live_preview.jpg")

# ============================================================
# 3. 카메라 설정
# ============================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print(f"[ERROR] /dev/video{CAMERA_INDEX} 카메라를 열 수 없습니다.")
    print("컨테이너 안에서 다음을 확인하세요:")
    print("  ls -l /dev/video*")
    sys.exit(1)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

print("============================================================")
print(" Raspberry Pi / Docker Headless ChArUco Calibration")
print("============================================================")
print(f"[CAMERA] /dev/video{CAMERA_INDEX}")
print(f"[SIZE]   {actual_width} x {actual_height}")
print(f"[FPS]    {actual_fps:.2f}")
print(f"[SAVE]   {target_dir}")
print(f"[PREVIEW] {preview_path}")
print()
print("키 조작 (Enter 필요 없음)")
print("  c 또는 Space : 현재 프레임 저장")
print("  q            : 촬영 종료 → 캘리브레이션 시작")
print("  x            : 취소하고 종료")
print()
print("보드가 인식되면 [DETECTED] 로 표시됩니다.")
print("============================================================")

# ============================================================
# 4. 캘리브레이션 데이터
# ============================================================
all_charuco_corners = []
all_charuco_ids = []
image_size = None

latest_frame = None
latest_charuco_corners = None
latest_charuco_ids = None
latest_marker_count = 0
latest_corner_count = 0
latest_board_ok = False

STATUS_INTERVAL = 0.25
PREVIEW_INTERVAL = 1.0

last_status_time = 0.0
last_preview_time = 0.0

# ============================================================
# 5. 터미널을 1키 즉시 입력 모드로 변경
# ============================================================
if not sys.stdin.isatty():
    cap.release()
    print("[ERROR] 현재 stdin이 TTY 터미널이 아닙니다.")
    print("SSH 또는 VS Code Terminal에서 직접 실행하세요.")
    sys.exit(1)

stdin_fd = sys.stdin.fileno()
old_terminal_settings = termios.tcgetattr(stdin_fd)

cancelled = False

try:
    # Enter 없이 한 글자씩 즉시 읽기
    tty.setcbreak(stdin_fd)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("\n[ERROR] 카메라 프레임을 가져올 수 없습니다.")
            break

        latest_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if image_size is None:
            image_size = gray.shape[::-1]

        (
            charuco_corners,
            charuco_ids,
            marker_corners,
            marker_ids
        ) = charuco_detector.detectBoard(gray)

        marker_count = 0 if marker_ids is None else len(marker_ids)
        corner_count = 0 if charuco_ids is None else len(charuco_ids)

        board_ok = (
            charuco_corners is not None
            and charuco_ids is not None
            and len(charuco_corners) > 3
        )

        latest_charuco_corners = charuco_corners
        latest_charuco_ids = charuco_ids
        latest_marker_count = marker_count
        latest_corner_count = corner_count
        latest_board_ok = board_ok

        # ----------------------------------------------------
        # 최신 상태 이미지 저장
        # ----------------------------------------------------
        preview_frame = frame.copy()

        if marker_ids is not None and marker_count > 0:
            cv2.aruco.drawDetectedMarkers(
                preview_frame,
                marker_corners,
                marker_ids
            )

        if board_ok:
            cv2.aruco.drawDetectedCornersCharuco(
                preview_frame,
                charuco_corners,
                charuco_ids,
                (0, 255, 0)
            )

        state_text = "DETECTED" if board_ok else "NOT DETECTED"

        cv2.putText(
            preview_frame,
            (
                f"{state_text} | markers:{marker_count} | "
                f"corners:{corner_count} | saved:{len(all_charuco_corners)}"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if board_ok else (0, 0, 255),
            2
        )

        now = time.monotonic()

        if now - last_preview_time >= PREVIEW_INTERVAL:
            cv2.imwrite(preview_path, preview_frame)
            last_preview_time = now

        # ----------------------------------------------------
        # 터미널 실시간 상태 출력
        # ----------------------------------------------------
        if now - last_status_time >= STATUS_INTERVAL:
            state = "DETECTED    " if board_ok else "NOT DETECTED"

            print(
                f"\r[{state}] "
                f"markers={marker_count:2d} | "
                f"corners={corner_count:2d} | "
                f"saved={len(all_charuco_corners):2d} "
                f"| c/Space=save, q=finish, x=cancel ",
                end="",
                flush=True
            )

            last_status_time = now

        # ----------------------------------------------------
        # Enter 없이 키 1개 즉시 입력
        # ----------------------------------------------------
        readable, _, _ = select.select([sys.stdin], [], [], 0)

        if readable:
            key = sys.stdin.read(1).lower()

            # c 또는 Space -> 저장
            if key == "c" or key == " ":
                print()

                if latest_board_ok:
                    all_charuco_corners.append(
                        latest_charuco_corners.copy()
                    )
                    all_charuco_ids.append(
                        latest_charuco_ids.copy()
                    )

                    img_name = (
                        f"cap_{len(all_charuco_corners) - 1:03d}.jpg"
                    )
                    img_save_path = os.path.join(
                        target_dir,
                        img_name
                    )

                    cv2.imwrite(
                        img_save_path,
                        latest_frame
                    )

                    print(
                        f"[CAPTURE OK] {img_name} 저장 완료 "
                        f"| corners={latest_corner_count} "
                        f"| 총 {len(all_charuco_corners)}장"
                    )

                else:
                    print(
                        "[CAPTURE FAIL] 현재 ChArUco 보드가 "
                        "충분히 인식되지 않았습니다."
                    )

            # q -> 촬영 종료 및 캘리브레이션
            elif key == "q":
                print()
                print(
                    "[INFO] 데이터 수집 종료 → "
                    "캘리브레이션을 시작합니다."
                )
                break

            # x -> 완전 취소
            elif key == "x":
                print()
                print("[INFO] 캘리브레이션을 취소합니다.")
                cancelled = True
                break

except KeyboardInterrupt:
    print("\n[INFO] Ctrl+C 입력. 프로그램을 종료합니다.")
    cancelled = True

finally:
    # 어떤 경우든 터미널 입력 상태를 원래대로 복구
    termios.tcsetattr(
        stdin_fd,
        termios.TCSADRAIN,
        old_terminal_settings
    )

    cap.release()

# ============================================================
# 취소한 경우 여기서 종료
# ============================================================
if cancelled:
    sys.exit(0)

# ============================================================
# 6. 캘리브레이션 데이터 개수 확인
# ============================================================
if len(all_charuco_corners) < 5:
    print(
        f"\n[ERROR] 캡처 데이터가 부족합니다. "
        f"현재 {len(all_charuco_corners)}장 / 최소 5장 필요"
    )
    sys.exit(1)

print(
    "\n[INFO] 캘리브레이션 연산 중... "
    "(불량 프레임 검사 포함)"
)

# ============================================================
# 7. 유효 데이터 필터링
# ============================================================
valid_obj_points = []
valid_img_points = []

for i in range(len(all_charuco_corners)):

    objp, imgp = board.matchImagePoints(
        all_charuco_corners[i],
        all_charuco_ids[i]
    )

    if (
        objp is not None
        and imgp is not None
        and len(objp) >= 4
    ):
        obj_pts = np.array(
            objp,
            dtype=np.float32
        ).reshape(-1, 3)

        img_pts = np.array(
            imgp,
            dtype=np.float32
        ).reshape(-1, 2)

        # 평면 보드이므로 XY 좌표만으로 Homography 검사
        obj_pts_2d = obj_pts[:, :2]

        H, _ = cv2.findHomography(
            obj_pts_2d,
            img_pts
        )

        if H is not None and H.shape == (3, 3):
            valid_obj_points.append(obj_pts)
            valid_img_points.append(img_pts)

        else:
            print(
                f"[WARN] {i}번째 캡처는 "
                "Homography 검증 실패로 제외"
            )

    else:
        print(
            f"[WARN] {i}번째 캡처는 "
            "코너 수 부족으로 제외"
        )

# ============================================================
# 8. 유효 데이터 확인
# ============================================================
if len(valid_obj_points) < 5:
    print(
        f"\n[ERROR] 유효한 캘리브레이션 데이터가 "
        f"5장 미만입니다. 현재 {len(valid_obj_points)}장"
    )
    sys.exit(1)

print(
    f"\n[INFO] 총 {len(all_charuco_corners)}장 중 "
    f"유효한 {len(valid_obj_points)}장으로 "
    "캘리브레이션을 진행합니다."
)

# ============================================================
# 9. 카메라 캘리브레이션
# ============================================================
retval, camera_matrix, dist_coeffs, rvecs, tvecs = (
    cv2.calibrateCamera(
        valid_obj_points,
        valid_img_points,
        image_size,
        None,
        None
    )
)

print(
    f"\n[DONE] 재투영 오차(RMS Error): "
    f"{retval:.4f} px"
)

# ============================================================
# 10. YAML 결과 저장
# ============================================================
save_path = os.path.join(
    target_dir,
    "calibration_result.yaml"
)

cv_file = cv2.FileStorage(
    save_path,
    cv2.FILE_STORAGE_WRITE
)

cv_file.write(
    "camera_matrix",
    camera_matrix
)

cv_file.write(
    "dist_coeffs",
    dist_coeffs
)

cv_file.write(
    "image_width",
    image_size[0]
)

cv_file.write(
    "image_height",
    image_size[1]
)

cv_file.write(
    "rms_error",
    float(retval)
)

cv_file.release()

# ============================================================
# 11. 최종 결과 출력
# ============================================================
print("\n============================================================")
print("[완료] 카메라 캘리브레이션 성공")
print("============================================================")
print(f"[YAML]    {save_path}")
print(f"[IMAGES]  {target_dir}/cap_*.jpg")
print(f"[PREVIEW] {preview_path}")

print("\n=== Camera Matrix ===")
print(camera_matrix)

print("\n=== Distortion Coefficients ===")
print(dist_coeffs)

print("\n============================================================")
