import cv2
import numpy as np
import os
import sys
import time
from statistics import mean

from tflite_runtime.interpreter import Interpreter, load_delegate


import math

MODEL_PATH = "/root/coral_project/best_full_integer_quant_edgetpu.tflite"



CAMERA_DEVICE = "/dev/video0"

CAPTURE_WIDTH = 1280
CAPTURE_HEIGHT = 720
CAPTURE_FPS = 30

USE_MJPG = True

CAPTURE_BUFFER_SIZE = 1

PRINT_EVERY_N_FRAMES = 10

SAVE_PREVIEW = False
PREVIEW_PATH = "/root/coral_project/pnp_live_result.jpg"
SAVE_EVERY_N_FRAMES = 30

HARD_EXIT_ON_CLOSE = True

CLASS_NAMES = [
    "building",  # class 0
    "tank",      # class 1
]

TARGET_CLASS_ID = 1        # 최종 표적 번호 입력

CONF_THRESHOLD = 0.25

IOU_THRESHOLD = 0.45

TARGET_WIDTH_MM = 3000.0

TARGET_HEIGHT_MM = 2500.0

CALIB_WIDTH = 1280.0

CALIB_HEIGHT = 720.0

ORIGINAL_CAMERA_MATRIX = np.array([
    [742.51159615338906, 0.0, 633.04462231163859],
    [0.0,748.93836669159089, 348.57380282650468],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

DIST_COEFFS = np.array([
    [ -0.38764635459923219, 0.14223568774240347,
       0.00074706479027792532, -0.0014494246580076171,
       -0.023106162043305165 ]
], dtype=np.float64)

OBJ_POINTS = np.array([
    [-TARGET_WIDTH_MM / 2, -TARGET_HEIGHT_MM / 2, 0],
    [ TARGET_WIDTH_MM / 2, -TARGET_HEIGHT_MM / 2, 0],
    [ TARGET_WIDTH_MM / 2,  TARGET_HEIGHT_MM / 2, 0],
    [-TARGET_WIDTH_MM / 2,  TARGET_HEIGHT_MM / 2, 0]
], dtype=np.float32)

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

        if w >= (frame_width * 0.4) or h >= (frame_height * 0.4):
            return False

        ratio = w / h

        if 0.8 <= ratio <= 2.7:
            return True

        return False

    def process(
        self,
        detected_box,
        frame_width,
        frame_height,
        detected_cls=None,
        detected_conf=None
    ):
        if detected_box is not None:
            if not self._is_valid_shape(
                detected_box,
                frame_width,
                frame_height
            ):
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

def order_corners(pts):

    pts = np.array(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).flatten()
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return [tl, tr, br, bl]

def normalize_and_filter_line(line, is_horizontal):
    if line is None:
        return None

    vx = float(line[0][0])
    vy = float(line[1][0])
    x0 = float(line[2][0])
    y0 = float(line[3][0])

    if is_horizontal:
        if abs(vy) > abs(vx) * 0.364:
            return None

        if vx < 0:
            vx, vy = -vx, -vy

    else:
        if abs(vx) > abs(vy) * 0.364:
            return None

        if vy < 0:
            vx, vy = -vx, -vy

    mag = math.hypot(vx, vy)

    if mag < 1e-12:
        return None

    return (
        vx / mag,
        vy / mag,
        x0,
        y0
    )

def get_intersection(l1, l2):
    if l1 is None or l2 is None:
        return None

    vx1, vy1, x1, y1 = l1
    vx2, vy2, x2, y2 = l2

    A1 = -vy1
    B1 = vx1
    C1 = vy1 * x1 - vx1 * y1

    A2 = -vy2
    B2 = vx2
    C2 = vy2 * x2 - vx2 * y2

    det = A1 * B2 - A2 * B1

    if abs(det) < 1e-6:
        return None

    return np.array([
        (B1 * C2 - B2 * C1) / det,
        (C1 * A2 - C2 * A1) / det
    ], dtype=np.float64)

def reconstruct_rectangle_top_priority(
    t_line,
    b_line,
    l_line,
    r_line,
    bbox,
    f_w,
    f_h
):


    x1, y1, x2, y2 = bbox

    w_yolo = x2 - x1
    h_yolo = y2 - y1

    cx_yolo = (x1 + x2) / 2.0
    cy_yolo = (y1 + y2) / 2.0

    T_raw = normalize_and_filter_line(
        t_line,
        True
    )

    B_raw = normalize_and_filter_line(
        b_line,
        True
    )

    L_raw = normalize_and_filter_line(
        l_line,
        False
    )

    R_raw = normalize_and_filter_line(
        r_line,
        False
    )

    Vh = None
    Vv = None

    if T_raw:
        Vh = (T_raw[0], T_raw[1])

    elif B_raw:
        Vh = (B_raw[0], B_raw[1])

    elif L_raw:
        Vv = (L_raw[0], L_raw[1])
        Vh = (Vv[1], -Vv[0])

    elif R_raw:
        Vv = (R_raw[0], R_raw[1])
        Vh = (Vv[1], -Vv[0])

    if Vh is None:
        return None

    if Vv is None:
        Vv = (-Vh[1], Vh[0])

    def make_line(pt, vec):
        return (
            vec[0],
            vec[1],
            pt[0],
            pt[1]
        )

    T = (
        T_raw
        if T_raw
        else make_line(
            (cx_yolo, y1),
            Vh
        )
    )

    B = (
        B_raw
        if B_raw
        else make_line(
            (cx_yolo, y2),
            Vh
        )
    )

    L = (
        L_raw
        if L_raw
        else make_line(
            (x1, cy_yolo),
            Vv
        )
    )

    R = (
        R_raw
        if R_raw
        else make_line(
            (x2, cy_yolo),
            Vv
        )
    )

    T = (
        Vh[0],
        Vh[1],
        T[2],
        T[3]
    )

    B = (
        Vh[0],
        Vh[1],
        B[2],
        B[3]
    )

    L = (
        Vv[0],
        Vv[1],
        L[2],
        L[3]
    )

    R = (
        Vv[0],
        Vv[1],
        R[2],
        R[3]
    )

    if T_raw or B_raw is None:
        anchor_type = "TOP"

        tl = get_intersection(T, L)
        tr = get_intersection(T, R)

        if tl is None or tr is None:
            return None

        W = np.linalg.norm(tr - tl)

        if W <= 1e-6:
            W = float(w_yolo)

        H = W * (5.0 / 6.0)

        bl = tl + np.array(Vv) * H
        br = tr + np.array(Vv) * H

        anchor_pt = (tl + tr) / 2.0

    elif B_raw:
        anchor_type = "BOTTOM"

        bl = get_intersection(B, L)
        br = get_intersection(B, R)

        if bl is None or br is None:
            return None

        W = np.linalg.norm(br - bl)

        if W <= 1e-6:
            W = float(w_yolo)

        H = W * (5.0 / 6.0)

        tl = bl - np.array(Vv) * H
        tr = br - np.array(Vv) * H

        anchor_pt = (bl + br) / 2.0

    else:
        return None

    min_W = w_yolo * 1.05
    min_H = h_yolo * 1.05

    scale = max(
        1.0,
        min_W / W if W > 0 else 1.0,
        min_H / H if H > 0 else 1.0
    )

    if scale > 1.0:
        W *= scale
        H *= scale

        Vh_arr = np.array(Vh)
        Vv_arr = np.array(Vv)

        if anchor_type == "TOP":
            tl = anchor_pt - Vh_arr * (W / 2.0)
            tr = anchor_pt + Vh_arr * (W / 2.0)

            bl = tl + Vv_arr * H
            br = tr + Vv_arr * H

        elif anchor_type == "BOTTOM":
            bl = anchor_pt - Vh_arr * (W / 2.0)
            br = anchor_pt + Vh_arr * (W / 2.0)

            tl = bl - Vv_arr * H
            tr = br - Vv_arr * H

    return (
        tl,
        tr,
        br,
        bl
    )

def letterbox(image, new_shape, color=(114, 114, 114)):

    h, w = image.shape[:2]

    new_w = int(new_shape[0])
    new_h = int(new_shape[1])

    scale = min(
        new_w / w,
        new_h / h
    )

    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_LINEAR
    )

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    left = pad_w // 2
    right = pad_w - left

    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top

def quantize_input(rgb_image, input_detail):
    dtype = input_detail["dtype"]
    scale, zero_point = input_detail["quantization"]

    if dtype == np.int8:
        if scale == 0:
            raise RuntimeError("input quantization scale이 0입니다.")

        real = rgb_image.astype(np.float32) / 255.0

        quantized = np.round(
            real / scale + zero_point
        )

        quantized = np.clip(
            quantized,
            -128,
            127
        ).astype(np.int8)

        return np.expand_dims(
            quantized,
            axis=0
        )

    if dtype == np.uint8:
        return np.expand_dims(
            rgb_image.astype(np.uint8),
            axis=0
        )

    if dtype == np.float32:
        return np.expand_dims(
            rgb_image.astype(np.float32) / 255.0,
            axis=0
        )

    raise RuntimeError(
        f"지원하지 않는 input dtype: {dtype}"
    )

def dequantize_output(output, output_detail):
    dtype = output_detail["dtype"]
    scale, zero_point = output_detail["quantization"]

    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            return output.astype(np.float32)

        return (
            output.astype(np.float32) - zero_point
        ) * scale

    return output.astype(np.float32)

def detect_best_target(
    output_float,
    original_shape,
    model_width,
    model_height,
    resize_scale,
    pad_x,
    pad_y,
    target_class_id,
    conf_threshold,
    iou_threshold
):


    pred = np.squeeze(
        output_float,
        axis=0
    )

    # [C, N] -> [N, C]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    if pred.ndim != 2 or pred.shape[1] < 5:
        raise RuntimeError(
            f"예상하지 못한 output shape: {pred.shape}"
        )

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    confidences = class_scores[
        np.arange(
            class_scores.shape[0]
        ),
        class_ids
    ]

    # "모든 클래스 중 1위"가 아니라
    # tank(class 1)만 후보로 남긴다.
    mask = (
        (class_ids == target_class_id)
        & (confidences >= conf_threshold)
    )

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    candidate_count = len(boxes_xywh)

    if candidate_count == 0:
        return None, 0

    # 일부 export가 normalized xywh를 내보내는 경우 대비
    if float(np.max(np.abs(boxes_xywh))) <= 2.0:
        boxes_xywh = boxes_xywh.copy()

        boxes_xywh[:, 0] *= model_width
        boxes_xywh[:, 1] *= model_height
        boxes_xywh[:, 2] *= model_width
        boxes_xywh[:, 3] *= model_height

    original_h, original_w = original_shape[:2]

    mapped_boxes = []
    nms_boxes = []

    for box in boxes_xywh:
        cx, cy, bw, bh = box

        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0

        x2 = cx + bw / 2.0
        y2 = cy + bh / 2.0

        # letterbox 좌표 -> 원본 이미지 좌표
        x1 = (x1 - pad_x) / resize_scale
        y1 = (y1 - pad_y) / resize_scale

        x2 = (x2 - pad_x) / resize_scale
        y2 = (y2 - pad_y) / resize_scale

        x1 = max(
            0.0,
            min(
                original_w - 1.0,
                x1
            )
        )

        y1 = max(
            0.0,
            min(
                original_h - 1.0,
                y1
            )
        )

        x2 = max(
            0.0,
            min(
                original_w - 1.0,
                x2
            )
        )

        y2 = max(
            0.0,
            min(
                original_h - 1.0,
                y2
            )
        )

        mapped_boxes.append(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

        nms_boxes.append([
            int(round(x1)),
            int(round(y1)),
            max(
                1,
                int(round(x2 - x1))
            ),
            max(
                1,
                int(round(y2 - y1))
            )
        ])

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        confidences.tolist(),
        conf_threshold,
        iou_threshold
    )

    if len(indices) == 0:
        return None, candidate_count

    indices = np.array(
        indices
    ).reshape(-1)

    best = None

    for i in indices:
        x1, y1, x2, y2 = mapped_boxes[i]

        det = {
            "class_id": int(class_ids[i]),
            "confidence": float(confidences[i]),
            "bbox": np.array(
                [x1, y1, x2, y2],
                dtype=np.float32
            )
        }

        if (
            best is None
            or det["confidence"] > best["confidence"]
        ):
            best = det

    return best, candidate_count

# ============================================================
# K=3 고정 벤치마크 설정
# ============================================================

TOP_K = 3
WARMUP_RUNS = 10
BENCHMARK_RUNS = 100
EDGE_THRESHOLD = 50

RESULT_IMAGE = "/root/coral_project/pnp_k3_benchmark_result.jpg"

CLAHE = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)


def ms(sec):
    return sec * 1000.0


def percentile(values, p):
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def fast_quantize_input(rgb_image, input_detail):
    dtype = input_detail["dtype"]
    scale, zero_point = input_detail["quantization"]

    if (
        dtype == np.int8
        and zero_point == -128
        and abs(float(scale) - (1.0 / 255.0)) < 1e-6
    ):
        q = (rgb_image.astype(np.int16) - 128).astype(np.int8)
        return np.expand_dims(q, axis=0)

    return quantize_input(rgb_image, input_detail)


def prepare_edge_maps(gray_img, ex1, ey1, ex2, ey2):
    h, w = gray_img.shape[:2]

    ex1 = max(0, min(w, int(ex1)))
    ex2 = max(0, min(w, int(ex2)))
    ey1 = max(0, min(h, int(ey1)))
    ey2 = max(0, min(h, int(ey2)))

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    roi = gray_img[ey1:ey2, ex1:ex2]

    if roi.size == 0:
        return None

    enhanced = CLAHE.apply(roi)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    sobel_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1, ksize=3)
    sobel_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)

    edge_h = cv2.convertScaleAbs(sobel_y)
    edge_v = cv2.convertScaleAbs(sobel_x)

    return edge_h, edge_v, ex1, ey1


def find_topk_line(
    edge_map,
    origin_x,
    origin_y,
    rx1,
    ry1,
    rx2,
    ry2,
    direction,
    top_k=3
):
    map_h, map_w = edge_map.shape[:2]

    lx1 = int(rx1) - origin_x
    ly1 = int(ry1) - origin_y
    lx2 = int(rx2) - origin_x
    ly2 = int(ry2) - origin_y

    lx1 = max(0, min(map_w, lx1))
    lx2 = max(0, min(map_w, lx2))
    ly1 = max(0, min(map_h, ly1))
    ly2 = max(0, min(map_h, ly2))

    if lx2 <= lx1 or ly2 <= ly1:
        return None, 0

    crop = edge_map[ly1:ly2, lx1:lx2]

    if crop.size == 0:
        return None, 0

    normalized = cv2.normalize(
        crop,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    pts_list = []

    if direction == "H":
        height, width = normalized.shape
        k = min(top_k, height)

        for x in range(width):
            col = normalized[:, x]

            idxs = np.argpartition(
                col,
                -k
            )[-k:]

            for y in idxs:
                if int(col[y]) >= EDGE_THRESHOLD:
                    pts_list.append([
                        x + origin_x + lx1,
                        int(y) + origin_y + ly1
                    ])

    else:
        height, width = normalized.shape
        k = min(top_k, width)

        for y in range(height):
            row = normalized[y, :]

            idxs = np.argpartition(
                row,
                -k
            )[-k:]

            for x in idxs:
                if int(row[x]) >= EDGE_THRESHOLD:
                    pts_list.append([
                        int(x) + origin_x + lx1,
                        y + origin_y + ly1
                    ])

    if len(pts_list) < 15:
        return None, len(pts_list)

    pts = np.asarray(
        pts_list,
        dtype=np.float32
    )

    line = cv2.fitLine(
        pts,
        cv2.DIST_L1,
        0,
        0.01,
        0.01
    )

    return line, len(pts)

def main():
    """
    실시간 단일 카메라 좌표 출력 전용 버전.

    정상 동작 중 stdout 출력 형식:
        X,Y,Z

    단위:
        meter [m]

    예:
        -0.20413,-0.12289,13.71810

    검출/PnP 실패 프레임에서는 아무 것도 출력하지 않는다.
    """

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"모델 파일 없음: {MODEL_PATH}")

    delegate = load_delegate("libedgetpu.so.1")

    interpreter = Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[delegate]
    )

    interpreter.allocate_tensors()

    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    model_h = int(input_detail["shape"][1])
    model_w = int(input_detail["shape"][2])

    cap = cv2.VideoCapture(
        CAMERA_DEVICE,
        cv2.CAP_V4L2
    )

    if USE_MJPG:
        cap.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_BUFFER_SIZE)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"카메라를 열 수 없습니다: {CAMERA_DEVICE}"
        )

    # 시작 직후 카메라 안정화용 프레임
    first_frame = None

    for _ in range(5):
        ok, tmp = cap.read()

        if ok and tmp is not None:
            first_frame = tmp

    if first_frame is None:
        cap.release()
        raise RuntimeError("카메라에서 첫 프레임을 읽지 못했습니다.")

    frame_h, frame_w = first_frame.shape[:2]


    sx = frame_w / CALIB_WIDTH
    sy = frame_h / CALIB_HEIGHT

    CAMERA_MATRIX = np.array([
        [
            ORIGINAL_CAMERA_MATRIX[0, 0] * sx,
            0.0,
            ORIGINAL_CAMERA_MATRIX[0, 2] * sx
        ],
        [
            0.0,
            ORIGINAL_CAMERA_MATRIX[1, 1] * sy,
            ORIGINAL_CAMERA_MATRIX[1, 2] * sy
        ],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    tracker_filter = ShapeAndSizeFilter(
        max_lost_frames=5
    )

    def process_frame(original_frame):
        frame_h, frame_w = original_frame.shape[:2]

        gray_frame = cv2.cvtColor(
            original_frame,
            cv2.COLOR_BGR2GRAY
        )

        input_bgr, resize_scale, pad_x, pad_y = letterbox(
            original_frame,
            (model_w, model_h)
        )

        input_rgb = cv2.cvtColor(
            input_bgr,
            cv2.COLOR_BGR2RGB
        )

        input_tensor = fast_quantize_input(
            input_rgb,
            input_detail
        )

        interpreter.set_tensor(
            input_detail["index"],
            input_tensor
        )

        interpreter.invoke()

        output_raw = interpreter.get_tensor(
            output_detail["index"]
        )

        output_float = dequantize_output(
            output_raw,
            output_detail
        )

        detection, candidate_count = detect_best_target(
            output_float,
            original_frame.shape,
            model_w,
            model_h,
            resize_scale,
            pad_x,
            pad_y,
            TARGET_CLASS_ID,
            CONF_THRESHOLD,
            IOU_THRESHOLD
        )

        if detection is None:
            filtered_box, filtered_cls, filtered_conf = tracker_filter.process(
                None,
                frame_w,
                frame_h,
                None,
                None
            )
        else:
            filtered_box, filtered_cls, filtered_conf = tracker_filter.process(
                detection["bbox"],
                frame_w,
                frame_h,
                detection["class_id"],
                detection["confidence"]
            )

        if filtered_box is None:
            return None

        x1, y1, x2, y2 = map(int, filtered_box)

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        bw = x2 - x1
        bh = y2 - y1

        if bw <= 0 or bh <= 0:
            return None

        ex1 = max(0, cx - bw)
        ex2 = min(frame_w, cx + bw)
        ey1 = max(0, int(cy - 1.5 * bh))
        ey2 = min(frame_h, int(cy + 1.5 * bh))

        in_w = int(bw * 0.25)
        in_h = int(bh * 0.25)

        edge_pack = prepare_edge_maps(
            gray_frame,
            ex1,
            ey1,
            ex2,
            ey2
        )

        if edge_pack is None:
            return None

        edge_h, edge_v, origin_x, origin_y = edge_pack

        top_line, _ = find_topk_line(
            edge_h,
            origin_x,
            origin_y,
            ex1,
            ey1,
            ex2,
            y1 + in_h,
            "H",
            TOP_K
        )

        bottom_line, _ = find_topk_line(
            edge_h,
            origin_x,
            origin_y,
            ex1,
            y2 - in_h,
            ex2,
            ey2,
            "H",
            TOP_K
        )

        left_line, _ = find_topk_line(
            edge_v,
            origin_x,
            origin_y,
            ex1,
            ey1,
            x1 + in_w,
            ey2,
            "V",
            TOP_K
        )

        right_line, _ = find_topk_line(
            edge_v,
            origin_x,
            origin_y,
            x2 - in_w,
            ey1,
            ex2,
            ey2,
            "V",
            TOP_K
        )

        raw_corners = reconstruct_rectangle_top_priority(
            top_line,
            bottom_line,
            left_line,
            right_line,
            (x1, y1, x2, y2),
            frame_w,
            frame_h
        )

        if raw_corners is None:
            return None

        img_points = np.array(
            order_corners(raw_corners),
            dtype=np.float32
        )

        ok, rvec, tvec = cv2.solvePnP(
            OBJ_POINTS,
            img_points,
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not ok:
            return None

        # 카카메라 기존 좌표계 유지:
        # X = right(+)
        # Y = up(+) 이거 아님!!!! 아래가 +
        # Z = forward(+)
        return (
            float(tvec[0][0]) / 1000.0,
            float(tvec[1][0]) / 1000.0,
            float(tvec[2][0]) / 1000.0
        )

    # --------------------------------------------------------
    # Live loop
    # --------------------------------------------------------
    try:
        pending_frame = first_frame

        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                ok, frame = cap.read()

                if not ok or frame is None:
                    time.sleep(0.02)
                    continue

            xyz = process_frame(frame)

            # 정상 좌표가 계산된 경우에만 딱 한 줄 출력
            if xyz is not None:
                x_m, y_m, z_m = xyz

                print(
                    f"{x_m:.5f},{y_m:.5f},{z_m:.5f}",
                    flush=True
                )

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()

        sys.stdout.flush()
        sys.stderr.flush()

        if HARD_EXIT_ON_CLOSE:
            os._exit(0)


if __name__ == "__main__":
    main()
