import os
import sys
import time
import threading
from collections import deque
import cv2
import numpy as np
import MNN

# ---------------------------------------------------------------------------
# 全局常量配置
# ---------------------------------------------------------------------------
NUM_CLASSES = 80
INPUT_SIZE = 320
OBJ_THRESH = 0.50
CONF_THRESH = 0.50
NMS_THRESH = 0.50
NUM_THREADS = 4
SCORE_COMPENSATION = 0.04

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# 异步多线程摄像头读取类
class CameraStream:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.is_picam = False
        self.cap = None
        self.picam2 = None

        try:
            from picamera2 import Picamera2
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.5)
            self.is_picam = True
            print("[INFO] Initialized Picamera2 (BGR888 mode).")
        except Exception as e:
            print(f"[WARN] Picamera2 unavailable ({e}), falling back to cv2.VideoCapture(0)...")
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            if self.is_picam:
                img = self.picam2.capture_array()
                if img is not None and img.size > 0:
                    with self.lock:
                        self.frame = img
            elif self.cap and self.cap.isOpened():
                ret, img = self.cap.read()
                if ret and img is not None:
                    with self.lock:
                        self.frame = img  # cv2.VideoCapture 原生即 BGR
                else:
                    time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        if self.is_picam and self.picam2:
            self.picam2.stop()
        if self.cap:
            self.cap.release()


# 高效预处理 (Letterbox + 预分配内存复用)

def preprocess_fast(bgr_frame, input_size, preproc_buffer):
    h, w, _ = bgr_frame.shape
    max_side = max(h, w)
    ratio = float(input_size) / float(max_side)
    
    fx = int(w * ratio)
    fy = int(h * ratio)
    pad_w = (input_size - fx) // 2
    pad_h = (input_size - fy) // 2

    # 先缩放到目标尺寸 (小图)，再转 RGB 供模型推理
    resized_bgr = cv2.resize(bgr_frame, (fx, fy), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

    # 复用预分配的背景画布 (127/255 ≈ 0.498)
    preproc_buffer.fill(0.498)
    preproc_buffer[pad_h:pad_h + fy, pad_w:pad_w + fx] = resized_rgb.astype(np.float32) * (1.0 / 255.0)

    mat_info = {
        "ratio": ratio,
        "pad_w": pad_w,
        "pad_h": pad_h
    }
    return preproc_buffer, mat_info

# ---------------------------------------------------------------------------
# NMS 算法 (NumPy 向量化加速)
# ---------------------------------------------------------------------------
def nms(boxes, scores, nms_thresh):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr < nms_thresh)[0]
        order = order[inds + 1]

    return keep

# ---------------------------------------------------------------------------
# 绘制边界框
# ---------------------------------------------------------------------------
def draw_boxes(img, boxes, scores, labels, mat_info):
    ratio = mat_info["ratio"]
    pad_w = mat_info["pad_w"]
    pad_h = mat_info["pad_h"]

    for box, score, label in zip(boxes, scores, labels):
        x1 = int((box[0] / ratio) - (pad_w / ratio))
        y1 = int((box[1] / ratio)- (pad_h / ratio))
        objw = int((box[2] / ratio) - (box[0] / ratio))
        objh = int((box[3] / ratio)- (box[1] / ratio))

        cv2.rectangle(img, (x1, y1), (x1 + objw, y1 + objh), (0, 255, 0), 2)
        label_text = f"{CLASS_NAMES[label]} {score * 100:.1f}%"
        pos_y = max(y1 - 5, 12)
        cv2.putText(img, label_text, (x1, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# ---------------------------------------------------------------------------
# 安全高效读取 MNN Tensor 数据
# ---------------------------------------------------------------------------
def read_mnn_tensor(tensor, host_tensor):
    tensor.copyToHostTensor(host_tensor)
    if hasattr(host_tensor, "getNumpyData"):
        return host_tensor.getNumpyData()
    elif hasattr(host_tensor, "read"):
        return host_tensor.read()
    else:
        return np.array(host_tensor.getData()).reshape(host_tensor.getShape())

# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------
def main():
    # 默认优先采用 FP16 模型，推理性能提升显著
    default_model = "v5lite-e-mnnd_fp16.mnn" if os.path.exists("v5lite-e-mnnd_fp16.mnn") else "v5lite-e-mnnd.mnn"
    model_path = sys.argv[1] if len(sys.argv) >= 2 else default_model

    print(f"[INFO] Loading MNN model: {model_path}")
    interpreter = MNN.Interpreter(model_path)

    # 开启低精度模式 (FP16 NEON 加速) 与高功耗大核调度
    session_config = {
        "numThread": NUM_THREADS,
        "type": "MNN_FORWARD_CPU",
        "precision": "low",   # 启用 FP16 加速
        "power": "high"       # 绑定大核/高频调度
    }
    session = interpreter.createSession(session_config)
    input_tensor = interpreter.getSessionInput(session)
    output_tensor = interpreter.getSessionOutput(session)

    # 预分配 Host 端的内存 Tensor 与输入 Buffer (避免循环内反复 malloc)
    input_shape = (1, INPUT_SIZE, INPUT_SIZE, 3)
    preproc_buffer = np.full((INPUT_SIZE, INPUT_SIZE, 3), 0.498, dtype=np.float32)
    
    out_shape = output_tensor.getShape()
    host_output_tensor = MNN.Tensor(
        out_shape,
        output_tensor.getDataType(),
        np.zeros(out_shape, dtype=np.float32),
        output_tensor.getDimensionType()
    )

    # 启动异步摄像头采集
    cam = CameraStream(width=640, height=480)
    print("[INFO] Camera stream started. Press ESC to quit.")

    fps_window = 30
    frame_times = deque(maxlen=fps_window)  # O(1) 自动淘汰旧帧时间
    avg_fps = 0.0
    frame_count = 0

    try:
        while True:
            t_begin = time.perf_counter()

            # 获取最新一帧 (Picamera2 RGB888 实际内存字节序为 BGR)
            raw_bgr = cam.read()
            
            if raw_bgr is None:
                time.sleep(0.005)
                continue
            # 快速预处理 (仅在缩放后的小图上做 BGR→RGB 转换供模型推理)
            preproc_img, mat_info = preprocess_fast(raw_bgr, INPUT_SIZE, preproc_buffer)

            # 构造输入并拷贝到 Session Input
            input_data = np.expand_dims(preproc_img, axis=0)
            tmp_input = MNN.Tensor(input_shape, MNN.Halide_Type_Float, input_data, MNN.Tensor_DimensionType_Tensorflow)
            input_tensor.copyFrom(tmp_input)

            # 执行推理
            interpreter.runSession(session)

            # 高效读取输出 Tensor
            output_data = read_mnn_tensor(output_tensor, host_output_tensor)
            output_data = np.squeeze(output_data, axis=0)

            # 后处理与 NMS
            obj_confs = output_data[:, 4]
            mask = obj_confs >= OBJ_THRESH
            filtered_data = output_data[mask]

            boxes_list, scores_list, labels_list = [], [], []
            if len(filtered_data) > 0:
                cls_probs = filtered_data[:, 5:]
                max_labels = np.argmax(cls_probs, axis=1)
                max_cls_confs = np.max(cls_probs, axis=1)
                confs = filtered_data[:, 4] * max_cls_confs
                conf_mask = confs >= CONF_THRESH

                if np.any(conf_mask):
                    valid_data = filtered_data[conf_mask]
                    valid_confs = np.minimum(confs[conf_mask] + SCORE_COMPENSATION, 1.0)
                    valid_labels = max_labels[conf_mask]

                    cx, cy, w, h = valid_data[:, 0], valid_data[:, 1], valid_data[:, 2], valid_data[:, 3]
                    x1 = np.maximum(0.0, cx - w / 2.0)
                    y1 = np.maximum(0.0, cy - h / 2.0)
                    x2 = np.minimum(cx + w / 2.0, float(INPUT_SIZE - 1))
                    y2 = np.minimum(cy + h / 2.0, float(INPUT_SIZE - 1))

                    boxes = np.stack([x1, y1, x2, y2], axis=1)
                    keep = nms(boxes, valid_confs, NMS_THRESH)

                    boxes_list = boxes[keep]
                    scores_list = valid_confs[keep]
                    labels_list = valid_labels[keep]

            # 渲染与显示 (直接在原帧上绘制，省去 .copy() 的 640x480x3 内存拷贝)
            draw_boxes(raw_bgr, boxes_list, scores_list, labels_list, mat_info)
            cv2.putText(raw_bgr, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.imshow("YOLOv5-Lite MNN", raw_bgr)

            # 实时 FPS 计算
            t_end = time.perf_counter()
            frame_times.append((t_end - t_begin) * 1000.0)
            avg_fps = 1000.0 / (sum(frame_times) / len(frame_times))
            frame_count += 1

            # 按 ESC 退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] Exited gracefully. Total frames processed: {frame_count}")

if __name__ == "__main__":
    main()

