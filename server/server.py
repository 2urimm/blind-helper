"""
Blind Helper - 경로 B 서버 (YOLO + Metric Depth, 박스별 절대거리 + 폰 회신)

YOLO로 객체 탐지 → 각 박스 중심의 metric depth 값(미터)을 읽어
"라벨 + 거리(m)"를 폰으로 회신. 폰이 프리뷰 위에 오버레이.

Metric Depth 모델: 절대 거리(미터) 추정.
  실내: depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf
  실외: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf

종료: Ctrl+C
"""

import asyncio
import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

# 설정
MODEL_PATH = "yolo11n.pt"
CONF_THRES = 0.25
IMG_SIZE = 480
SHOW_WINDOW = True
TORCH_THREADS = 4
USE_OPENVINO = False
USE_GPU = True                # GPU 컴퓨터: True

# Depth (metric, 미터 단위)
USE_DEPTH = True
# 실내/실외 전환: 아래 둘 중 하나 주석 해제 (각각 측정)
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"   # 실내
# DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"  # 실외
DEPTH_EVERY_N = 1            # GPU면 1 (매 프레임)

if USE_GPU:
    USE_OPENVINO = False

torch.set_num_threads(TORCH_THREADS)
cv2.setNumThreads(TORCH_THREADS)

app = FastAPI()

print("YOLO 모델 로딩 중...")
model = None
using_openvino = False

if USE_OPENVINO:
    import os
    ov_dir = MODEL_PATH.replace(".pt", "_openvino_model")
    try:
        if not os.path.isdir(ov_dir):
            print("OpenVINO 변환 중...")
            YOLO(MODEL_PATH).export(format="openvino", imgsz=IMG_SIZE)
        model = YOLO(ov_dir, task="detect")
        using_openvino = True
    except Exception as e:
        print(f"OpenVINO 실패 ({e}) -> CPU 폴백")

if model is None:
    model = YOLO(MODEL_PATH)
    model.fuse()

_predict_kwargs = dict(imgsz=IMG_SIZE, conf=CONF_THRES, verbose=False)
if USE_GPU:
    _predict_kwargs["device"] = 0

depth_pipe = None
if USE_DEPTH:
    print(f"Metric Depth 로딩 중... ({DEPTH_MODEL})")
    try:
        from transformers import pipeline as hf_pipeline
        depth_device = 0 if USE_GPU else -1
        depth_pipe = hf_pipeline(task="depth-estimation", model=DEPTH_MODEL, device=depth_device)
        print("Depth 모델 로드 완료.")
    except Exception as e:
        print(f"Depth 로드 실패 ({e}) -> depth 없이 진행")
        depth_pipe = None

print(f"모델 로드 완료 (gpu={USE_GPU}, depth={depth_pipe is not None}). 폰 연결 대기 중...")

_lock = threading.Lock()
_latest = {"data": None, "recv_ms": 0.0}
_stop = threading.Event()
_result_lock = threading.Lock()
_pending_result = {"json": None}
latest_view = {"annotated": None}
proc_times = deque(maxlen=30)


def run_metric_depth(frame_bgr):
    """BGR → metric depth(미터) 맵(numpy, HxW), 추론시간ms, 컬러맵"""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t0 = time.time()
    out = depth_pipe(pil)
    depth_ms = (time.time() - t0) * 1000
    # metric 모델: predicted_depth가 미터 단위. out["depth"]는 시각화용 PIL.
    # predicted_depth 텐서를 미터맵으로 사용.
    depth_m = out["predicted_depth"].squeeze().cpu().numpy()  # HxW, 미터
    # 원본 프레임 크기로 리사이즈 (박스 좌표와 맞추기 위해)
    depth_m = cv2.resize(depth_m, (frame_bgr.shape[1], frame_bgr.shape[0]))
    # 시각화
    dnorm = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(dnorm, cv2.COLORMAP_INFERNO)
    return depth_ms, depth_m, depth_color


def inference_worker():
    frame_idx = 0
    last_depth_ms = 0.0
    last_depth_m = None
    while not _stop.is_set():
        with _lock:
            data = _latest["data"]
            recv_ms = _latest["recv_ms"]
            _latest["data"] = None
        if data is None:
            time.sleep(0.002)
            continue

        server_t0 = time.time()

        d0 = time.time()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        decode_ms = (time.time() - d0) * 1000
        if frame is None:
            continue
        H, W = frame.shape[:2]

        # YOLO
        i0 = time.time()
        results = model.predict(frame, **_predict_kwargs)
        yolo_ms = (time.time() - i0) * 1000
        r = results[0]
        annotated = r.plot()

        # Metric Depth
        depth_color = None
        frame_idx += 1
        if depth_pipe is not None and frame_idx % DEPTH_EVERY_N == 0:
            try:
                last_depth_ms, last_depth_m, depth_color = run_metric_depth(frame)
            except Exception as e:
                print(f"\ndepth 에러: {e}")

        # 각 박스 중심의 거리(미터) 계산 → 폰으로 보낼 detection 목록
        dets = []
        for b in r.boxes:
            cls = int(b.cls[0]); conf = float(b.conf[0])
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            cx = int((x1 + x2) / 2); cy = int((y1 + y2) / 2)
            dist_m = None
            if last_depth_m is not None and 0 <= cy < H and 0 <= cx < W:
                # 박스 중심 주변 작은 패치의 중앙값(노이즈 완화)
                y0 = max(0, cy - 3); yb = min(H, cy + 4)
                x0 = max(0, cx - 3); xb = min(W, cx + 4)
                patch = last_depth_m[y0:yb, x0:xb]
                if patch.size > 0:
                    dist_m = float(np.median(patch))
            dets.append({
                "label": model.names[cls],
                "conf": round(conf, 2),
                # 좌표는 원본 프레임(WxH) 기준 → 폰이 화면 크기로 스케일
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "dist": round(dist_m, 2) if dist_m is not None else None,
            })
            # 뷰어에 거리 표시
            if dist_m is not None:
                cv2.putText(annotated, f"{dist_m:.1f}m", (int(x1), int(y2) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        server_ms = (time.time() - server_t0) * 1000

        result_json = json.dumps({
            "server_ms": round(server_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "depth_ms": round(last_depth_ms, 1),
            "frame_w": W, "frame_h": H,
            "dets": dets,
        })
        with _result_lock:
            _pending_result["json"] = result_json

        proc_times.append(time.time() * 1000)
        fps = 0.0
        if len(proc_times) >= 2:
            span = (proc_times[-1] - proc_times[0]) / 1000.0
            if span > 0:
                fps = (len(proc_times) - 1) / span

        lag_ms = time.time() * 1000 - recv_ms
        print(f"fps={fps:4.1f} server={server_ms:6.1f}ms (yolo={yolo_ms:5.1f} depth={last_depth_ms:6.1f}) "
              f"lag={lag_ms:6.1f}ms objs={len(dets)}", end="\r")

        if depth_color is not None:
            combined = np.hstack([annotated, depth_color])
        else:
            combined = annotated
        latest_view["annotated"] = combined


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    print("\n폰 연결됨.")
    try:
        while True:
            data = await ws.receive_bytes()
            with _lock:
                _latest["data"] = data
                _latest["recv_ms"] = time.time() * 1000
            with _result_lock:
                rj = _pending_result["json"]
                _pending_result["json"] = None
            if rj is not None:
                await ws.send_text(rj)
    except WebSocketDisconnect:
        print("\n폰 연결 끊김.")
    except Exception as e:
        print(f"\n에러: {e}")


async def viewer():
    if not SHOW_WINDOW:
        return
    while True:
        img = latest_view["annotated"]
        if img is not None:
            cv2.imshow("Blind Helper - YOLO + Metric Depth", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
        await asyncio.sleep(0.01)


@app.on_event("startup")
async def on_startup():
    threading.Thread(target=inference_worker, daemon=True).start()
    asyncio.create_task(viewer())


@app.on_event("shutdown")
async def on_shutdown():
    _stop.set()