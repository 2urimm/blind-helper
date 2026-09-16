"""
Blind Helper - 경로 B 서버 (구간측정 + 단안 Depth 추론)

YOLO(객체) + Depth Anything V2(깊이) 동시 추론. 구간별 시간 측정.
Depth는 무거우므로 DEPTH_EVERY_N 프레임마다 한 번만 (노트북 CPU 대비).

측정: decode_ms / yolo_ms / depth_ms / server_ms → 응답으로 폰에 전달
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
CONF_THRES = 0.4
IMG_SIZE = 480
SHOW_WINDOW = True
TORCH_THREADS = 4
USE_OPENVINO = True     # GPU면 False
USE_GPU = False         # GPU 컴퓨터면 True

# --- Depth 설정 ---
USE_DEPTH = True                          # depth 추론 켜기
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_EVERY_N = 5                         # N프레임마다 1번만 depth (CPU 부담↓). GPU면 1로.

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

# --- Depth 모델 로드 ---
depth_pipe = None
if USE_DEPTH:
    print("Depth Anything V2 로딩 중... (최초 다운로드 시 오래 걸림)")
    try:
        from transformers import pipeline as hf_pipeline
        depth_device = 0 if USE_GPU else -1   # -1=CPU, 0=첫 GPU
        depth_pipe = hf_pipeline(
            task="depth-estimation",
            model=DEPTH_MODEL,
            device=depth_device,
        )
        print("Depth 모델 로드 완료.")
    except Exception as e:
        print(f"Depth 로드 실패 ({e}) -> depth 없이 진행")
        depth_pipe = None

print(f"모델 로드 완료 (openvino={using_openvino}, gpu={USE_GPU}, depth={depth_pipe is not None}). 폰 연결 대기 중...")

_lock = threading.Lock()
_latest = {"data": None, "recv_ms": 0.0}
_stop = threading.Event()
_result_lock = threading.Lock()
_pending_result = {"json": None}
latest_view = {"annotated": None}
proc_times = deque(maxlen=30)
_frame_idx = 0
_last_depth_ms = 0.0


def run_depth(frame_bgr):
    """BGR numpy → depth 추론 → (추론시간ms, 컬러맵 depth 이미지)"""
    from PIL import Image
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t0 = time.time()
    out = depth_pipe(pil)
    depth_ms = (time.time() - t0) * 1000
    # depth: PIL 이미지(그레이). 컬러맵으로 시각화
    depth_np = np.array(out["depth"])
    depth_norm = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_color = cv2.resize(depth_color, (frame_bgr.shape[1], frame_bgr.shape[0]))
    return depth_ms, depth_color


def inference_worker():
    global _frame_idx, _last_depth_ms
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

        # YOLO
        i0 = time.time()
        results = model.predict(frame, **_predict_kwargs)
        yolo_ms = (time.time() - i0) * 1000
        r = results[0]
        annotated = r.plot()

        # Depth (N프레임마다)
        depth_ms = 0.0
        depth_color = None
        _frame_idx += 1
        if depth_pipe is not None and _frame_idx % DEPTH_EVERY_N == 0:
            try:
                depth_ms, depth_color = run_depth(frame)
                _last_depth_ms = depth_ms
            except Exception as e:
                print(f"\ndepth 에러: {e}")

        server_ms = (time.time() - server_t0) * 1000

        dets = [{"label": model.names[int(b.cls[0])], "conf": round(float(b.conf[0]), 2)}
                for b in r.boxes]

        result_json = json.dumps({
            "server_ms": round(server_ms, 1),
            "decode_ms": round(decode_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "depth_ms": round(_last_depth_ms, 1),
            "n": len(dets),
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
        info = (f"fps={fps:4.1f} server={server_ms:6.1f}ms "
                f"(dec={decode_ms:4.1f} yolo={yolo_ms:5.1f} depth={_last_depth_ms:6.1f}) "
                f"lag={lag_ms:6.1f}ms objs={len(dets)}")
        print(info, end="\r")

        # 뷰어: YOLO 박스 + depth를 나란히
        if depth_color is not None:
            combined = np.hstack([annotated, depth_color])
        else:
            combined = annotated
        cv2.putText(combined, f"server={server_ms:.0f}ms yolo={yolo_ms:.0f} depth={_last_depth_ms:.0f} fps={fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
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
            cv2.imshow("Blind Helper - YOLO + Depth", img)
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
