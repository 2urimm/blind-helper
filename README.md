# Camera Access App

A sample Android application demonstrating integration with Meta Wearables Device Access Toolkit. This app walks the SDK's camera lifecycle as explicit steps — start a session, start the preview, capture or record, stop the preview, end the session — on a single full-bleed camera screen.

이 프로젝트(Blind Helper)는 여기에 **경로 B**를 더했습니다: 폰이 안경 카메라 프레임(JPEG)을 WebSocket으로 노트북 서버에 보내고, 서버가 YOLO 객체 탐지 + Depth Anything V2 깊이 추론을 돌려 결과를 폰으로 돌려줍니다.

```
Meta AI 안경 ──(DAT SDK)──▶ Android 앱 ──(ws://<노트북IP>:8000/stream, JPEG)──▶ server/server.py
                                  ▲                                                      │
                                  └──────────────── 결과 JSON (탐지 수·구간별 시간) ◀──────┘
```

## Features

- Connect to Meta AI glasses
- Explicit camera lifecycle: start/end a device session and start/stop the live preview
- Stream the camera feed from the device
- Capture photos
- Record video, with optional sound-in-video
- Keep recording while the app is backgrounded (foreground service)
- Preview and share captured photos and recorded videos
- Open the firmware update flow when required
- (경로 B) 프레임을 노트북 서버로 전송해 YOLO + 깊이 추론, 노트북 화면에 박스 표시

## Prerequisites

- Android Studio Narwhal (2025.1.1) or newer
- JDK 17 or newer
- Android SDK 36 or newer
- Meta Wearables Device Access Toolkit (included as a dependency)
- A Meta AI glasses device for testing (optional for development)
- (경로 B) Anaconda/Miniconda가 설치된 노트북, 폰과 **같은 Wi-Fi**

## Building the app

### Using Android Studio

1. Clone this repository
1. Open the project in Android Studio
1. Add your personal access token (classic) to the `local.properties` file (see [SDK for Android setup](https://wearables.developer.meta.com/docs/develop/dat/build-integration-android#step-2-add-the-sdk-to-gradle))
1. 서버 주소를 노트북 IP로 맞추기: `CameraViewModel.kt`의 `SERVER_URL` (아래 [노트북 IP 주소 확인](#3-노트북-ip-주소-확인) 참고)
1. Click **File** > **Sync Project with Gradle Files**
1. Click **Run** > **Run...** > **app**

## Running the app

1. Turn 'Developer Mode' on in the Meta AI app.
1. (경로 B) 먼저 노트북에서 서버를 실행합니다 ([아래](#server-경로-b-노트북) 참고).
1. Launch the app.
1. Press the "Connect" button to complete app registration.
1. Tap "Start Session" to connect to your glasses, then "Preview" to begin the live camera feed.
1. Use the on-screen controls to:
   - Capture photos
   - Record video, toggling the microphone for sound-in-video
   - Preview and share captured photos and recorded videos
   - Stop the preview, end the session, or disconnect from the device
1. If a firmware update is required, tap "Update firmware".

## Server (경로 B, 노트북)

폰에서 JPEG 프레임을 받아 YOLO로 탐지(+ 깊이 추론)하고, 노트북 화면에 박스를 그려 보여줍니다. 코드는 `server/` 폴더에 있습니다.

### 1. conda 환경 만들기

Anaconda Prompt에서 `server/` 폴더로 이동한 뒤:

```
conda env create -f environment.yml
conda activate blind-helper
```

> ultralytics 설치 중 torch(CPU 버전)가 자동으로 깔립니다. 몇 분 걸릴 수 있어요.

### 2. 서버 실행

`server/` 폴더에서:

```
uvicorn server:app --host 0.0.0.0 --port 8000
```

- `--host 0.0.0.0` : 같은 Wi-Fi의 폰이 접속할 수 있게 모든 IP에서 수신
- 처음 실행 시 `yolo11n.pt` 모델이 자동 다운로드되고, OpenVINO 모델로 변환됩니다.
- 깊이 모델(`depth-anything/Depth-Anything-V2-Small-hf`)도 처음 실행 시 받아옵니다.
- "모델 로드 완료 ... 폰 연결 대기 중..." 이 뜨면 준비 완료.

주요 설정은 `server.py` 상단에 있습니다:

| 설정 | 기본값 | 설명 |
|---|---|---|
| `USE_OPENVINO` | `True` | CPU 가속. GPU면 `False` |
| `USE_GPU` | `False` | GPU 컴퓨터면 `True` |
| `USE_DEPTH` | `True` | 깊이 추론 켜기/끄기 |
| `DEPTH_EVERY_N` | `5` | N프레임마다 한 번만 깊이 추론 (GPU면 1) |
| `CONF_THRES` / `IMG_SIZE` | `0.4` / `480` | YOLO 신뢰도 임계값 / 입력 크기 |

### 3. 노트북 IP 주소 확인

Anaconda Prompt 또는 CMD에서:

```
ipconfig
```

- "무선 LAN 어댑터 Wi-Fi" 항목의 **IPv4 주소** (예: `192.168.0.15`)를 확인.
- 폰 앱이 접속할 주소: `ws://192.168.0.15:8000/stream` (실제 IP로 바꿔서)
- 이 주소는 앱의 `CameraViewModel.kt` → `SERVER_URL`에 넣고 다시 빌드합니다.

### 4. 방화벽

처음 uvicorn 실행 시 Windows 방화벽 팝업이 뜨면 **"액세스 허용"** (특히 개인 네트워크).
안 뜨는데 폰이 연결 안 되면, Windows Defender 방화벽에서 8000 포트 인바운드 허용 필요.

### 서버 응답 (폰으로 돌아오는 JSON)

```json
{ "server_ms": 0.0, "decode_ms": 0.0, "yolo_ms": 0.0, "depth_ms": 0.0, "n": 0 }
```

- 구간별 처리 시간(ms)과 탐지된 객체 수(`n`).
- 폰 오버레이를 만들 때 이 JSON을 그대로 쓰면 됩니다.

### 종료

콘솔에서 `Ctrl+C`.

## Troubleshooting

For issues related to the Meta Wearables Device Access Toolkit, please refer to the [developer documentation](https://wearables.developer.meta.com/docs/develop/dat/) or visit our [discussions forum](https://github.com/facebook/meta-wearables-dat-android/discussions)

- 폰이 서버에 연결이 안 되면: 같은 Wi-Fi인지, `SERVER_URL`의 IP가 현재 노트북 IP와 같은지, 방화벽 8000 포트가 열려 있는지 확인.

## License

This source code is licensed under the license found in the LICENSE file in the root directory of this source tree.
