# run_3cls_audio

**Jetson Nano + TensorRT** 환경에서 **표정(negative / neutral / positive)** 과 **자세(좋음 / 보통 / 나쁨)·다리떨기**를 실시간 추정하고, **주석 영상·원본 영상·오디오·XML 이벤트**를 저장하는 단일 실행 스크립트입니다.  
본 스크립트는 **GStreamer/RTSP 미사용**, OpenCV `VideoCapture`로 카메라를 직접 엽니다. `.sh` 없이 **`python3` 한 줄**로 실행합니다.

---

## ✨ 주요 기능

- TensorRT 엔진(얼굴 3클래스, MoveNet 포즈) 기반 **실시간 추론**
- **오버레이**: 스켈레톤, 얼굴 박스/라벨, FPS, 상태 텍스트
- **핫키**: `b`(베이스라인) / `r`(해제) / `t`(녹화 토글) / `o`(90° 회전) / `q|ESC`(종료)
- **출력 저장**: 주석/원본 MP4, WAV, XML 이벤트 로그
- **이벤트 감지**: `negative_emotion`, `bad_posture`, `leg_shake`(강/중/약)

---

## ✅ 요구사항

- NVIDIA **Jetson Nano** (JetPack + CUDA + TensorRT)
- Python 3.x, OpenCV(`cv2`), NumPy, ALSA(`arecord`)
- TensorRT 엔진 2개
  - 얼굴 분류: `face_3cls.engine`
  - 포즈 추정: `pose_movenet.engine`

> 고정 엔진 경로(예시, 파일 위치에 맞게 수정)
>
> ```python
> FER_ENGINE_FIXED  = "/home/jetson10/fer/scripts/face_3cls.engine"
> POSE_ENGINE_FIXED = "/home/jetson10/fer/scripts/pose_movenet.engine"
> ```

---

## 🚀 빠른 시작

```bash
# /dev/video3, 640x480@30, FPS 표시
python3 run_3cls_audio.py --src 3 --width 640 --height 480 --fps 30 --show_fps
```

추가 예시:
```bash
# 기본값(640x480@30)으로 /dev/video3
python3 run_3cls_audio.py --src 3

# 1280x720 프리뷰
python3 run_3cls_audio.py --src 3 --width 1280 --height 720 --show_fps

# 파일 입력 디버깅
python3 run_3cls_audio.py --src sample.mp4 --show_fps

# XML 파일명에 타임스탬프 사용
python3 run_3cls_audio.py --src 3 --log_xml xml_$(date +%Y%m%d_%H%M%S).xml

# 오버레이(annot)만 자동 저장
python3 run_3cls_audio.py --src 3 --out_video mp4/annot_out.mp4
```

> `--src`: 정수면 `/dev/videoN`, 문자열이면 파일/URL로 처리

---

## 🎛 명령행 옵션

| 옵션 | 타입/예시 | 기본값 | 설명 |
|---|---|---:|---|
| `--src` | `3` 또는 `"/path/file.mp4"` | `3` | 카메라 장치 번호 또는 파일/URL |
| `--width` / `--height` | `640` / `480` | `640` / `480` | 캡처 해상도 |
| `--fps` | `30` | `30` | 목표 FPS(장치/드라이버에 따라 변동 가능) |
| `--show_fps` | 플래그 | 꺼짐/켬 | 화면 우상단 FPS 표시 |
| `--pose_thresh` | `0.3` | `0.3` | 포즈 키포인트 가시성 임계값 |
| `--rec_codec` | `mp4v` | `mp4v` | 녹화 fourcc |
| `--out_video` | `mp4/annot_out.mp4` | 빈값 | 지정 시 **주석 영상만** 자동 저장 |
| `--out_codec` | `mp4v` | `mp4v` | `--out_video` 코덱 |
| `--mp4_dir` / `--wav_dir` / `--xml_dir` | 경로 | `./mp4` / `./wav` / `./xml` | 저장 폴더(없으면 생성) |
| `--max_side` | `720` | `720` | 긴 변을 이 크기로 리사이즈(0=원본) |
| `--frame_skip` | `0` | `0` | n>0이면 n프레임 건너뜀 |
| `--process_every` | `1` | `1` | 파일 입력 디코더 프레임 스킵 비율 |
| `--sync_playback` | 플래그 | 꺼짐 | 파일 입력을 원래 FPS에 동기 |
| `--rotate` | `0/90/180/270` | `0` | 출력 회전 각도 |
| `--auto_rotate` | 플래그 | 꺼짐 | 세로 영상 자동 90° 회전 |
| `--neg_thresh` | `0.30` | `0.30` | negative 확률 임계 |
| `--bad_frames` / `--neg_frames` / `--leg_frames` | `8/8/12` | `8/8/12` | 이벤트 확정 프레임 수 |
| `--show_debug` | 플래그 | 꺼짐 | 디버그 메트릭 표시 |
| `--fer_gamma` / `--fer_clahe` / `--fer_tta_flip` / `--fer_temp` | `1.2`/flag/flag/`1.2` | `1.2`/끔/끔/`1.2` | FER 전처리/추론 튜닝 |
| `--fer_labels` | `"neg,neu,pos"` | `"neg,neu,pos"` | 라벨 순서 |
| `--mic_enable` | 플래그 | **켬** | 오디오 녹음 |
| `--mic_device` | `plughw:2,0` | `plughw:2,0` | ALSA 장치(`arecord -l` 참조) |
| `--mic_rate` / `--mic_channels` / `--mic_format` | `16000` / `1` / `S16_LE` | 동일 | 오디오 설정 |

---

## ⌨️ 실행 중 키

- **b**: 베이스라인 캡처(이후 이벤트 로깅 활성화)  
- **r**: 베이스라인 해제(로깅 비활성화)  
- **t**: 녹화 토글(주석/원본 비디오 + 오디오)  
- **o**: 90° 회전 토글  
- **q / ESC**: 종료

---

## 📦 산출물

- **비디오(mp4)**  
  - 주석: `mp4/rec_YYYYMMDD_HHMMSS_annot.mp4`  
  - 원본: `mp4/rec_YYYYMMDD_HHMMSS_raw.mp4`
- **오디오(wav)**  
  - `wav/rec_YYYYMMDD_HHMMSS.wav` (`arecord` 기반)
- **XML 이벤트 로그**  
  - 시작 시: `xml/xml_YYYYMMDD_HHMMSS.xml`  
  - 녹화 토글 시점마다 새 파일 생성(로깅 재시작)
- **이벤트 예시**  
  - `{"type":"bad_posture","timestamp_sec":"...","score":"...","frame":"..."}`  
  - `{"type":"negative_emotion","timestamp_sec":"...","prob":"...","frame":"..."}`  
  - `{"type":"leg_shake","timestamp_sec":"...","side":"L/R","freq_hz":"...","ratio":"...","amp":"...","frame":"..."}`

---

## 🧩 동작 흐름

1. 입력 열기: `/dev/video*`(정수) 또는 파일/URL(문자열)  
2. 포즈 추정 → 얼굴 ROI 검출(포즈 기반, 폴백: Haar)  
3. 얼굴 3클래스 확률·자세 라벨·다리 떨기 계산 및 오버레이  
4. **b** 이후 이벤트 기준 충족 시 XML 기록  
5. **t**로 주석/원본 비디오와 오디오 동시 저장

---

## 🛠 트러블슈팅

- `VIDEOIO ERROR: V4L2: setting property ... not supported` → 장치 미지원 옵션. 미리보기가 되면 무시 가능. 필요 시 해상도/FPS 조정  
- FPS 고정 안 됨 → 일부 UVC 장치 특성. 화면 표시 FPS 참고  
- 마이크 장치 확인 → `arecord -l`, `--mic_device plughw:<card>,<device>` 지정  
- TensorRT dtype → 전처리 dtype을 모델 스펙에 맞게 유지(캐스팅 오류 방지)  
- 디바이스 점유 → 다른 프로세스가 `/dev/video*` 사용 중이면 종료 후 실행

---

## 📁 폴더 구조(예시)

```
project_root/
├─ run_3cls_audio.py
├─ engines/
│  ├─ face_3cls.engine
│  └─ pose_movenet.engine
├─ mp4/
├─ wav/
└─ xml/
```

---

## 📄 라이선스

Internal/Project use. 필요 시 라이선스 문구를 추가하세요.
