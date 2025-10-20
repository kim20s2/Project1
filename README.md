✅ 요구사항

NVIDIA Jetson Nano (JetPack + CUDA + TensorRT 구성)

Python 3.x + OpenCV (cv2), NumPy, PyAudio/arecord 사용 가능 환경

TensorRT 엔진 파일(예: face_3cls.engine, pose_movenet.engine)이 스크립트에서 지정된 경로에 존재

※ 엔진/모델 경로는 코드 상단 상수 또는 인자에서 조정 가능. OpenCV는 V4L2 백엔드를 통해 /dev/video*를 엽니다.

🚀 빠른 시작 (One-liner)

USB 카메라 /dev/video3를 640×480@30으로 열고 FPS 오버레이를 표시:

python3 run_3cls_audio.py --src 3 --width 640 --height 480 --fps 30 --show_fps


--src: 정수면 /dev/videoN 장치로 인식, 문자열이면 파일/URL 경로로 인식

--width/--height/--fps: 캡처 해상도와 프레임레이트 설정

--show_fps: 화면 우상단에 FPS 표시

자주 쓰는 축약 예시:

# 1) 기본값(640x480@30)으로 /dev/video3 열기
python3 run_3cls_audio.py --src 3

# 2) 1280x720으로 미리보기 + 녹화 토글은 실행 중 't' 키
python3 run_3cls_audio.py --src 3 --width 1280 --height 720

# 3) 파일 재생으로 디버깅
python3 run_3cls_audio.py --src sample.mp4 --show_fps

🎛 주요 옵션 (CLI)
옵션	타입/예시	기본값	설명
--src	3, "/path/file.mp4"	3	정수 → /dev/videoN 장치. 문자열 → 파일/스트림 경로
--width	640	640	캡처 가로 해상도
--height	480	480	캡처 세로 해상도
--fps	30	30	목표 캡처 FPS(장치/드라이버에 따라 고정 안될 수 있음)
--show_fps	플래그	끔/켬	화면에 FPS 오버레이
--mic_enable	플래그	켬	마이크 녹음 사용 여부
--mic_device	plughw:2,0	plughw:2,0	ALSA 장치. arecord -l로 카드/디바이스 확인
--out_video	mp4/annot_only.mp4	자동파일명	주석 영상 출력 파일명(지정 시 명시적 저장)
--log_xml	xml/custom.xml	xml/session_events.xml	이벤트 로그 XML 파일 경로(파일명만 주면 xml/ 아래 저장)

마이크: USB 사운드카드/내장마이크 사용 시 --mic_device를 알맞게 지정하세요. 예) plughw:1,0, plughw:2,0

⌨️ 실행 중 키 조작

b : 베이스라인 캡처(이후 이벤트 로깅 활성화)

r : 베이스라인 해제(로깅 비활성화)

t : 녹화 토글(주석 영상 + 원본 영상, 오디오 동시)

o : 90° 회전 토글

q/ESC : 종료

화면 하단에 힌트로 [b] baseline [r] reset [t] record [o] rotate 90 가 표시됩니다.

📦 산출물(기본 경로 & 형식)

비디오(mp4)

주석 영상: mp4/rec_YYYYMMDD_HHMMSS_annot.mp4

원본 영상: mp4/rec_YYYYMMDD_HHMMSS_raw.mp4

오디오(wav)

wav/rec_YYYYMMDD_HHMMSS.wav (ALSA arecord 기반)

이벤트 로그(XML)

기본: xml/session_events.xml

권장: 실행 인자로 타임스탬프 포함 파일명 지정

python3 run_3cls_audio.py --src 3 --log_xml xml_$(date +%Y%m%d_%H%M%S).xml


로그 내용(예)

negative_emotion, bad_posture, leg_shake 등 이벤트를 강/중/약으로 기록

세션 요약(발생 비율/빈도)은 후처리에서 활용 가능

🧩 동작 개요

OpenCV로 /dev/video* 또는 파일/URL을 열어 프레임을 수신

얼굴/자세 엔진(TensorRT) 전처리 → 추론 → 후처리

오버레이(박스/라벨/스켈레톤)와 FPS를 화면에 표시

b 키 이후 이벤트가 발생하면 XML로 기록

t 키로 녹화를 시작/종료(주석/원본, 오디오 동시)

🛠 트러블슈팅

VIDEOIO ERROR: V4L2: setting property #.. not supported
장치가 해당 속성 설정을 지원하지 않는 경우입니다. 무시해도 미리보기가 동작하면 문제 없습니다.
필요 시 해상도/FPS를 장치가 보장하는 값으로 낮춰보세요.

해상도/프레임이 고정되지 않음
일부 UVC 장치는 요청한 해상도/FPS를 정확히 고정하지 못합니다. 실제 FPS는 화면 표시값을 참고하세요.

마이크 장치 확인
arecord -l 로 카드/디바이스 번호 확인 후 --mic_device plughw:<card>,<device> 지정.

dtype 불일치 에러(TRT 입출력)
입력 텐서는 보통 float32입니다. 전처리에서 dtype을 맞추고, np.copyto 시 캐스팅 오류가 없도록 유지하세요.

디바이스 점유/충돌
다른 프로세스에서 /dev/video*를 쓰고 있으면 스트림이 열리지 않을 수 있습니다. 점유 프로세스를 종료하세요.

📁 폴더 구조(예시)
project_root/
├─ run_3cls_audio.py
├─ mp4/
├─ wav/
├─ xml/
└─ engines/
   ├─ face_3cls.engine
   └─ pose_movenet.engine


엔진 파일명/경로는 실제 보유 엔진에 맞게 조정하세요.
