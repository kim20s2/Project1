# ✅ 요구사항

NVIDIA Jetson Nano (JetPack + CUDA + TensorRT)

Python 3.x 환경 (OpenCV cv2, NumPy, ALSA arecord 사용 가능)

TensorRT 엔진 파일(예: engines/face_3cls.engine, engines/pose_movenet.engine)

엔진/모델 경로는 스크립트 상단 상수 또는 인자로 조정 가능합니다.

# 🚀 빠른 시작

USB 카메라 /dev/video3를 640×480@30으로 열고 FPS를 표시:

python3 run_3cls_audio.py --src 3 --width 640 --height 480 --fps 30 --show_fps


자주 쓰는 예시:

1) 기본값(640x480@30)으로 /dev/video3 열기
python3 run_3cls_audio.py --src 3

2) 1280x720 프리뷰
python3 run_3cls_audio.py --src 3 --width 1280 --height 720 --show_fps

3) 파일 재생으로 디버깅
python3 run_3cls_audio.py --src sample.mp4 --show_fps

4) XML 파일명을 타임스탬프로 저장
python3 run_3cls_audio.py --src 3 --log_xml xml_$(date +%Y%m%d_%H%M%S).xml

# 🎛 옵션 (CLI)
옵션	타입/예시	기본값	설명
--src	3, "/path/file.mp4"	3	정수 → /dev/videoN 장치, 문자열 → 파일/URL 경로
--width	640	640	캡처 가로 해상도
--height	480	480	캡처 세로 해상도
--fps	30	30	목표 FPS (장치/드라이버에 따라 고정 안될 수 있음)
--show_fps	플래그	끔/켬	화면 우상단 FPS 표시
--mic_enable	플래그	켬	마이크 녹음 사용
--mic_device	plughw:2,0	plughw:2,0	ALSA 장치 지정 (arecord -l로 확인)
--out_video	mp4/annot_only.mp4	자동파일명	주석 영상 출력 파일명(지정 시 명시 저장)
--log_xml	xml/custom.xml	xml/session_events.xml	이벤트 로그 XML 경로(파일명만 주면 xml/ 아래 저장)

# ⌨️ 실행 중 키 조작

b : 베이스라인 캡처(이후 이벤트 로깅 활성화)

r : 베이스라인 해제(로깅 비활성화)

t : 녹화 토글(주석/원본 비디오 + 오디오)

o : 90° 회전 토글

q / ESC : 종료

화면 하단에 [b] baseline [r] reset [t] record [o] rotate 90 힌트가 표시됩니다.

# 📦 산출물

비디오 (mp4)

주석: mp4/rec_YYYYMMDD_HHMMSS_annot.mp4

원본: mp4/rec_YYYYMMDD_HHMMSS_raw.mp4

오디오 (wav)

wav/rec_YYYYMMDD_HHMMSS.wav (ALSA arecord)

이벤트 로그 (XML)

기본: xml/session_events.xml

권장: --log_xml xml_$(date +%Y%m%d_%H%M%S).xml 로 타임스탬프 파일명 사용

이벤트 태그 예시

negative_emotion, bad_posture, leg_shake 를 강/중/약 단계로 기록

# 🧩 동작 개요

OpenCV로 /dev/video* 또는 파일/URL을 열어 프레임 수신

얼굴/자세 TensorRT 엔진 전처리 → 추론 → 후처리

박스/라벨/스켈레톤 및 FPS 오버레이

b 키 이후 이벤트를 XML로 기록

t 키로 주석/원본 비디오 + 오디오 동시 녹화
