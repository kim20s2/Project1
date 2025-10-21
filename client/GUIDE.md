# Client (Feedback Generation) 

## 1) 주요 기능

* **면접 진행 UI**: 면접 시작, 답변 시작, 답변 종료, 면접 종료 총 4개의 버튼 구성 
* **녹음 제어**: 서버에 `/command/start_record`, `/command/stop_record` 호출
* **결과 자동 다운로드**: `/download/` 엔드포인트 통해 WAV, XML(자세/표정), mp4 결과 파일 다운로드
* **STT/피드백**: Whisper STT 및 랭체인을 통한 LLM(Gemini 2.5 Flash) 피드백 체인
* **음성 안정성(eGeMAPS)**: opensmile로 jitter/shimmer/HNR 등 추출
* **총평과 항목별 피드백**: 답변별 + 최종 리포트
##

## 2) 시스템 개요 & 데이터 플로우

```
[User] ─▶ (질문 재생/면접관 패널)
         └──requests──▶ [pi_server]
                      ├─ start_record / stop_record (오케스트레이션)
                      ├─ RTSP Router ─▶ Pose/Facial (Hailo‑8 or jetson nano) ─▶ XML(f_log/p_log)
                      └─ Recorder: MP4/WAV

[Client]
  ├─ Downloader: WAV/XML(+MP4) 수집(폴링+검증)
  ├─ STT: Whisper (로컬)
  ├─ LLM: Gemini 2.5 Flash (코칭/총평)
  └─ Report: 답변별 피드백 카드 + 최종 리포트(LLM)
  
```
## 3) 디렉터리 구조

```text
AI-interview/
 ├─ app/
 │   ├─ main_app.py               # Streamlit 엔트리
 │   ├─ interviewer.py            # 메인 섹션  
 │   ├─ upload_section.py         # 업로드 섹션        
 │   ├─ adapters/
 │   │   ├─ interviewer_adapters.py  # UI와 백엔드 연결 어뎁터
 │   │   └─ posture_adapters.py          # xml 자세 데이터 ─▶ LLM 연결 브릿지
 │   ├─ assets/ # 면접관 mp4 저장 디렉터리 
 │
 ├─ core/
 │   ├─ analysis_audio.py # opensmile기반 음정분석 라벨
 │   ├─ analysis_pose.py  # hailo 자세 분석 라벨 
 │   ├─ analysis_pose_jetson.py # jetson 자세 분석 라벨 
 │   ├─ chains.py # 랭체인 프롬프트 
 │   ├─ recording_io.py # 파일 저장 유틸
 │   ├─ whisper_run.py # whisper STT 구현 
 ├─ requirements.txt
 └─ README.md (본 파일) 

```
## 4) 데이터/결과물 구조

다운로드는 세션 폴더 기준으로 관리됩니다.

```text
./data/records
└─ <session_id>/
├─ audio_YYYYmmdd_HHMMSS.wav
├─ video_YYYYmmdd_HHMMSS.mp4
├─ log_YYYYmmdd_HHMMSS.xml
```
---

## 5) 인터페이스 계약 (Contracts)

###  REST API 

* `POST /command/start_record` — 녹화/분석 시작
* `POST /command/stop_record` — 녹화/분석 종료, XML Mixer 실행 → `log.xml`
* `GET  /download/wav/audio.wav` — WAV (Content-Type: `audio/*`)
* `GET  /download/mp4/video.mp4` — mp4
* `GET  /download/xml/log.xml`   — 자세 분석 xml

> 서버 구현에 따라 `GET`/`POST` 호환. 클라이언트는 메서드 차이를 어댑터에서 흡수.


## 6) 파일 포맷 & 네이밍 정책

* 산출물: `audio.wav`, `video.mp4`, `log.xml`
* 네이밍: **타임스탬프 포함**으로 덮어쓰기 방지

  * 예: `audio_YYYYmmdd_HHMMSS.wav`, `log_YYYYmmdd_HHMMSS.xml`
* 세션 디렉터리: `<DOWNLOAD_DIR>/<session_id>/...`
* 유효성 검증: Content-Type 검사 + 크기/프레임 검사



## 7) 기술 선택 및 근거 (Client)

| 영역        | 선택                      | 근거                            | 트레이드오프                     |
| --------- | ----------------------- | ----------------------------- | -------------------------- |
| UI 프레임워크  | **Streamlit**           | 빠른 프로토타이핑, 단순 상태 관리, 데모/배포 용이 | 복잡 라우팅/오프라인 캐시 한계          |
| HTTP/다운로드 | **requests + 스트리밍**     | 안정적 스트리밍, API 단순, 의존성 적음      | 대량 병렬엔 httpx/asyncio 대비 제약 |
| 설정        | **python‑dotenv(.env)** | 환경/비밀키 분리, 배포별 구성 용이          | 런타임 변경 반영 즉시성 낮음           |
| STT       | **Whisper(로컬)**         | 한/영 혼용 안정, 오프라인 대응, 높은 음성 인식           | 모델/지연 관리 필요                |
| LLM       | **Gemini 2.5 Flash**    | 빠른 응답 + 코칭 품질 균형              | 장문·고난도는 Pro 권장             |
| 음성 지표     | **openSMILE(eGeMAPS)**  | 표준 지표로 비교/트렌드 용이,매우 빠르고 정확한 feature 추출              | OS 의존 설치, 러닝커브             |
| 데이터 포맷    | **XML 수신 → 파서**         | 서버 로그 포맷과 호환, 스키마 진화 용이       | JSON 대비 가독성↓               |
| LLM 오케스트레이션 | **LangChain (LCEL/Runnable)** | 모델 교체/멀티 모델에 유연, `PromptTemplate/OutputParser`로 프롬프트·출력 스키마 표준화, 체인 재사용·조합(map/parallel) 쉬움, 툴·에이전트 연동 용이 | 러닝커브·추상화 오버헤드(직접 SDK 대비 지연/의존성 증가), API 변화 속도 빠름, 단순 호출엔 과함, 비동기/캐시/리트라이 설정 추가 부담 |




