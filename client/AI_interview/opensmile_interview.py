import streamlit as st
import os
import numpy as np
import librosa
import scipy.ndimage
import tempfile

# ============================
# 0️⃣ API 키 설정
# ============================
# Google GenAI
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDuI6DK3v17kGqqSyM4uHRWoC2qRC-Kzpg")
# ============================
# 1️⃣ Whisper STT
# ============================

import whisper
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small")  # small/medium/large 선택

def run_whisper(audio_file):
    result = model.transcribe(audio_file, language="ko")
    return result["text"]
print("현재 장치:", device)
# ============================
# 2️⃣ 목소리 안정성 분석
# ============================
import opensmile
import pandas as pd

# 🔹 openSMILE 초기화
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def analyze_stability(audio_path: str):
    """음성 안정성 피처 추출"""
    result = smile.process_file(audio_path).iloc[0]

    features = {
        "jitter": result["jitterLocal_sma3nz_amean"],
        "shimmer": result["shimmerLocaldB_sma3nz_amean"],
        "hnr": result["HNRdBACF_sma3nz_amean"],
        "f0_std": result["F0semitoneFrom27.5Hz_sma3nz_stddevNorm"],
        "loudness_std": result["loudness_sma3_stddevNorm"],
    }
    return features, "ok"

def get_stability_score(jitter, shimmer,hnr):
      # 1️⃣ 기준값 완화 (실제 환경에 맞게)
    JITTER_REF = 0.05      # 현실 기준으로 약 3배 완화
    SHIMMER_REF = 0.45   # 기존 0.10보다 완화
    HNR_REF_GOOD = 15.0
    HNR_REF_BAD = 10.0

    # 2️⃣ 개별 점수 (0~10)
    # 낮을수록 좋은 jitter/shimmer → 1 - (value/ref)
    score_jitter = max(0, 1 - (jitter / JITTER_REF))
    score_shimmer = max(0, 1 - (shimmer / SHIMMER_REF))
    
    # HNR은 높을수록 좋음 → (value / 20)으로 스케일링
    # 20dB 이상이면 매우 좋은 음질
    score_hnr = min(1.0, max(0, hnr / 20.0))

    # 3️⃣ 가중 평균 (jitter/shimmer 40%, HNR 20%)
    score = (score_jitter * 0.3 + score_shimmer * 0.3 + score_hnr * 0.4) * 10

    # 4️⃣ 스케일 보정 (전체 점수 상향)
    score = min(10.0, (score * 1.2) + 1.0)
    # 4️⃣ 라벨 결정
    if score >= 8.0:
        label, color = "안정적 ✅", "success"
    elif score >= 5.0:
        label, color = "보통 ⚠️", "warning"
    else:
        label, color = "불안정 ❌", "error"

    return round(score, 2), label, color


# ============================
# 3️⃣ 자세 분석
# ============================
import xml.etree.ElementTree as ET

import xml.etree.ElementTree as ET

def parse_posture_summary(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    head_tilt_count = 0
    body_tilt_count = 0
    gesture_count = 0
    frame_total = 0

    for frame in root.findall("frame"):
        frame_total += 1
        analysis = frame.find("analysis")
        if analysis is None:
            continue

        for result in analysis.findall("result"):
            rtype = result.get("type", "")
            if rtype == "head_tilt":
                head_tilt_count += 1
            elif rtype == "body_tilt":
                body_tilt_count += 1
            elif rtype == "gesture":
                gesture_count += 1

    # 기본 라벨링
    labels = []
    if head_tilt_count > frame_total * 0.3:
        labels.append("머리 기울임 잦음")
    if body_tilt_count > frame_total * 0.3:
        labels.append("몸 기울어짐 잦음")

    # 제스처 라벨링
    if gesture_count == 0:
        gesture_label = "제스처 없음 (답변이 딱딱할 수 있음)"
    elif 1 <= gesture_count <= 3:
        gesture_label = "자연스러운 제스처 사용"
    else:
        gesture_label = "제스처 과다 (산만할 수 있음)"

    labels.append(gesture_label)

    return {
        "frames": frame_total,
        "head_tilt_count": head_tilt_count,
        "body_tilt_count": body_tilt_count,
        "gesture_count": gesture_count,
        "label": ", ".join(labels)
    }


# ============================
# 4️⃣ Google GenAI + LangChain
# ============================
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 전문 면접관 역할을 맡는다. 아래는 면접자의 행동 분석 데이터이다. 이 데이터에는 면접 중 감지된 자세, 표정, 음성의 특징이 포함된다."),
    ("human", """
면접 답변 분석:
- Transcript: {transcript}

[음성 분석 결과]
- 피치 흔들림 (Jitter): {jitter:.4f}
- 볼륨 흔들림 (Shimmer): {shimmer:.4f}
- 배음대잡음비 (HNR): {hnr:.2f} dB
- 피치 표준편차 (F0_std): {f0_std:.2f}
- 볼륨 표준편차 (Loudness_std): {loudness_std:.2f}
- 전반적 평가: {label}

자세 및 제스처 활용: {posture}

요구사항:
1. 데이터를 기반으로 면접자의 행동을 **객관적으로 요약**한다. (숫자, 횟수 포함)
2. 면접관의 입장에서 면접자에게 도움이 될 **피드백을 2~3문장**으로 제시한다.  
   - 긍정적인 부분을 먼저 짚고, 개선할 점을 제안한다.  
   - 너무 공격적이지 말고 **친절하고 전문적인 톤**을 유지한다.  
3. 피드백은 **간결하고 실용적으로** 작성한다.

출력 형식:
- 분석 요약: ...
- 강점: ...
- 피드백: ...
""")
])

chain = LLMChain(llm=llm, prompt=prompt)

chain = LLMChain(llm=llm, prompt=prompt)

# ============================
# streamlit 실행
# ============================
st.set_page_config(page_title="AI 면접관", page_icon="🎤", layout="wide")

# 타이틀 + 서브타이틀
st.markdown(
    """
    <h1 style='text-align: center; color: #1E90FF; font-size: 40px;'>
        🎤 AI 면접관
    </h1>
    <h3 style='text-align: center; color: #555;'>
        당신의 <b>목소리 · 자세 · 표정</b>을 종합 분석합니다
    </h3>
    """,
    unsafe_allow_html=True
)


st.divider()

# ----------------------
# 세션 상태 초기화
# ----------------------
if "chapters" not in st.session_state:
    st.session_state.chapters = []  # 전체 면접 기록 (세션별)
if "history" not in st.session_state:
    st.session_state.history = []   # 현재 세션 기록

# ----------------------
# 새 면접 세션 시작 버튼
# ----------------------
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "posture_key" not in st.session_state:
    st.session_state.posture_key = 0

if st.button("🆕 새로운 면접 시작"):
    if st.session_state.history:
        st.session_state.chapters.append(st.session_state.history)
    st.session_state.history = []
    st.session_state.uploader_key += 1  # key 바꿔줌
    st.session_state.posture_key += 1
    st.rerun()


# 서비스 소개 카드
st.markdown(
    """
    <div style="
        background-color:#f0f8ff;
        padding:20px;
        border-radius:12px;
        text-align:center;
        border: 1px solid #1E90FF;
        margin-bottom:15px;
    ">
        <h3 style="color:#1E90FF;">🎙️ 면접 답변 업로드</h3>
        <p style="color:#333;">지원자의 음성과 자세를 업로드해주세요.<br>
        지원 형식: <b>WAV, M4A, MP3, FLAC, XML</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader(
    "🎙️ 음성 데이터",
    type=["wav","m4a","mp3","flac"],
    key=f"file_uploader_{st.session_state.uploader_key}"  # key 매번 바뀜
)

posture_file = st.file_uploader(
    "🧍 자세 데이터",
    type=["xml"],
    key=f"posture_uploader_{st.session_state.posture_key}" # key 매번 바뀜
)
summary = None
if posture_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_xml:
        tmp_xml.write(posture_file.getbuffer())
        summary = parse_posture_summary(tmp_xml.name)  # 자세 분석 함수 사용

if uploaded_file and posture_file:
    st.success(f"✅ 파일 업로드 완료: {uploaded_file.name, posture_file.name}")
    st.audio(uploaded_file)
    # 임시 파일로 저장 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    with st.spinner("분석 중..."):
        # 1️⃣ 음성 텍스트 변환
        transcript = run_whisper(tmp_path)

        # 2️⃣ openSMILE 기반 피처 추출
        features, _ = analyze_stability(tmp_path)
        jitter = features["jitter"]
        shimmer = features["shimmer"]

        # 3️⃣ 안정성 점수 계산
        stability_score, voice_label, color = get_stability_score(
        jitter=features["jitter"],
        shimmer=features["shimmer"],
        hnr=features["hnr"]
    )

        # 4️⃣ LLM 피드백 생성
        feedback = chain.run({
            "transcript": transcript,
            "jitter": features["jitter"],
            "shimmer": features["shimmer"],
            "hnr": features["hnr"],
            "f0_std": features["f0_std"],
            "loudness_std": features["loudness_std"],
            "label": voice_label,
            "posture": summary["label"] if summary else "데이터 없음"
        })
        

        # 5️⃣ 결과 저장
        result = {
            "transcript": transcript,
            "jitter": jitter,
            "shimmer": shimmer,
            "stability_score": stability_score,
            "voice_label": voice_label,
            "color": color,
            "posture": summary["label"] if summary else "데이터 없음",
            "feedback": feedback
        }

        st.session_state.history.append(result)


def render_question_result(i, res):
    st.markdown(f"### ❓ 답변 {i}")
    st.info(res["transcript"])

    col1, col2 = st.columns([1, 2])
    with col1:
        # openSMILE 기반 주요 피처 표시
        st.metric("🎚️ Jitter (피치 흔들림)", f"{res['jitter']:.4f}")
        st.metric("🔉 Shimmer (볼륨 흔들림)", f"{res['shimmer']:.4f}")
    with col2:
        # 안정성 점수 시각화 (10점 환산)
        score = float(res.get("stability_score", 0))
        st.metric("목소리 안정성 점수", f"{score:.2f}/10")
        st.progress(score / 10)

    # 안정성 라벨 색상 표시
    label = res.get("voice_label", "데이터 없음")
    color = res.get("color", "warning")

    if color == "success":
        st.success(f"✅ 목소리 안정성: {label}")
    elif color == "warning":
        st.warning(f"⚠️ 목소리 안정성: {label}")
    else:
        st.error(f"❌ 목소리 안정성: {label}")

    if "posture" in res:
        if "안정적" in res["posture"]:
            st.success(f" 자세: {res['posture']}")
        elif "불안정" in res["posture"] or "기울어짐" in res["posture"]:
            st.error(f" 자세: {res['posture']}")
        else:
            st.warning(f" 자세: {res['posture']}")

    st.success(f"{res['feedback']}")
    st.divider()


# ---------------- 결과 출력 ----------------
if st.session_state.history or st.session_state.chapters:
    st.subheader("📂 면접 기록")

    # ✅ 현재 세션
    if st.session_state.history:
        st.markdown("## 🚀 현재 진행중인 면접")
        for i, res in enumerate(st.session_state.history, 1):
            render_question_result(i, res)

    # ✅ 과거 세션 (expander로 접기)
    for c_idx, chapter in enumerate(st.session_state.chapters, 1):
        with st.expander(f"📌 과거 면접 세션 {c_idx}", expanded=False):
            for i, res in enumerate(chapter, 1):
                render_question_result(i, res)


# ----------------------
# 푸터
# ----------------------

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        © 2025 AI Interview Project | Powered by Whisper · LangChain · Hailo
    </div>
    """,
    unsafe_allow_html=True
)