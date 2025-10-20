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

import numpy as np
import librosa
import scipy.ndimage

def analyze_stability(audio_file):
    y, sr = librosa.load(audio_file, sr=16000)

    # --- Pitch 분석 ---
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) == 0: 
        pitch_values = np.array([1.0])
    pitch_values = scipy.ndimage.median_filter(pitch_values, size=3)

    jitter_std = np.std(pitch_values) / np.mean(pitch_values)
    jitter_range = (np.percentile(pitch_values, 95) - np.percentile(pitch_values, 5)) / np.mean(pitch_values)
    jitter = 0.7 * jitter_std + 0.3 * jitter_range

    # --- 볼륨 분석 ---
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    volume_values = rms[rms > 0]
    if len(volume_values) == 0: 
        volume_values = np.array([1.0])
    volume_values = scipy.ndimage.median_filter(volume_values, size=3)

    shimmer_std = np.std(volume_values) / np.mean(volume_values)
    shimmer_range = (np.percentile(volume_values, 95) - np.percentile(volume_values, 5)) / np.mean(volume_values)
    shimmer = 0.7 * shimmer_std + 0.3 * shimmer_range

    # --- 안정성 지표 ---
    instability = 0.5*jitter + 0.5*shimmer

    # 고정 범위 (실험적으로 0~5.0)
    inst_min, inst_max = 0.0, 5.0

    score = 10 * (1 - (instability - inst_min) / (inst_max - inst_min))
    score = round(max(0, min(10, score)), 1)

    print(f"instability={instability:.4f}, score={score}")

    return score


def label_from_stability(score):
    if score >= 6: return "안정적 ✅","success"
    elif score >= 3: return "보통 ⚠️","warning"
    else: return "불안정 ❌","error"

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
    ("system", "너는 전문 면접관 역할을 맡는다.  아래는 면접자의 행동 분석 데이터이다. 이 데이터에는 면접 중 감지된 자세, 표정, 음성의 특징이 포함된다"),
    ("human", """
면접 답변 분석:
- Transcript: {transcript}
- 목소리 안정성 지수: {voice_stability}
- 자세 및 제스처 활용: {posture}

요구사항:
1. 데이터를 기반으로 면접자의 행동을 **객관적으로 요약**한다. (숫자, 횟수 포함)  
2. 면접관의 입장에서 면접자에게 도움이 될 **피드백을 2~3문장**으로 제시한다.  
   - 피드백은 긍정적인 부분을 먼저 짚고, 개선할 점을 제안한다.  
   - 너무 공격적이지 말고 **친절하고 전문적인 톤**을 유지한다.  
3. 피드백은 **간결하고 실용적으로** 작성한다.  
     
출력 형식:
- 분석 요약: ...
- 강점: ...
- 피드백: ...
""")
])

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
        transcript = run_whisper(tmp_path)
        score = analyze_stability(tmp_path)
        voice_label, color = label_from_stability(score)
        feedback = chain.run({
            "transcript": transcript,
            "voice_stability": voice_label,
            "posture" : summary["label"]
        })
        # 결과 저장
        result = {
            "transcript": transcript,
            "score": score,
            "voice_label": voice_label,
            "color": color,
            "posture": summary["label"] if summary else "데이터 없음",
            #"expression": expression_label,
            "feedback": feedback
        }
        st.session_state.history.append(result)


def render_question_result(i, res):
    st.markdown(f"### ❓ 답변 {i}")
    st.info(res["transcript"])

    col1, col2, = st.columns([1, 2])
    with col1:
        st.metric("목소리 안정성 점수", f"{res['score']:.2f}/10")
    with col2:
        st.progress(float(res["score"])/10)

    if res["color"] == "success":
        st.success(f"목소리 안정성: {res['voice_label']}")
    elif res["color"] == "warning":
        st.warning(f"목소리 안정성: {res['voice_label']}")
    else:
        st.error(f"목소리 안정성: {res['voice_label']}")

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