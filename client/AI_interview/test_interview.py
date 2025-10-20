import streamlit as st
import os
import numpy as np
import librosa
import scipy.ndimage

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
def analyze_voice(audio_file):
    """
    음성 파일을 분석하여 피치(Jitter), 볼륨(Shimmer), 안정성(Stability) 점수를 계산합니다.
    점수는 자연스러운 면접 피드백 기준으로 0~10점 척도로 환산됩니다.
    """
    # 오디오 로드
    y, sr = librosa.load(audio_file, sr=16000)

    # ------------------ 피치 ------------------
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    if len(pitch_values) == 0:
        pitch_values = np.array([1.0])
    pitch_values = scipy.ndimage.median_filter(pitch_values, size=5)

    jitter = np.std(pitch_values) / np.mean(pitch_values)

    # ------------------ 볼륨 ------------------
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    volume_values = rms[rms > 0]
    if len(volume_values) == 0:
        volume_values = np.array([1.0])
    volume_values = scipy.ndimage.median_filter(volume_values, size=5)

    shimmer = np.std(volume_values) / np.mean(volume_values)

    # ------------------ 안정성 점수 ------------------
    stability_raw = 1 / (1 + jitter + shimmer)
    stability_score = round(max(0, min(10, stability_raw * 10)), 1)
    return stability_score

# ============================
# 3️⃣ Google GenAI + LangChain
# ============================
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=GOOGLE_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 면접관이야. 목소리 떨림, 안정성을 기반으로 잘 대답하고 있는지 평가해."),
    ("human", """
면접 답변 분석:
- Transcript: {transcript}
- 목소리 안정성 지수: {stability_score:.2f}

위 데이터를 바탕으로 목소리 안정성, 떨림 정도, 자신감 여부를 평가하고,
피드백 코멘트를 간단하게 작성해줘.
     
출력형식은 두 줄정도로 해줘 
""")
])

chain = LLMChain(llm=llm, prompt=prompt)

# ============================
# 4️⃣ 실행
# ============================
if __name__ == "__main__":
    audio_path = "test1.m4a"  # 분석할 면접 음성 파일

    print("🎙️ Whisper로 음성 → 텍스트 변환 중...")
    transcript = run_whisper(audio_path)

    print("변환된 텍스트:", transcript)

    print("📊 음성 안정성 분석 중...")
    stability_score = analyze_voice(audio_path)

    print("💡 Google LLM 피드백 생성 중...")
    feedback = chain.run({
    "transcript": transcript,
    "stability_score": stability_score  # 그냥 숫자
})

    print("\n=== 최종 면접 피드백 ===\n")
    print(feedback)
