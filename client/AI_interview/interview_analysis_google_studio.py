# 파일: interview_analysis_google_studio.py
import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from google import genai


# ----------------- 1. Google AI Studio API 키 설정 -----------------
client = genai.Client(api_key="AIzaSyDuI6DK3v17kGqqSyM4uHRWoC2qRC-Kzpg")

# ----------------- 2. 오디오 불러오기 -----------------
def load_audio(file_path, target_sr=16000):
    waveform, sr = torchaudio.load(file_path)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)
        sr = target_sr
    return waveform.squeeze(0).numpy(), sr

# ----------------- 3. 음성 분석 (Pitch, Volume) -----------------
def analyze_audio(waveform, sr=16000, frame_size=4096, hop_size=2048):
    # RMS 볼륨 계산
    volume = waveform ** 2
    avg_vol = np.mean(volume)
    vol_var = np.std(volume)
    
    # 프레임 단위 피치 계산
    def autocorr_pitch_frame(sig):
        pitches = []
        for start in range(0, len(sig)-frame_size, hop_size):
            frame = sig[start:start+frame_size]
            frame = frame - np.mean(frame)
            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr)//2:]
            d = np.diff(corr)
            start_idx = np.where(d > 0)[0]
            if len(start_idx) == 0:
                pitches.append(0)
                continue
            peak = np.argmax(corr[start_idx[0]:]) + start_idx[0]
            pitch = sr / peak if peak != 0 else 0
            pitches.append(pitch)
        return np.array(pitches)
    
    pitch_values = autocorr_pitch_frame(waveform)
    avg_pitch = np.mean(pitch_values[pitch_values>0])  # 0 제외
    pitch_var = np.std(pitch_values[pitch_values>0])
    
    return {
        "avg_pitch": avg_pitch,
        "pitch_var": pitch_var,
        "avg_vol": avg_vol,
        "vol_var": vol_var
    }

# ----------------- 4. Whisper로 음성 → 텍스트 -----------------
def transcribe_audio(file_path):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    waveform, sr = load_audio(file_path)
    input_features = processor(waveform, sampling_rate=sr, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# ----------------- 5. Google LLM 피드백 -----------------
def generate_feedback(transcript, analysis):
    # 1️⃣ LLM에게 전달할 prompt 정의
    prompt = f"""
면접 답변 분석:
- Transcript: {transcript}
- Pitch 평균: {analysis['avg_pitch']:.2f} Hz, 변동: {analysis['pitch_var']:.2f}
- 볼륨 평균: {analysis['avg_vol']:.3f}, 변동: {analysis['vol_var']:.3f}

위 데이터를 바탕으로 면접 답변의 목소리 톤, 안정성, 자신감 등을 평가하고,
1~10점 척도로 점수와 코멘트를 작성해줘.
"""

    # 2️⃣ 최신 1.39.1 SDK 기준 generate_content 사용
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # 3️⃣ 결과 텍스트 가져오기
    feedback = response.text
    print("=== LLM Feedback ===\n", feedback)
    return feedback

# ----------------- 6. 실행 -----------------
if __name__ == "__main__":
    audio_file = "sample.flac"  # 테스트용 음성 파일 경로
    
    print("🎤 오디오 분석 중...")
    waveform, sr = load_audio(audio_file)
    analysis = analyze_audio(waveform)
    
    print("📝 Whisper로 텍스트 변환 중...")
    transcript = transcribe_audio(audio_file)
    
    print("💡 Google LLM 피드백 생성 중...")
    feedback = generate_feedback(transcript, analysis)
    
    print("\n=== Transcript ===")
    print(transcript)
    print("\n=== Voice Analysis ===")
    for k, v in analysis.items():
        print(f"{k}: {v:.2f}")
    print("\n=== LLM Feedback ===")
    print(feedback)
