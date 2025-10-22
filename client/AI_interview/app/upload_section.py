from pathlib import Path
import tempfile, os,streamlit as st
from core.whisper_run import transcribe_file
from core.analysis_pose import parse_posture_summary
from core.analysis_audio import analyze_stability, get_stability_score
from core.analysis_pose_jetson import parse_posture_summary_jetson

# ── 업로드 모드(배치 분석) ────────────────────────────────────────────
def render_upload_section(*, ss, whisper_model, feedback_chain):
    with st.expander("📤 업로드 모드", expanded=False):
        # 세션별 기록 보관
        ss.setdefault("chapters", [])
        ss.setdefault("history", [])
        ss.setdefault("uploader_key", 0)
        ss.setdefault("posture_key", 0)

        if st.button("🆕 새로운 면접 시작"):
            if ss["history"]:
                ss["chapters"].append(ss["history"])
            ss["history"] = []
            ss["uploader_key"] += 1
            ss["posture_key"] += 1
            st.rerun()

        st.markdown(
            """
            <div style="
                background-color:#f0f8ff; padding:20px; border-radius:12px;
                text-align:center; border: 1px solid #1E90FF; margin-bottom:15px;">
                <h3 style="color:#1E90FF;">🎙️ 면접 답변 업로드</h3>
                <p style="color:#333;">지원자의 음성과 자세를 업로드해주세요.<br>
                지원 형식: <b>WAV, M4A, MP3, FLAC, XML</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader("🎙️ 음성 데이터", type=["wav","m4a","mp3","flac"], key=f"file_uploader_{ss['uploader_key']}")
        posture_file  = st.file_uploader("🧍 자세 데이터", type=["xml"], key=f"posture_uploader_{ss['posture_key']}")

        # 자세 요약
        posture_summary = None
        if posture_file:
            xml_bytes = posture_file.getvalue()  # ← 한 번에 바이트로
            if not xml_bytes or not xml_bytes.strip():
                st.error("XML 파일이 비어 있습니다.")
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_xml:
                    tmp_xml.write(xml_bytes)
                    tmp_xml.flush()
                    os.fsync(tmp_xml.fileno())
                    tmp_xml_path = tmp_xml.name  # 경로만 꺼내고

                # ★ 파일이 닫힌 뒤 파싱
                try:
                    posture_summary = parse_posture_summary_jetson(tmp_xml_path)
                except Exception as e:
                    st.error(f"XML 파싱 실패: {e}")

        if uploaded_file and posture_file:
            st.success(f"✅ 파일 업로드 완료: {uploaded_file.name}, {posture_file.name}")
            st.audio(uploaded_file)

            # 임시 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                wav_path = tmp_file.name

            with st.spinner("분석 중..."):
                # 1) Whisper 전사
                whisper_res = transcribe_file(ss["whisper_model"], wav_path, language="ko")
                transcript = whisper_res.get("text", "")
                
                # 2) 음성 피처
                features, _ = analyze_stability(wav_path)
                jitter   = features["jitter"]
                shimmer  = features["shimmer"]
                hnr      = features["hnr"]

                # 3) 안정성 점수
                stability_score, voice_label, color = get_stability_score(jitter=jitter, shimmer=shimmer, hnr=hnr)

                # 4) LLM 피드백
                feedback = ss["feedback_chain"].run({
                    "transcript": transcript,
                    "stability_score": stability_score,
                    "label": voice_label,
                    "posture": posture_summary["label"] if posture_summary else "데이터 없음"
                })

                # 5) 결과 저장(메모리)
                result = {
                    "transcript": transcript, "jitter": jitter, "shimmer": shimmer,
                    "stability_score": stability_score, "voice_label": voice_label, "color": color,
                    "posture": posture_summary["label"] if posture_summary else "데이터 없음",
                    "feedback": feedback
                }
                ss["history"].append(result)

        # 결과 렌더
        def render_question_result(i, res):
            st.markdown(f"### ❓ 답변 {i}")
            st.info(res["transcript"])

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("🎚️ Jitter (피치 흔들림)", f"{res['jitter']:.4f}")
                st.metric("🔉 Shimmer (볼륨 흔들림)", f"{res['shimmer']:.4f}")
            with c2:
                score = float(res.get("stability_score", 0))
                st.metric("목소리 안정성 점수", f"{score:.2f}/10")
                st.progress(min(1.0, score/10))

            label = res.get("voice_label", "데이터 없음")
            color = res.get("color", "warning")
            if color == "success":   st.success(f"✅ 목소리 안정성: {label}")
            elif color == "warning": st.warning(f"⚠️ 목소리 안정성: {label}")
            else:                    st.error(f"❌ 목소리 안정성: {label}")

            if "posture" in res:
                text = res["posture"]
                if "안정적" in text: st.success(f" 자세: {text}")
                elif ("불안정" in text) or ("기울어짐" in text): st.error(f" 자세: {text}")
                else: st.warning(f" 자세: {text}")

            st.success(res["feedback"])
            st.divider()

        if ss["history"] or ss["chapters"]:
            st.subheader("📂 면접 기록")
            if ss["history"]:
                st.markdown("## 🚀 현재 진행중인 면접")
                for i, res in enumerate(ss["history"], 1):
                    render_question_result(i, res)
            for c_idx, chapter in enumerate(ss["chapters"], 1):
                with st.expander(f"📌 과거 면접 세션 {c_idx}", expanded=False):
                    for i, res in enumerate(chapter, 1):
                        render_question_result(i, res)
