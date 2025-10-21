import os
from functools import lru_cache
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# 업로드 모드 프롬프트 
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 전문 면접관이다. 아래는 면접자의 행동 분석 결과이다. 음성, 자세, 제스처 등의 데이터를 참고하여 면접자의 발성과 태도를 평가하고 피드백을 제공하라."),
    ("human", """
면접 답변:
- Transcript: {transcript}

[음성 분석 요약]
- 안정성 점수: {stability_score:.2f}/10
- 전반적 평가 등급: {label}

[자세 및 제스처]
- {posture}

요구사항:
1. 위 데이터를 참고해 면접자의 **발성 안정성, 자신감, 전달력, 태도,자세에 대한 것 까지**를 종합적으로 평가하라.  
2. Jitter, Shimmer, HNR 같은 기술적 용어는 언급하지 말고,  
   **청각적으로 느껴지는 인상**으로 표현하라.  
3. 피드백은 **3문장 이내** (강점→개선→긍정 마무리).  
4. 말투는 **전문적이고 따뜻하게**, 실제 면접관처럼 작성.

출력 형식:
- 분석 요약: ...
- 강점: ...
- 피드백: ...
""")
])

def build_feedback_chain(*, api_key: str, model: str = "gemini-2.5-flash", temperature: float = 0.6):
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=api_key)
    return LLMChain(llm=llm, prompt=prompt)
@lru_cache(maxsize=1)
def get_feedback_chain() -> LLMChain:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY 미설정")
    return build_feedback_chain(api_key=key)

# --- 총평용 프롬프트 (UI에서 render(...)해 문자열로 넘김) ---
SESSION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "너는 전문 면접관이다. 아래 세션 요약 데이터를 바탕으로 한국어로 면접에 대한 총평을 제공하라."),
    ("human",
     "【음성 요약】\n{voice_summary_json}\n\n"
     "【자세 요약】\n{posture_summary_json}\n\n"
     "【최근 답변들】\n{history_compact_json}\n\n"
     "총 {answers}개 답변을 기반으로:\n"
     "1) 전반 총평(1~2문장)\n2) 강점(불릿 2개)\n3) 개선 제안(불릿 2개)\n"
     "**음성(발성/전달력, 음정/억양/볼륨)**과 **자세(시선/기울임/제스처)**에 대한 코멘트를 각각 최소 1문장씩 반드시 포함해서 총 2~3문장 정도로.\n"
     "기술지표명(jitter/shimmer/HNR)은 언급하지 말 것.")
])
# --- 답변별 코칭용 프롬프트  ---
LIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "너는 실시간 면접 코치다. 최신 발화와 비언어 신호 요약을 보고, "
     "한국어로 2문장 정도로 간결하게 해당 답변에 대한 것을 코칭해줘."),
    ("human",
     "[이번 발화]\n{latest_utt}\n\n[비언어 요약]\n{nonverbal}\n\n"
     "요구사항: 2문장. 과한 칭찬/사족 금지. 속도/명료성/구조/톤 중 1~2개만 짚기.")
])

# 총평용, 답변별 프롬프트 선택 함수
def get_prompt(name: str) -> ChatPromptTemplate:
    if name == "session":
        return SESSION_PROMPT
    if name == "live":
        return LIVE_PROMPT
    raise KeyError(f"unknown prompt: {name}")

# --- 라이브(답변별) 체인 (adapters.my_feedback에서 사용) ---
@lru_cache(maxsize=1)
def get_live_feedback_chain(model: str = "gemini-2.5-flash", temperature: float = 0.6) -> LLMChain:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY 미설정")
    llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=key)
    return LLMChain(llm=llm, prompt=LIVE_PROMPT)

# --- 총평용: 문자열 프롬프트를 직접 넣어서 호출 (LangChain LLM만 사용) ---
@lru_cache(maxsize=1)
def _get_session_llm(model: str = "gemini-2.5-flash", temperature: float = 0.5):
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY 미설정")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, api_key=key)

def call_llm(payload) -> str:
    """
    payload: str 또는 List[BaseMessage] (ChatPromptTemplate.format_messages 결과)
    """
    llm = _get_session_llm()
    resp = llm.invoke(payload)  # str도 되고, messages 리스트도 됨
    text = getattr(resp, "content", None) or getattr(resp, "text", None) or str(resp)
    return text.strip()
