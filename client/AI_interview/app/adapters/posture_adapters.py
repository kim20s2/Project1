import json, os, math
from core.analysis_pose import parse_posture_summary as parse_legacy, normalize_posture
from core.analysis_pose_jetson import parse_posture_summary_jetson

try:
    import numpy as np
except Exception:
    np = None

def _to_native_jsonable(x):
    """np.float32 등 NumPy 스칼라/컬렉션을 파이썬 내장 타입으로 변환."""
    # NumPy 스칼라 → 파이썬 스칼라
    if np is not None and isinstance(x, np.generic):
        x = x.item()

    if isinstance(x, float):
        # NaN/Inf는 None으로 치환 (LLM 입력 안전성)
        if not math.isfinite(x):
            return None
        return float(x)

    if isinstance(x, (int, str, bool)) or x is None:
        return x

    if isinstance(x, dict):
        return {str(k): _to_native_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_native_jsonable(v) for v in x]

    # 그 외(예: set, custom obj)는 문자열화 또는 리스트화 필요 시 확장 가능
    return str(x)



def _severity_by_ratio(count, total):
    if not total: return ""
    r = count / float(total)
    return "강" if r >= 0.40 else "중" if r >= 0.20 else "약" if r >= 0.05 else ""

def _to_common_from_legacy(norm: dict) -> dict:
    frames = int(norm.get("frames_total") or norm.get("frames") or 0)
    neg = int(norm.get("negative_emotion_count", 0))
    bad = int(norm.get("bad_posture_count", 0))
    leg = int(norm.get("leg_shake_count", 0))
    label = norm.get("label") or "정상"
    return {
        "frames": frames,
        "negative_emotion_count": neg,
        "bad_posture_count": bad,
        "leg_shake_count": leg,
        "negative_emotion_severity": _severity_by_ratio(neg, frames),
        "bad_posture_severity": _severity_by_ratio(bad, frames),
        "leg_shake_severity": _severity_by_ratio(leg, frames),
        "label": label,
    }

def parse_posture_auto(xml_path: str) -> tuple[str, dict, dict]:
    """
    return (flavor, common_for_llm, legacy_normalized)
      - flavor: 'jetson' | 'legacy' | 'unknown'
      - common_for_llm: 젯슨 스키마 형태(dict)
      - legacy_normalized: normalize_posture(...) 결과(레거시일 때만)
    """
    # 1) 젯슨 우선
    try:
        jet = parse_posture_summary_jetson(xml_path)
        if jet and isinstance(jet.get("frames"), int):
            return "jetson", jet, {}
    except Exception:
        pass
    # 2) 레거시로 폴백
    try:
        raw = parse_legacy(xml_path) or {}
        norm = normalize_posture(raw) or {}
        common = _to_common_from_legacy(norm)
        return "legacy", common, norm
    except Exception:
        pass
    return "unknown", {"frames": 0, "label": "데이터 없음"}, {}

def _rate(n, d): 
    return None if not d else round(float(n)/float(d), 4)

def build_nonverbal_json(voice: dict, posture_common: dict) -> str:
    frames = posture_common.get("frames") or 0
    payload = {
        "voice": {
            "stability_score": voice.get("stability_score"),
            "jitter": voice.get("jitter"),
            "shimmer": voice.get("shimmer"),
            "hnr": voice.get("hnr"),
            "label": voice.get("stability_label"),
        },
        "posture": {
            "frames": int(frames),
            "negative_emotion": {
                "count": posture_common.get("negative_emotion_count", 0),
                "rate": _rate(posture_common.get("negative_emotion_count", 0), frames),
                "severity": posture_common.get("negative_emotion_severity", ""),
            },
            "bad_posture": {
                "count": posture_common.get("bad_posture_count", 0),
                "rate": _rate(posture_common.get("bad_posture_count", 0), frames),
                "severity": posture_common.get("bad_posture_severity", ""),
            },
            "leg_shake": {
                "count": posture_common.get("leg_shake_count", 0),
                "rate": _rate(posture_common.get("leg_shake_count", 0), frames),
                "severity": posture_common.get("leg_shake_severity", ""),
            },
            "label": posture_common.get("label", "정상"),
        }
    }
    return json.dumps(_to_native_jsonable(payload), ensure_ascii=False)
