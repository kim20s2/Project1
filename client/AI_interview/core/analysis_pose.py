import xml.etree.ElementTree as ET
from typing import Any, Dict

def parse_posture_summary(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    head_tilt_count = body_tilt_count = gesture_count = frame_total = 0

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

    labels = []
    if frame_total > 0:
        if head_tilt_count > frame_total * 0.3:
            labels.append("머리 기울임 잦음")
        if body_tilt_count > frame_total * 0.3:
            labels.append("몸 기울어짐 잦음")

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
def _norm_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return 0

def normalize_posture(d: Dict[str, Any]) -> Dict[str, int]:
    """
    다양한 XML/JSON 스키마를 집계용 키로 통일.
    반환 키: frames, head_tilt_count, body_tilt_count, gesture_count (모두 int)
    """
    if not isinstance(d, dict):
        return {"frames": 0, "head_tilt_count": 0, "body_tilt_count": 0, "gesture_count": 0}

    def pick(*keys, default=0):
        # 1) 루트에서 찾기
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        # 2) 흔한 중첩 노드에서 찾기
        for sub in ("stats", "summary", "meta", "data"):
            subd = d.get(sub)
            if isinstance(subd, dict):
                for k in keys:
                    if subd.get(k) is not None:
                        return subd[k]
        return default

    frames  = pick("frames", "frames_total", "total_frames")
    head    = pick("head_tilt_count", "tilt_head", "headTilts", "head_tilt")
    body    = pick("body_tilt_count", "tilt_body", "bodyTilts", "body_tilt")
    gesture = pick("gesture_count", "gestures", "gesture_total", "gesturePerFrame")

    return {
        "frames":           _norm_int(frames),
        "head_tilt_count":  _norm_int(head),
        "body_tilt_count":  _norm_int(body),
        "gesture_count":    _norm_int(gesture),
    }


