import xml.etree.ElementTree as ET

_THRESHOLDS = {
    "strong": 0.40,  # 강
    "medium": 0.20,  # 중
    "weak":   0.05,  # 약
}

def _severity_by_ratio(count: int, total_frames: int):
    if total_frames <= 0:
        return ""
    r = count / float(total_frames)
    if r >= _THRESHOLDS["strong"]:
        return "강"
    elif r >= _THRESHOLDS["medium"]:
        return "중"
    elif r >= _THRESHOLDS["weak"]:
        return "약"
    else:
        return ""

def parse_posture_summary_jetson(xml_path: str):
    """
    frame/analysis/result(type="...") 구조를 가정:
      - negative_emotion
      - bad_posture
      - leg_shake
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    neg_cnt = bad_cnt = leg_cnt = total_frames = 0

    for frame in root.findall("frame"):
        total_frames += 1
        analysis = frame.find("analysis")
        if analysis is None:
            continue
        for result in analysis.findall("result"):
            rtype = (result.get("type") or "").strip()
            if rtype == "negative_emotion":
                neg_cnt += 1
            elif rtype == "bad_posture":
                bad_cnt += 1
            elif rtype == "leg_shake":
                leg_cnt += 1

    neg_sev = _severity_by_ratio(neg_cnt, total_frames)
    bad_sev = _severity_by_ratio(bad_cnt, total_frames)
    leg_sev = _severity_by_ratio(leg_cnt, total_frames)

    labels = []
    if neg_sev:
        labels.append(f"표정(부정):{neg_sev}")
    if bad_sev:
        labels.append(f"자세:{bad_sev}")
    if leg_sev:
        labels.append(f"다리 떨림:{leg_sev}")
    if not labels:
        labels.append("정상")

    return {
        "frames": total_frames,
        "negative_emotion_count": neg_cnt,
        "bad_posture_count": bad_cnt,
        "leg_shake_count": leg_cnt,
        "negative_emotion_severity": neg_sev,
        "bad_posture_severity": bad_sev,
        "leg_shake_severity": leg_sev,
        "label": ", ".join(labels),
    }
