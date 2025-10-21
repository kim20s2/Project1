# -*- coding: utf-8 -*-
import os, sys, time, argparse, math, subprocess
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from collections import deque

# TensorRT
import pycuda.autoinit
from trt_utils import TRTModule

# ===== 고정 엔진 경로 =====
FER_ENGINE_FIXED  = "/home/jetson10/fer/scripts/face_3cls.engine"
POSE_ENGINE_FIXED = "/home/jetson10/fer/scripts/pose_movenet.engine"

EMO3_LABELS_FIXED = ["negative", "neutral", "positive"]  # [neg, neu, pos]

# -------- MoveNet COCO 인덱스 --------
NOSE=0; L_EYE=1; R_EYE=2; L_EAR=3; R_EAR=4
L_SH=5; R_SH=6; L_ELB=7; R_ELB=8; L_WRI=9; R_WRI=10
L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16

# --------- 다리 떨기 FFT 파라미터(완화 세팅) ---------
# 상단에 함께 추가
LEG_MIN_AMP_ON   = 0.050  # 켤 때
LEG_MIN_AMP_OFF  = 0.025  # 끌 때 (조금 느슨)
LEG_MIN_RATIO_ON = 0.10
LEG_MIN_RATIO_OFF= 0.05

LEG_WIN_SEC   = 2.5
LEG_BAND_LO   = 2.5
LEG_BAND_HI   = 9.0
# LEG_MIN_AMP   = 0.04
# LEG_MIN_RATIO = 0.20
LEG_VIS_THR   = 0.50

leg_state_on = False

# ---------------- utils ----------------
def _now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def softmax_temp(logits, T=1.0):
    z = logits / max(1e-6, float(T))
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def find_haar_xml():
    for p in [
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml"),
    ]:
        if os.path.isfile(p): return p
    return None

# ---------- 포즈→얼굴 박스 ----------
def _clip_square(cx, cy, s, W, H):
    x = int(cx - s/2); y = int(cy - s/2)
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    s = int(min(s, W-x, H-y))
    return (x, y, s, s)

def face_box_from_pose(k, W, H, thr=0.30):
    def vis(i): return k[i,2] >= thr
    def dist(a,b): return float(np.hypot(k[a,0]-k[b,0], k[a,1]-k[b,1]))
    cands = []
    if vis(L_EYE) and vis(R_EYE): cands.append(2.4 * dist(L_EYE, R_EYE))
    if vis(L_EAR) and vis(R_EAR): cands.append(2.0 * dist(L_EAR, R_EAR))
    if vis(L_SH)  and vis(R_SH):  cands.append(0.9 * dist(L_SH, R_SH))
    if not cands: return None
    s = float(np.median(cands)); s = max(32.0, min(s, 0.6*min(W, H)))
    if vis(L_EYE) and vis(R_EYE):
        cx = 0.5*(k[L_EYE,0]+k[R_EYE,0]); cy = 0.5*(k[L_EYE,1]+k[R_EYE,1])
    elif vis(L_EAR) and vis(R_EAR):
        cx = 0.5*(k[L_EAR,0]+k[R_EAR,0]); cy = 0.5*(k[L_EAR,1]+k[R_EAR,1])
    elif vis(NOSE):
        cx = k[NOSE,0]; cy = k[NOSE,1] - 0.1*s
    elif vis(L_SH) and vis(R_SH):
        cx = 0.5*(k[L_SH,0]+k[R_SH,0]); cy = 0.5*(k[L_SH,1]+k[R_SH,1]) - 0.9*s
    else:
        return None
    cy = cy - 0.10*s
    return _clip_square(cx, cy, s, W=W, H=H)

def _ema_box(prev, curr, alpha=0.5):
    if prev is None: return curr
    px,py,pw,ph = prev; cx,cy,cw,ch = curr
    return (int(alpha*cx+(1-alpha)*px),
            int(alpha*cy+(1-alpha)*py),
            int(alpha*cw+(1-alpha)*pw),
            int(alpha*ch+(1-alpha)*ph))

# ---------- FER 전처리/추론 ----------
def preprocess_face_roi(face_bgr, size=224, use_clahe=False, gamma=1.0):
    x = face_bgr.copy()
    if gamma != 1.0:
        g = max(0.1, float(gamma))
        lut = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)], dtype=np.float32)
        x = cv2.LUT(x, np.clip(lut,0,255).astype("uint8"))
    if use_clahe:
        ycrcb = cv2.cvtColor(x, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        x = cv2.cvtColor(cv2.merge([y,cr,cb]), cv2.COLOR_YCrCb2BGR)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = (x - 0.5)/0.5   # rgb_m11
    x = np.transpose(x, (2,0,1))[None, ...]
    return x

def infer_fer(face_bgr, fer_engine, temp, use_clahe, gamma, tta_flip, label_tokens):
    x = preprocess_face_roi(face_bgr, use_clahe=use_clahe, gamma=gamma)
    out = fer_engine.infer(x)
    logits = np.squeeze(out[0] if isinstance(out,(list,tuple)) else out).astype(np.float32)
    if tta_flip:
        x2 = preprocess_face_roi(cv2.flip(face_bgr,1), use_clahe=use_clahe, gamma=gamma)
        out2 = fer_engine.infer(x2)
        logits2 = np.squeeze(out2[0] if isinstance(out2,(list,tuple)) else out2).astype(np.float32)
        logits = 0.5*(logits + logits2)
    probs = softmax_temp(logits[None, ...], T=temp)[0]
    order = [label_tokens.index("neg"), label_tokens.index("neu"), label_tokens.index("pos")]
    return probs[order]  # [neg, neu, pos]

# ---------- MoveNet ----------
def preprocess_movenet(frame, size=192):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(img.astype(np.int32), 0)

def postprocess_movenet(out, w, h):
    arr = np.array(out).squeeze().reshape(17,3)
    xs = arr[:,1] * w; ys = arr[:,0] * h; sc = arr[:,2]
    return np.stack([xs,ys,sc], axis=1)

def open_camera(src, width, height, fps):
    if isinstance(src, str) and not src.isdigit():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"ERROR: cannot open video file: {src}"); sys.exit(1)
        return cap
    try: dev = int(src)
    except: dev = 3
    cap = cv2.VideoCapture(dev)
    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:    cap.set(cv2.CAP_PROP_FPS, fps)
   
    return cap

# -------- posture metrics & scoring --------
def angle_to_vertical(dx, dy):
    vnorm = math.hypot(dx, dy)
    if vnorm < 1e-6: return 0.0
    cos_th = max(-1.0, min(1.0, dy / vnorm))
    return math.degrees(math.acos(cos_th))

def posture_metrics(k, thr=0.3):
    need = [L_SH,R_SH,L_HIP,R_HIP]
    if any(k[i,2] < thr for i in need): return None
    def dist(a,b): return float(np.hypot(k[a,0]-k[b,0], k[a,1]-k[b,1]))
    sh_w = max(1.0, dist(L_SH, R_SH))
    ear_w = dist(L_EAR, R_EAR) if (k[L_EAR,2] >= thr and k[R_EAR,2] >= thr) else sh_w
    shoulder_slope = abs(k[L_SH,1]-k[R_SH,1]) / sh_w
    sh_mid = ((k[L_SH,0]+k[R_SH,0])/2.0, (k[L_SH,1]+k[R_SH,1])/2.0)
    hip_mid= ((k[L_HIP,0]+k[R_HIP,0])/2.0, (k[L_HIP,1]+k[R_HIP,1])/2.0)
    dx = sh_mid[0]-hip_mid[0]; dy = sh_mid[1]-hip_mid[1]
    torso_lean = angle_to_vertical(dx, dy)
    fwd_head=None
    if k[L_EAR,2]>=thr: fwd_head=abs(k[L_EAR,0]-k[L_SH,0])/sh_w
    if k[R_EAR,2]>=thr:
        v2=abs(k[R_EAR,0]-k[R_SH,0])/sh_w
        fwd_head = v2 if fwd_head is None else min(fwd_head, v2)
    if fwd_head is None:
        if k[NOSE,2]>=thr:
            nx=k[NOSE,0]; fwd_head=min(abs(nx-k[L_SH,0]), abs(nx-k[R_SH,0]))/sh_w
        else: fwd_head=0.4
    head_roll=0.0
    if k[L_EAR,2]>=thr and k[R_EAR,2]>=thr:
        dx_e=(k[R_EAR,0]-k[L_EAR,0]); dy_e=(k[R_EAR,1]-k[L_EAR,1])
        head_roll=abs(math.degrees(math.atan2(dy_e, dx_e)))
    ear_mid_x=(k[L_EAR,0]+k[R_EAR,0])/2.0 if (k[L_EAR,2]>=thr and k[R_EAR,2]>=thr) else (sh_mid[0])
    ear_mid_y=(k[L_EAR,1]+k[R_EAR,1])/2.0 if (k[L_EAR,2]>=thr and k[R_EAR,2]>=thr) else (sh_mid[1])
    head_yaw  = abs((k[NOSE,0]-ear_mid_x))/max(1.0,ear_w) if k[NOSE,2]>=thr else 0.0
    head_pitch= abs((k[NOSE,1]-ear_mid_y))/max(1.0,ear_w) if k[NOSE,2]>=thr else 0.0
    nose_sh_vert = abs((k[NOSE,1]-sh_mid[1])) / sh_w if k[NOSE,2]>=thr else 0.0
    torso_mid_x=(sh_mid[0]+hip_mid[0])*0.5
    parts=[]
    for p in (L_WRI,R_WRI,L_ELB,R_ELB):
        if k[p,2]>=thr: parts.append(abs(k[p,0]-torso_mid_x)/sh_w)
    arm_spread=float(np.mean(parts)) if parts else 0.0
    return {"ShoulderSlope": shoulder_slope, "TorsoLean": torso_lean, "FwdHead": fwd_head,
            "Roll": head_roll, "Yaw": head_yaw, "Pitch": head_pitch,
            "NoseShoulderVert": nose_sh_vert, "ArmSpread": arm_spread, "scale_w": sh_w}

def posture_score(m, baseline=None):
    def rel(k,v): return max(0.0, v - (baseline.get(k,0.0) if baseline else 0.0))
    ss=rel("ShoulderSlope",m["ShoulderSlope"]); tl=rel("TorsoLean",m["TorsoLean"])
    fh=rel("FwdHead",m["FwdHead"]); rl=rel("Roll",m["Roll"]); yw=rel("Yaw",m["Yaw"])
    pt=rel("Pitch",m["Pitch"]); nv=rel("NoseShoulderVert",m["NoseShoulderVert"]); aspr=rel("ArmSpread",m["ArmSpread"])
    score=100.0
    if ss>0.10: score -= (ss-0.10)*400.0
    if tl>8.0:  score -= (tl-8.0)*2.5
    if fh>0.35: score -= (fh-0.35)*240.0
    if rl>5.0:  score -= (rl-5.0)*3.2
    if yw>0.18: score -= (yw-0.18)*300.0
    if pt>0.22: score -= (pt-0.22)*280.0
    if nv>0.12: score -= (nv-0.12)*320.0
    if aspr>0.65: score -= (aspr-0.65)*180.0
    score=float(np.clip(score,0,100))
    label="Good" if score>=80 else ("Okay" if score>=60 else "Bad")
    debug_rel={"ShoulderSlope":ss,"TorsoLean":tl,"FwdHead":fh,"Roll":rl,"Yaw":yw,"Pitch":pt,
               "NoseShoulderVert":nv,"ArmSpread":aspr}
    return label, score, debug_rel

# -------- drawing --------
def put_text(img, text, org, scale=0.55, color=(255,255,255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_pose(img, kpts, thr=0.3):
    skel=[(0,1),(0,2),(1,3),(2,4),
          (0,5),(0,6),(5,7),(7,9),(6,8),(8,10),
          (5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for (x,y,s) in kpts:
        if s<thr: continue
        cv2.circle(img,(int(x),int(y)),3,(0,255,0),-1)
    for a,b in skel:
        if kpts[a,2]>=thr and kpts[b,2]>=thr:
            pa=(int(kpts[a,0]),int(kpts[a,1])); pb=(int(kpts[b,0]),int(kpts[b,1]))
            cv2.line(img,pa,pb,(0,255,0),2)

def draw_left_texts(img, label, score, fps=None, log_on=False, leg_shake=None):
    x, y = 10, 20
    col = (0,255,0) if label=="Good" else ((0,215,255) if label=="Okay" else (0,170,255))
    cv2.putText(img, f"Posture: {label} ({int(score)})", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    yy = y + 24
    if fps is not None:
        put_text(img, f"FPS: {fps:.1f}", (x, yy), 0.55, (0,0,0)); yy += 20
    put_text(img, f"LOG: {'ON' if log_on else 'OFF'}", (x, yy), 0.5, (0,255,0) if log_on else (180,180,180)); yy += 18
    if leg_shake is not None:
        put_text(img, f"Leg: {'Shaking' if leg_shake else 'Calm'}", (x, yy), 0.5, (0,0,255) if leg_shake else (200,200,200))

def draw_right_emotion_text_only(img, probs):
    H, W = img.shape[:2]
    x = W - 150
    y = 18
    lines = [ f"positive: {probs[2]*100:.1f}%",
              f"neutral : {probs[1]*100:.1f}%",
              f"negative: {probs[0]*100:.1f}%", ]
    for i,t in enumerate(lines):
        put_text(img, t, (x, y + i*18), 0.5, (0,0,0))

def draw_bottom_right_debug(img, metrics):
    if metrics is None: return
    H, W = img.shape[:2]; x = W - 250
    keys=("ShoulderSlope","TorsoLean","FwdHead","Roll","Yaw","Pitch","NoseShoulderVert","ArmSpread")
    start_y = H - 8 - (len(keys)*18)
    for i,k in enumerate(keys):
        v = metrics[k]
        txt = f"{k}: {v:.2f}" if k!="Roll" else f"{k}: {v:.1f} deg"
        put_text(img, txt, (x, start_y + i*18), 0.5, (0,255,255))

# -------- 다리 떨기 --------
def _norm_scale_len(k):
    if k[L_HIP,2] >= LEG_VIS_THR and k[L_ANK,2] >= LEG_VIS_THR: 
        lh = np.hypot(k[L_HIP,0]-k[L_ANK,0], k[L_HIP,1]-k[L_ANK,1])
    else: lh = np.inf
    if k[R_HIP,2] >= LEG_VIS_THR and k[R_ANK,2] >= LEG_VIS_THR:
        rh = np.hypot(k[R_HIP,0]-k[R_ANK,0], k[R_HIP,1]-k[R_ANK,1])
    else: rh = np.inf
    s = min(lh, rh)
    if not np.isfinite(s):
        if np.isfinite(lh): s = lh
        elif np.isfinite(rh): s = rh
        else: s = 100.0
    return max(1.0, min(s, 150.0))

def update_leg_buffers(k, deq_l, deq_r):
    if k[L_ANK,2] >= LEG_VIS_THR and k[L_HIP,2] >= LEG_VIS_THR:
        deq_l.append(k[L_ANK,1] / _norm_scale_len(k))
    else: deq_l.append(np.nan)
    if k[R_ANK,2] >= LEG_VIS_THR and k[R_HIP,2] >= LEG_VIS_THR:
        deq_r.append(k[R_ANK,1] / _norm_scale_len(k))
    else: deq_r.append(np.nan)

def _interp_series(x):
    x = np.asarray(list(x), dtype=np.float32)
    n = len(x); idx = np.arange(n); m = np.isfinite(x)
    if not np.any(m): return np.zeros_like(x)
    x[~m] = np.interp(idx[~m], idx[m], x[m])
    x = x - np.mean(x[m]); return x

def _ready(deq, need=24):
    """FFT 돌리기 전에 유효 샘플이 충분히 쌓였는지 체크"""
    x = np.asarray(deq, np.float32)
    return (len(x) >= need) and (np.isfinite(x).sum() >= int(need*0.7))

def _band_peak(y, fps_est):
    n = len(y)
    if n < 16 or fps_est is None or fps_est <= 0: return 0.0, 0.0, 0.0
    win = np.hanning(n); yf = np.fft.rfft(y * win)
    fq  = np.fft.rfftfreq(n, d=1.0/float(fps_est))
    pw  = (np.abs(yf)**2)
    band = (fq >= LEG_BAND_LO) & (fq <= LEG_BAND_HI)
    if not np.any(band): return 0.0, 0.0, 0.0
    band_power = float(np.sum(pw[band])); total_power = float(np.sum(pw))
    ratio = band_power / max(1e-9, total_power)
    pfreq = float(fq[band][np.argmax(pw[band])])
    amp   = float(0.5*(np.nanmax(y)-np.nanmin(y)))
    return pfreq, ratio, amp

def detect_leg_shake_simple(deq_l, deq_r, fps_est):
    global leg_state_on  # 밖에서 선언된 leg_state_on을 갱신하려면 필요
    
    #if not (_ready(deq_l, need=24) or _ready(deq_r, need=24)):
    #    return False, {"side":"?", "freq":0.0, "ratio":0.0, "amp":0.0}
    
    yl = _interp_series(deq_l)
    yr = _interp_series(deq_r)
    fl, rl, al = _band_peak(yl, fps_est)
    fr, rr, ar = _band_peak(yr, fps_est)
    left_on_now  = (LEG_BAND_LO <= fl <= LEG_BAND_HI) and (rl >= (LEG_MIN_RATIO_ON if not leg_state_on else LEG_MIN_RATIO_OFF)) and (al >= (LEG_MIN_AMP_ON if not leg_state_on else LEG_MIN_AMP_OFF))
    right_on_now = (LEG_BAND_LO <= fr <= LEG_BAND_HI) and (rr >= (LEG_MIN_RATIO_ON if not leg_state_on else LEG_MIN_RATIO_OFF)) and (ar >= (LEG_MIN_AMP_ON if not leg_state_on else LEG_MIN_AMP_OFF))
    on_now = left_on_now or right_on_now

    info = {"side": "L" if (rl*al) >= (rr*ar) else "R",
            "freq": fl if (rl*al) >= (rr*ar) else fr,
            "ratio": rl if (rl*al) >= (rr*ar) else rr,
            "amp":   al if (rl*al) >= (rr*ar) else ar}
    
    leg_state_on = on_now
    return on_now, info


# -------- XML pretty --------
def _indent_elem(elem, level=0):
    i="\n"+level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip(): elem.text=i+"  "
        for e in elem: _indent_elem(e, level+1)
        if not e.tail or not e.tail.strip(): e.tail=i
    if level and (not elem.tail or not elem.tail.strip()): elem.tail=i

def save_events_xml(events, path):
    root=ET.Element("session")
    for ev in events:
        e=ET.SubElement(root,"event")
        for k,v in ev.items(): e.set(k, str(v))
    tree=ET.ElementTree(root); _indent_elem(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="3")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--show_fps", default=True, action="store_true")
    ap.add_argument("--pose_thresh", type=float, default=0.3)

    # 출력 경로: 디폴트는 /home/jetson10/fer/{mp4,wav,xml}
    ap.add_argument("--out_dir", default="", help="지정 시 해당 경로 아래 mp4/wav/xml 하위 폴더에 저장")
    default_root = "/home/jetson10/fer"
    ap.add_argument("--xml_always", default=True, action="store_true", help="이벤트가 없어도 빈 XML 파일을 생성")

    # 속도/해상도/프레임 제어
    ap.add_argument("--max_side", type=int, default=720)
    ap.add_argument("--frame_skip", type=int, default=0)
    ap.add_argument("--process_every", type=int, default=1)
    ap.add_argument("--sync_playback", action="store_true")

    # 회전
    ap.add_argument("--rotate", type=int, default=0, choices=[0,90,180,270])
    ap.add_argument("--auto_rotate", action="store_true")

    # 로깅 조건
    ap.add_argument("--neg_thresh", type=float, default=0.30)  # 30% 이상만 기록
    ap.add_argument("--bad_frames", type=int, default=10)
    ap.add_argument("--neg_frames", type=int, default=10)
    ap.add_argument("--leg_frames", type=int, default=10)
    ap.add_argument("--show_debug", defualt=True, action="store_true")
    
    # FER 튜닝(중립 기본)
    ap.add_argument("--fer_gamma", type=float, default=1.1)
    ap.add_argument("--fer_clahe", default=True, action="store_true")
    ap.add_argument("--fer_tta_flip", default=True, action="store_true")
    ap.add_argument("--fer_temp", type=float, default=1.1)
    ap.add_argument("--fer_labels", default="neg,neu,pos")

    # 오디오 녹음
    ap.add_argument("--mic_enable", default=True, action="store_true", help="t로 녹화할 때 마이크도 같이 녹음")
    ap.add_argument("--mic_device", default="plughw:2,0")
    ap.add_argument("--mic_rate", type=int, default=16000)
    ap.add_argument("--mic_channels", type=int, default=1)
    ap.add_argument("--mic_format", default="S16_LE")

    # 녹화 저장 FPS/모드
    ap.add_argument("--rec_fps", type=float, default=0.0, help="0=최근 1초 실측, >0=고정 FPS")
    ap.add_argument("--rec_mode", choices=["auto","realtime","fixed"], default="auto",
                    help="auto=실측+페이싱, realtime=벽시계 페이싱, fixed=헤더만 고정")

    # 코덱
    ap.add_argument("--rec_codec", default="mp4v")

    args = ap.parse_args()

    # 출력 폴더 결정
    root = (args.out_dir if args.out_dir else default_root)
    mp4_dir = os.path.join(root, "mp4")
    wav_dir = os.path.join(root, "wav")
    xml_dir = os.path.join(root, "xml")
    for d in [mp4_dir, wav_dir, xml_dir]:
        os.makedirs(d, exist_ok=True)
    print(f"[INFO] output dirs: mp4={mp4_dir}  wav={wav_dir}  xml={xml_dir}")

    label_tokens = [s.strip().lower() for s in args.fer_labels.split(",")]
    if set(label_tokens) != {"neg","neu","pos"}:
        print("[WARN] --fer_labels must be a permutation of neg,neu,pos. Using default.")
        label_tokens = ["neg","neu","pos"]

    fer  = TRTModule(FER_ENGINE_FIXED)
    pose = TRTModule(POSE_ENGINE_FIXED)

    haar = find_haar_xml()
    if not haar: print("ERROR: haarcascade xml not found"); sys.exit(1)
    face_cascade = cv2.CascadeClassifier(haar)

    cap = open_camera(args.src, args.width, args.height, args.fps)
    if not cap.isOpened(): print("ERROR: camera/file open failed"); sys.exit(1)
    # 최적화(가능하면)
    try:
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 실패해도 조용히 통과
    except:
        pass
    cv2.setUseOptimized(True)
    try: cv2.setNumThreads(1)
    except: pass

    win = "FER(3cls)+Posture(MoveNet)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    prev_t=time.time(); fps_val=None
    last_probs=np.array([0.,0.,0.], np.float32)

    baseline=None; baseline_active=False
    events=[]; bad_cnt=0; neg_cnt=0; leg_cnt=0; frame_id=0

    is_file_src = (isinstance(args.src, str) and not args.src.isdigit())
    in_fps_meta = cap.get(cv2.CAP_PROP_FPS)
    buf_len = max(16, int(LEG_WIN_SEC * (in_fps_meta if in_fps_meta and in_fps_meta>1 else args.fps)))
    leg_l = deque(maxlen=buf_len); leg_r = deque(maxlen=buf_len)

    clahe_g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    # 녹화 상태
    recording = False
    vw_annot = None
    vw_raw = None
    audio_proc = None
    audio_path = None
    current_xml_path = None
    clip_t0 = None        # 녹화 시작 시각(벽시계)
    rec_frame0 = None   # 녹화 시작 시의 전역 frame_id (클립 frame=0 기준)
    # leg_state_on = False
    events = []
    next_write_t = time.time()

    # 실측 FPS 추산용
    ts_buf = deque(maxlen=120)

    rotate_angle = args.rotate
    play_base = time.time()
    frame_base = 0

    while True:
        if is_file_src and args.process_every > 1:
            for _ in range(max(0, args.process_every - 1)):
                ok_g = cap.grab()
                if not ok_g: break

        ok, frame_raw = cap.read()
        if not ok:
            if is_file_src: break
            else: continue

        ts_buf.append(time.time())

        h0, w0 = frame_raw.shape[:2]
        if args.auto_rotate and h0 > w0 and rotate_angle == 0:
            rotate_angle = 90
        if rotate_angle == 90:   frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_angle == 180:frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_180)
        elif rotate_angle == 270:frame_raw = cv2.rotate(frame_raw, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if args.frame_skip > 0 and (frame_id % (args.frame_skip + 1)) != 0:
            frame_id += 1
            if args.sync_playback and is_file_src:
                target_t = frame_base / max(1.0, in_fps_meta if in_fps_meta and in_fps_meta>1 else args.fps)
                delay = play_base + target_t - time.time()
                if delay > 0: time.sleep(min(0.05, delay))
                frame_base += 1
            continue

        if args.max_side and args.max_side > 0:
            h1, w1 = frame_raw.shape[:2]
            long_side = max(w1, h1)
            if long_side > args.max_side:
                scale = args.max_side / float(long_side)
                new_w, new_h = int(w1 * scale + 0.5), int(h1 * scale + 0.5)
                frame_raw = cv2.resize(frame_raw, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame = frame_raw.copy()
        canvas = frame_raw.copy()
        h,w = frame.shape[:2]

        # Pose
        pose_in  = preprocess_movenet(frame, 192)
        pose_out = pose.infer(pose_in)
        kpts = postprocess_movenet(pose_out[0] if isinstance(pose_out,(list,tuple)) else pose_out, w, h)
        draw_pose(canvas, kpts, thr=args.pose_thresh)

        m = posture_metrics(kpts, thr=args.pose_thresh)
        if m is not None: label, score, m_rel = posture_score(m, baseline)
        else:             label, score, m_rel = ("Unknown", 0.0, None)

        # 얼굴: 포즈 우선 → Haar 폴백
        last_head = locals().get("last_head_box", None)
        face_from_pose = face_box_from_pose(kpts, w, h, thr=args.pose_thresh)
        if face_from_pose is not None:
            face_from_pose = _ema_box(last_head, face_from_pose, alpha=0.5)
            last_head = face_from_pose
        globals()["last_head_box"] = last_head

        faces = []
        if face_from_pose is not None:
            faces = [face_from_pose]; det_name = "pose"
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_eq = clahe_g.apply(gray)
            small = cv2.resize(gray_eq, (320, 240), interpolation=cv2.INTER_AREA)
            faces_s = face_cascade.detectMultiScale(small, scaleFactor=1.08, minNeighbors=3, minSize=(20,20))
            sx, sy = w/320.0, h/240.0
            faces = [(int(x*sx), int(y*sy), int(ww*sx), int(hh*sy)) for (x,y,ww,hh) in faces_s]
            det_name = "haar" if faces else "none"

        # FER
        if faces:
            x,y,fw,fh = max(faces, key=lambda b: b[2]*b[3])
            pad=int(0.12*max(fw,fh))  # padding 늘려 안정화
            x0=max(0,x-pad); y0=max(0,y-pad); x1=min(w,x+fw+pad); y1=min(h,y+fh+pad)
            face = frame[y0:y1, x0:x1]
            if face.size>0:
                probs = infer_fer(
                    face_bgr=face, fer_engine=fer, temp=args.fer_temp,
                    use_clahe=args.fer_clahe, gamma=args.fer_gamma, tta_flip=args.fer_tta_flip,
                    label_tokens=label_tokens
                )
                # 약간의 템포럴 스무딩
                last_probs = 0.7*last_probs + 0.3*probs.astype(np.float32)
                lab_idx = int(np.argmax(last_probs)); lab_txt = EMO3_LABELS_FIXED[lab_idx]
                cv2.rectangle(canvas, (x0,y0), (x1,y1), (0,160,255), 2)
                cv2.putText(canvas, f"{lab_txt} {np.max(last_probs):.2f}",
                            (x0, max(0,y0-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,160,255), 1, cv2.LINE_AA)

        # FPS
        if args.show_fps:
            now=time.time(); fps_val=1.0/max(1e-6, now-prev_t); prev_t=now

        # 다리 떨기
        update_leg_buffers(kpts, leg_l, leg_r)
        fps_est = (cap.get(cv2.CAP_PROP_FPS) if is_file_src else (fps_val if (args.show_fps and fps_val) else args.fps))
        leg_on, leg_info = detect_leg_shake_simple(leg_l, leg_r, fps_est)
        if leg_on and label != "Unknown":
            score = max(0.0, score - 10.0)
            label = "Good" if score>=80 else ("Okay" if score>=60 else "Bad")

        # 오버레이
        draw_left_texts(canvas, label, score, fps=fps_val if args.show_fps else None,
                        log_on=baseline_active, leg_shake=leg_on)
        draw_right_emotion_text_only(canvas, last_probs)
        cv2.putText(canvas, f"FACE: {det_name}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        if args.show_debug: 
            draw_bottom_right_debug(canvas, m_rel)
            put_text(canvas,
                f"LEG dbg: {leg_info['side']} f={leg_info['freq']:.2f}Hz "
                f"r={leg_info['ratio']:.2f}(thr {LEG_MIN_RATIO_ON:.2f}/{LEG_MIN_RATIO_OFF:.2f}) "
                f"amp={leg_info['amp']:.3f}(thr {LEG_MIN_AMP_ON:.3f}/{LEG_MIN_AMP_OFF:.3f})",
                (10,170), 0.45, (0,0,0))


        # 녹화 쓰기 (realtime 페이싱)
        if recording:
            if args.rec_mode in ("realtime","auto"):
                if args.rec_mode in ("realtime","auto"):
                    target_dt = 1.0/max(1e-3, rec_fps_active)

                now = time.time()
                if now < next_write_t:
                    time.sleep(min(0.02, next_write_t - now))
                next_write_t = max(next_write_t + target_dt, time.time())

            if vw_annot is not None:
                vw_annot.write(canvas)
                cv2.putText(canvas, "REC", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (50, 145), 5, (0,0,255), -1)
            if vw_raw is not None:
                vw_raw.write(frame)

        # ===== 이벤트 감지/메모리에만 append (녹화 중 + baseline ON일 때만) =====
        if baseline_active and recording and current_xml_path:
            if label=="Bad": bad_cnt += 1
            else: bad_cnt = 0
            if bad_cnt >= args.bad_frames:
                ts = (time.time() - clip_t0) if clip_t0 else 0.0
                events.append({"type":"bad_posture","timestamp_sec":f"{ts:.2f}",
                               "score":f"{score:.1f}","frame": str( (frame_id - rec_frame0) if rec_frame0 is not None else frame_id )})
                bad_cnt=0

            if last_probs[0] >= args.neg_thresh: neg_cnt += 1
            else: neg_cnt = 0
            if neg_cnt >= args.neg_frames:
                ts = (time.time() - clip_t0) if clip_t0 else 0.0
                events.append({"type":"negative_emotion","timestamp_sec":f"{ts:.2f}",
                               "prob":f"{last_probs[0]:.2f}","frame": str( (frame_id - rec_frame0) if rec_frame0 is not None else frame_id )})
                neg_cnt=0

            if leg_on: leg_cnt += 1
            else: leg_cnt = 0
            if leg_cnt >= args.leg_frames:
                ts = (time.time() - clip_t0) if clip_t0 else 0.0
                events.append({"type":"leg_shake","timestamp_sec":f"{ts:.2f}",
                               "side":leg_info["side"],
                               "freq_hz":f"{leg_info['freq']:.2f}",
                               "ratio":f"{leg_info['ratio']:.2f}",
                               "amp":f"{leg_info['amp']:.3f}",
                               "frame": str( (frame_id - rec_frame0) if rec_frame0 is not None else frame_id )})
                leg_cnt=0
        else:
            bad_cnt=neg_cnt=leg_cnt=0

        # 힌트/키
        cv2.putText(canvas, "[b] baseline  [r] reset  [t] record  [o] rotate 90  [q/Esc] quit",
                    (10, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        if args.sync_playback and is_file_src:
            target_t = frame_base / max(1.0, in_fps_meta if in_fps_meta and in_fps_meta>1 else args.fps)
            delay = (play_base + target_t) - time.time()
            if delay > 0: time.sleep(min(0.1, delay))
            frame_base += 1

        cv2.imshow(win, canvas)
        key=cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # q 또는 Esc로 끄기
            break

        # ---- b: baseline 캡처(포즈 있어야 동작) ----
        if key==ord('b') and (m is not None):
            baseline={k:m[k] for k in ("ShoulderSlope","TorsoLean","FwdHead","Roll","Yaw","Pitch","NoseShoulderVert","ArmSpread")}
            baseline_active=True; bad_cnt=neg_cnt=leg_cnt=0
            print("[INFO] Baseline set. Event logging ENABLED (only while recording).")

        # ---- r: baseline 해제 ----
        if key==ord('r'):
            baseline=None; baseline_active=False; bad_cnt=neg_cnt=leg_cnt=0
            print("[INFO] Baseline cleared. Event logging DISABLED.")

        # ---- t: 녹화 토글 ----
        if key == ord('t'):
            if not recording:
                # 1) 저장 FPS 계산(기존 로직 유지)
                ts_str = _now_str()

                # 2) 파일명 분기: --out_dir 있으면 고정 이름, 없으면 타임스탬프
                if args.out_dir:
                    path_annot = os.path.join(mp4_dir, "video_ai.mp4")
                    path_raw   = os.path.join(mp4_dir, "video.mp4")
                    audio_path = os.path.abspath(os.path.join(wav_dir, "audio.wav"))
                    current_xml_path = os.path.join(xml_dir, "log.xml")
                else:
                    path_annot = os.path.join(mp4_dir, f"rec_{ts_str}_annot.mp4")
                    path_raw   = os.path.join(mp4_dir, f"rec_{ts_str}_raw.mp4")
                    audio_path = os.path.abspath(os.path.join(wav_dir, f"rec_{ts_str}.wav"))
                    current_xml_path = os.path.join(xml_dir, f"xml_{ts_str}.xml")

                # ---- [추가] 녹화 저장 FPS 결정 ----
                # fixed 모드면 고정값, auto/realtime이면 최근 1초 측정값(없으면 기본값) 사용
                if args.rec_mode == "fixed" and args.rec_fps > 0:
                    fps_rec = float(args.rec_fps)
                else:
                    # auto/realtime
                    fps_rec = float(args.fps)  # fallback
                    if len(ts_buf) >= 5:
                        t_now = ts_buf[-1]
                        recent = [t for t in ts_buf if t_now - t <= 1.0]
                        if len(recent) >= 3:
                            dt = max(1e-3, recent[-1] - recent[0])
                            fps_rec = float(np.clip((len(recent)-1)/dt, 5.0, 60.0))

                # 페이싱용 활성 FPS 저장
                rec_fps_active = fps_rec
                next_write_t = time.time()

                # 3) VideoWriter 오픈
                H, W = canvas.shape[:2]
                Hr, Wr = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*args.rec_codec)
                # fps_rec/rec_fps_active/next_write_t 계산은 기존 그대로 유지
                vw_annot = cv2.VideoWriter(path_annot, fourcc, fps_rec, (W, H))
                vw_raw   = cv2.VideoWriter(path_raw,   fourcc, fps_rec, (Wr, Hr))
                if not vw_annot.isOpened(): vw_annot=None; print(f"[WARN] open failed: {path_annot}")
                if not vw_raw.isOpened():   vw_raw=None;   print(f"[WARN] open failed: {path_raw}")

                # 4) 오디오 시작 (경로는 위에서 만든 audio_path 사용)
                audio_proc=None
                if args.mic_enable:
                    mic_dev = args.mic_device
                    if mic_dev.startswith("hw:"): mic_dev = "plughw:" + mic_dev.split("hw:",1)[1]
                    cmd = ["arecord", "-D", mic_dev, "-t", "wav",
                        "-f", args.mic_format, "-r", str(args.mic_rate), "-c", str(args.mic_channels),
                        audio_path]
                    try:
                        audio_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        time.sleep(0.2)
                        if audio_proc.poll() is not None: raise RuntimeError("arecord exited immediately")
                        print(f"[INFO] Audio REC ON  → {audio_path}  ({mic_dev}, {args.mic_rate}Hz, {args.mic_channels}ch, {args.mic_format})")
                    except Exception as e:
                        print(f"[WARN] audio start failed: {e}"); audio_proc=None; audio_path=None

                # 5) XML는 “클립 끝에서 1회 저장” 전략 → events 리셋만
                print(f"[INFO] XML logging → {current_xml_path}")
                events = []
                clip_t0 = time.time()
                rec_frame0 = frame_id

                recording = (vw_annot is not None) or (vw_raw is not None)
                if recording:
                    print(f"[INFO] Recording ON  → annot:{path_annot}  raw:{path_raw}  ({W}x{H} @{fps_rec:.1f}fps)")
                else:
                    print("[WARN] Recording failed to start.")

            else:
                # 녹화 종료
                if vw_annot is not None: vw_annot.release(); vw_annot=None
                if vw_raw   is not None: vw_raw.release();   vw_raw=None
                print("[INFO] Recording OFF")

                if audio_proc is not None:
                    try:
                        audio_proc.terminate()
                        try: audio_proc.wait(timeout=2.0)
                        except: audio_proc.kill()
                        if audio_path and os.path.isfile(audio_path):
                            print(f"[INFO] Audio REC OFF → {audio_path}  (size={os.path.getsize(audio_path)} bytes)")
                        else:
                            print(f"[WARN] Audio file not found after stop: {audio_path}")
                    except Exception as e:
                        print(f"[WARN] audio stop failed: {e}")
                    finally:
                        audio_proc=None; audio_path=None

                # XML 최종 저장(클립당 1회)
                if current_xml_path and (args.xml_always or len(events)>0):
                    save_events_xml(events, current_xml_path)
                    print(f"[INFO] XML saved: {os.path.abspath(current_xml_path)}  (events={len(events)})")

                current_xml_path=None
                events=[]
                clip_t0 = None
                recording = False

        # ---- o: 90도 회전 ----
        if key == ord('o'):
            rotate_angle = (rotate_angle + 90) % 360
            print(f"[INFO] rotate = {rotate_angle} deg")

        frame_id += 1

    # 종료 정리
    if vw_annot is not None: vw_annot.release()
    if vw_raw   is not None: vw_raw.release()
    if audio_proc is not None:
        audio_proc.terminate()
        try: audio_proc.wait(timeout=2.0)
        except: audio_proc.kill()
    cap.release(); cv2.destroyAllWindows()

    # 혹시 미저장 이벤트 있으면 저장
    if current_xml_path and (args.xml_always or len(events)>0):
        save_events_xml(events, current_xml_path)
        print(f"[INFO] XML saved at exit: {os.path.abspath(current_xml_path)}  (events={len(events)})")
    else:
        print("[INFO] Exit without pending XML.")

if __name__ == "__main__":
    main()
