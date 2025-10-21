import os, subprocess, time, glob, signal, json, re, threading
from typing import Optional, List

# ===== 경로/실행 설정 =====
WORK_DIR   = "/home/jetson10/fer"
SERVER_DIR = "/home/jetson10/fer/server"
SRV_TMP    = os.path.join(SERVER_DIR, "srv_tmp")
LOG_DIR    = os.path.join(SERVER_DIR, "logs")
SERVER_LOG = os.path.join(LOG_DIR, "server_debug.log")

SYSTEM_PY  = "python3"
SCRIPT     = "run_3cls_audio2.py"
BASE_CMD   = [SYSTEM_PY, SCRIPT, "--show_fps", "--src", "1"]  # --out_dir은 start()에서 강제

# ===== 타이밍 파라미터(필요시 조정) =====
WINDOW_WAIT        = 25.0   # 창 등장/안정 대기(스레드에서 사용)
WINDOW_STABLE_SEC  = 1.0    # geometry>0 유지 시간
WAIT_AFTER_WINDOW  = 2.0    # 창 보인 뒤 여유
START_B_DELAY      = 1.0    # 여유 후 b까지
START_T_DELAY      = 2.5    # b 이후 t까지(베이스라인 시간 확보)
STOP_WAIT_SEC      = 8.0    # (파일 안정 대기 용도 — 현재 stop은 q 1회만)
EXIT_WAIT_SEC      = 12.0   # q 후 종료 대기
KEY_DEBOUNCE       = 1.0    # 같은 키 연타 방지 간격

def _which(name: str) -> Optional[str]:
    from shutil import which as _w
    return _w(name)

XDOTOOL = _which("xdotool")
XSET    = _which("xset")

def _log(msg: str, **kw):
    os.makedirs(LOG_DIR, exist_ok=True)
    line = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "msg": msg}
    if kw: line.update(kw)
    with open(SERVER_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

class ProcessManager:
    """
    start():
      - run_3cls_audio2.py 실행(--out_dir srv_tmp 강제) 후 '즉시 반환'
      - 별도 스레드(_start_keys_worker)가 'PID 일치 + geometry 안정' 창을 찾은 뒤 b→t 1회 전송
        (활성창/터미널로 절대 보내지 않음). t 성공 시 _rec_on=True.
    stop():
      - _rec_on일 때만 t '한 번' 전송 → 짧게 유예 → q '한 번' 전송
      - 종료 대기 후 남아있으면 SIGTERM 1회
    """
    def __init__(self):
        self.proc: Optional[subprocess.Popen] = None
        self.last_cmd: Optional[List[str]] = None
        self.last_log_path: Optional[str] = None
        self._log_handle = None

        self._last_key_ts = 0.0
        self._last_start_ts = 0.0
        self._last_stop_ts  = 0.0

        self._active_before: Optional[str] = None  # 시작 직전 활성창(터미널) ID
        self._starter_thread: Optional[threading.Thread] = None

        self._rec_on = False  # ★ 현재 녹화 ON 상태

        for d in ("mp4","wav","xml"):
            os.makedirs(os.path.join(SRV_TMP, d), exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ---------- helpers ----------
    def _env(self):
        env = os.environ.copy()
        env.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))
        env.setdefault("XAUTHORITY", os.environ.get("XAUTHORITY", "/home/jetson10/.Xauthority"))
        return env

    def _set_arg(self, args: List[str], flag: str, val: Optional[str]):
        try:
            i = args.index(flag)
            if val is not None:
                if i+1 < len(args) and not args[i+1].startswith("--"):
                    args[i+1] = val
                else:
                    args.insert(i+1, val)
        except ValueError:
            args.append(flag)
            if val is not None: args.append(val)

    def _build_cmd(self, src: str) -> List[str]:
        cmd = list(BASE_CMD)
        if src: self._set_arg(cmd, "--src", str(src))
        self._set_arg(cmd, "--out_dir", SRV_TMP)  # srv_tmp로 강제
        return cmd

    def _debounce(self) -> bool:
        now = time.time()
        if now - self._last_key_ts < KEY_DEBOUNCE:
            return False
        self._last_key_ts = now
        return True

    def _auto_repeat(self, enable: bool):
        if not XSET: return
        try:
            subprocess.call(["xset", "r", "on" if enable else "off"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            _log("xset_failed", err=str(e))

    # ----- xdotool helpers -----
    def _get_active_window(self) -> Optional[str]:
        if not XDOTOOL: return None
        try:
            wid = subprocess.check_output(["xdotool", "getactivewindow"],
                                          stderr=subprocess.DEVNULL).decode().strip()
            return wid or None
        except subprocess.CalledProcessError:
            return None

    def _list_visible_windows(self) -> List[str]:
        if not XDOTOOL: return []
        try:
            out = subprocess.check_output(["xdotool", "search", "--onlyvisible", "--all", ""],
                                          stderr=subprocess.DEVNULL).decode().strip().splitlines()
            return [w for w in out if w]
        except subprocess.CalledProcessError:
            return []

    def _window_pid(self, win_id: str) -> Optional[int]:
        if not XDOTOOL: return None
        try:
            pid = subprocess.check_output(["xdotool", "getwindowpid", win_id],
                                          stderr=subprocess.DEVNULL).decode().strip()
            return int(pid) if pid else None
        except Exception:
            return None

    def _geometry_positive(self, win: str) -> bool:
        try:
            geo = subprocess.check_output(["xdotool", "getwindowgeometry", "--shell", win],
                                          stderr=subprocess.DEVNULL).decode()
            W = H = 0
            for line in geo.splitlines():
                if line.startswith("WIDTH="):  W = int(line.split("=",1)[1])
                if line.startswith("HEIGHT="): H = int(line.split("=",1)[1])
            return W > 0 and H > 0
        except subprocess.CalledProcessError:
            return False

    def _is_window_stable(self, win: str, stable_sec: float) -> bool:
        if not self._geometry_positive(win): return False
        start = time.time()
        while time.time() - start < stable_sec:
            if not self._geometry_positive(win): return False
            time.sleep(0.15)
        return True

    def _find_our_window(self, timeout: float) -> Optional[str]:
        """
        - 시작 직전 활성창(self._active_before)은 제외
        - getwindowpid == self.proc.pid 인 '보이는' 창만 채택
        - geometry가 stable_sec 동안 >0 유지
        """
        if not XDOTOOL or not self.proc: return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            wins = self._list_visible_windows()
            for w in wins:
                if self._active_before and w == self._active_before:
                    continue
                pid = self._window_pid(w)
                if pid == self.proc.pid and self._is_window_stable(w, WINDOW_STABLE_SEC):
                    return w
            time.sleep(0.2)
        return None

    def _press_key_strong(self, win_id: str, key: str, hold_ms: int = 80) -> bool:
        """
        windowactivate → windowfocus → keydown→(hold)→keyup
        ※ 실패해도 다른 창/활성창으로 절대 보내지 않음
        """
        if not XDOTOOL or not win_id:
            return False
        if not self._debounce():
            return False
        try:
            subprocess.check_call(["xdotool", "windowactivate", "--sync", win_id],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.check_call(["xdotool", "windowfocus", "--sync", win_id],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(0.12)
            subprocess.check_call(["xdotool", "keydown", "--window", win_id, key],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(max(0.03, hold_ms/1000.0))
            subprocess.check_call(["xdotool", "keyup", "--window", win_id, key],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            _log("key_sent_downup", key=key, win=win_id)
            return True
        except subprocess.CalledProcessError as e:
            _log("key_downup_failed", key=key, err=str(e))
            return False

    # ---------- 로그 패턴 대기(옵션) ----------
    def _wait_log(self, pattern: str, timeout: float) -> bool:
        """
        run_* 로그에서 pattern(정규식)이 '새로' 등장할 때까지 최대 timeout초 대기.
        """
        if not self.last_log_path or not os.path.isfile(self.last_log_path):
            return False
        prog = re.compile(pattern)
        end = time.time() + timeout
        last_size = 0
        try:
            last_size = os.path.getsize(self.last_log_path)
        except Exception:
            pass
        while time.time() < end:
            try:
                with open(self.last_log_path, "r", encoding="utf-8", errors="ignore") as f:
                    f.seek(last_size)
                    chunk = f.read()
                    if chunk and prog.search(chunk):
                        return True
                    last_size = f.tell()
            except Exception:
                pass
            time.sleep(0.2)
        return False

    # ---------- 자동 b→t 스레드 ----------
    def _start_keys_worker(self):
        """
        프로세스 생성 후 비동기 스레드에서 창을
        'PID 일치 + geometry 안정' 조건으로 찾은 뒤 b→t 1회 전송.
        t 전송 직전 한 번 더 창을 재탐색해(win_id 변경 대비) 그 창에 t를 보냄.
        """
        try:
            win = self._find_our_window(timeout=WINDOW_WAIT)
            _log("worker_window_b", win=win)
            if not (win and XDOTOOL and self.is_running()):
                _log("worker_window_not_found_or_proc_ended",
                     running=self.is_running(), xdotool=bool(XDOTOOL))
                return

            # b 전송
            self._auto_repeat(False)
            time.sleep(WAIT_AFTER_WINDOW)
            time.sleep(START_B_DELAY)
            ok_b = self._press_key_strong(win, "b")
            _log("worker_b_sent", ok=ok_b, win=win)

            # t 전송 직전 '다시' 창 찾기 (b 후 리사이즈/재생성 대응)
            time.sleep(START_T_DELAY)
            win2 = self._find_our_window(timeout=2.0)  # 짧게 재탐색
            _log("worker_window_t", win=win2)

            ok_t = False
            if win2 and self.is_running():
                ok_t = self._press_key_strong(win2, "t")
            else:
                _log("worker_t_skipped_no_window_or_proc_end",
                     running=self.is_running())

            # ★ t 성공으로 판단된 경우에만 녹화 ON 상태로
            if ok_t:
                self._rec_on = True

            self._auto_repeat(True)
            _log("worker_keys_sent", ok_b=ok_b, ok_t=ok_t, win_b=win, win_t=win2)

        except Exception as e:
            _log("worker_exception", err=str(e))

    # ---------- 파일 안정 대기 ----------
    def _wait_outputs_stable(self, timeout: float, step: float = 0.5) -> bool:
        end = time.time() + timeout
        pats = [
            os.path.join(SRV_TMP, "mp4", "*.mp4"),
            os.path.join(SRV_TMP, "wav", "*.wav"),
            os.path.join(SRV_TMP, "xml", "*.xml"),
        ]
        while time.time() < end:
            newest, newest_mtime = None, -1
            for p in pats:
                for f in glob.glob(p):
                    m = os.path.getmtime(f)
                    if m > newest_mtime:
                        newest, newest_mtime = f, m
            if newest:
                s1 = os.path.getsize(newest)
                time.sleep(step)
                s2 = os.path.getsize(newest) if os.path.exists(newest) else s1
                if s1 == s2 and s1 > 0:
                    _log("file_stable", path=newest, size=s2)
                    return True
            else:
                time.sleep(step)
        _log("file_wait_timeout", timeout=timeout)
        return False

    # ---------- lifecycle ----------
    def start(self, src: str = "1"):
        # 중복 시작 억제
        now = time.time()
        if now - self._last_start_ts < 1.0:
            return self.last_cmd
        self._last_start_ts = now

        if self.is_running():
            return self.last_cmd

        # 현재 활성창(터미널) 기록 → 타겟 제외용
        self._active_before = self._get_active_window()
        _log("active_before", win=self._active_before)

        # 로그 파일
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.last_log_path = os.path.join(LOG_DIR, f"run_{ts}.log")
        self._log_handle = open(self.last_log_path, "ab", buffering=0)

        # 프로세스 실행
        cmd = self._build_cmd(src)
        env = self._env()
        _log("popen_start", cmd=cmd, cwd=WORK_DIR,
             DISPLAY=env.get("DISPLAY"), XAUTHORITY=env.get("XAUTHORITY"))
        self.proc = subprocess.Popen(
            cmd, cwd=WORK_DIR, env=env,
            stdout=self._log_handle, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        self.last_cmd = cmd

        # 자동 b→t 스레드 시작 (비동기)
        self._starter_thread = threading.Thread(target=self._start_keys_worker, daemon=True)
        self._starter_thread.start()

        return cmd

    def stop(self):
        """
        t(1회, 녹화 중일 때만) → 짧게 대기(1s) → q(1회) → 종료 대기 → 남으면 SIGTERM
        ※ 재시도 없음: t/q는 각 1회만 보냄
        """
        now = time.time()
        if now - self._last_stop_ts < 1.0:
            _log("stop_ignored_too_soon")
            return
        self._last_stop_ts = now

        if not self.is_running():
            _log("stop_not_running")
            return

        # 우리 창 찾기(짧게)
        win = self._find_our_window(timeout=1.0)
        _log("stop_begin", win=win)

        # 1) 녹화 중일 때만 t 1회 (절대 재시도 없음)
        if self._rec_on and win and XDOTOOL:
            sent_t = self._press_key_strong(win, "t")
            _log("stop_t_sent_once", ok=sent_t)
            if sent_t:
                # 내부 상태를 'OFF'로 내려 상태 재진입 방지
                self._rec_on = False
        else:
            _log("stop_skip_t", rec_on=self._rec_on, has_win=bool(win))

        # 2) q 전에 짧게 유예 (파일 flush/프레임 마감 여유)
        time.sleep(1.0)  # 필요 시 0.8~1.5로 미세조정

        # 3) q 1회 (창이 남아있을 때만)
        win_q = self._find_our_window(timeout=0.5)
        if win_q and XDOTOOL:
            try:
                self._auto_repeat(False)
                time.sleep(0.2)
                sent_q = self._press_key_strong(win_q, "q")
                _log("stop_q_sent_once", ok=sent_q)
            finally:
                self._auto_repeat(True)
        else:
            _log("stop_skip_q_no_window")

        # 4) 종료 대기
        end = time.time() + EXIT_WAIT_SEC
        while time.time() < end:
            if not self.is_running():
                break
            time.sleep(0.2)

        # 5) 남아있으면 SIGTERM 1회
        if self.is_running():
            _log("sigterm_send", pid=self.proc.pid)
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception as e:
                _log("sigterm_failed", err=str(e))

        # 6) 핸들 정리
        try:
            if self._log_handle:
                self._log_handle.flush(); self._log_handle.close()
        except Exception:
            pass
        self.proc = None
        self._log_handle = None

    # ---------- debug ----------
    def read_last_log_tail(self, max_bytes: int = 4000) -> Optional[str]:
        p = self.last_log_path
        if not p or not os.path.isfile(p): return None
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            return f.read().decode("utf-8", errors="ignore")

    def read_server_log_tail(self, max_bytes: int = 4000) -> Optional[str]:
        p = SERVER_LOG
        if not os.path.isfile(p): return None
        with open(p, "rb") as f:
            f.seek(0, os.SEEK_END); size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            return f.read().decode("utf-8", errors="ignore")

    def press_manual(self, key: str) -> bool:
        win = self._find_our_window(timeout=2.0)
        return self._press_key_strong(win, key) if win else False
