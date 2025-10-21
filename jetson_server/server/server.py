import os
from flask import Flask, request, jsonify, send_file, abort
from process_call import ProcessManager, SRV_TMP

app = Flask(__name__)
pm = ProcessManager()

# -------- Commands --------
@app.route("/command/start_record")
def start_record():
    src = request.args.get("src", "1")
    try:
        cmd = pm.start(src=src)  # 프로세스만 띄움 (b→t는 백그라운드 스레드)
        return jsonify(ok=True, running=pm.is_running(), cmd=cmd, out_dir=SRV_TMP, log=pm.last_log_path)
    except Exception as e:
        return jsonify(ok=False, error=str(e)), 400

@app.route("/command/stop_record")
def stop_record():
    try:
        pm.stop()
        return jsonify(ok=True, running=pm.is_running(), out_dir=SRV_TMP, log=pm.last_log_path)
    except Exception as e:
        return jsonify(ok=False, error=f"stop failed: {e}"), 200

# -------- Manual key (디버그용) --------
@app.route("/command/key/<key>")
def press_key(key):
    ok = pm.press_manual(key)
    return jsonify(ok=bool(ok), key=key)

# -------- Status / Download / Debug --------
@app.route("/api/status")
def status():
    def list_files(sub):
        p = os.path.join(SRV_TMP, sub)
        try:
            return sorted(
                [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))],
                key=lambda n: os.path.getmtime(os.path.join(p, n)), reverse=True
            )
        except Exception:
            return []
    return jsonify(
        running=pm.is_running(),
        srv_tmp={"mp4": list_files("mp4"), "wav": list_files("wav"), "xml": list_files("xml")}
    )

@app.route("/download/<kind>/<filename>")
def download(kind, filename):
    kind = kind.lower()
    if kind not in {"mp4","wav","xml"}: abort(404)
    path = os.path.join(SRV_TMP, kind, filename)
    if not os.path.isfile(path): abort(404)
    mime = {"mp4":"video/mp4", "wav":"audio/wav", "xml":"application/xml"}[kind]
    return send_file(path, as_attachment=True, mimetype=mime, download_name=filename)

@app.route("/debug/log")
def debug_log():
    tail = pm.read_last_log_tail(4000)
    if tail is None:
        return jsonify(ok=False, error="no run log yet")
    return jsonify(ok=True, path=pm.last_log_path, tail=tail)

@app.route("/debug/serverlog")
def debug_serverlog():
    tail = pm.read_server_log_tail(4000)
    if tail is None:
        return jsonify(ok=False, error="no server log yet")
    return jsonify(ok=True, tail=tail)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
