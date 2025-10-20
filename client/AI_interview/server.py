import os
from flask import Flask, request, send_from_directory
import requests  # 🔥 추가

FOLDER = 'tmp'
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

app = Flask(__name__)
app.config['FOLDER'] = FOLDER

# Streamlit 서버 주소 (로컬 PC IP로 수정하세요)
STREAMLIT_URL = "http://10.10.14.101:5000/upload_audio"  # ❗️수정 필요

@app.route('/command/<cmd>')
def execute_command(cmd):
    print(f"'{cmd}' command received.")

    if cmd == 'start_record':
        print(">> start recording!")
        # 실제 녹음 시작 코드 (예: arecord 등)
        os.system("arecord -D plughw:1,0 -f cd -t wav -d 5 tmp/record.wav &")
        return f"'{cmd}' command executed successfully!", 200

    elif cmd == 'stop_record':
        print(">> stop recording!")
        # 실제 녹음 종료
        os.system("pkill arecord")

        # Streamlit 서버로 파일 업로드
        filepath = os.path.join(FOLDER, "record.wav")
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                files = {"file": f}
                try:
                    r = requests.post(STREAMLIT_URL, files=files)
                    print("Upload result:", r.text)
                except Exception as e:
                    print("Upload failed:", e)
        return f"'{cmd}' command executed successfully!", 200

    else:
        return "Unknown command.", 400


if __name__ == '__main__':
    app.run(host='10.10.14.101', port=5000)
