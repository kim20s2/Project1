import os
import time
from flask import Flask, request, send_from_directory
from scripts.process_call import ProcessManager

script_path = os.path.abspath(__file__)
main_dir = os.path.dirname(os.path.dirname(script_path))

FOLDER = os.path.join(main_dir, 'srv_tmp')
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

app = Flask(__name__)
app.config['FOLDER'] = FOLDER

# Single ProcessManager instance used by all HTTP requests
process_manager = ProcessManager()

# process_manager.rtsp_start()

# File download processing (http://<Raspberry_Pi_IP>:5000/download/subfolder/filename)
@app.route('/download/<path:filepath>')
def download_file(filepath):
    # 파일 경로에서 디렉토리와 파일 이름 분리
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    
    # 기본 폴더와 하위 경로 결합
    full_directory = os.path.join(app.config['FOLDER'], directory) if directory else app.config['FOLDER']
    
    return send_from_directory(full_directory, filename, as_attachment=True)

# http://<Raspberry_Pi_IP>:5000/command/command 형식으로 GET 요청을 받습니다.
@app.route('/command/<cmd>')
def execute_command(cmd):
    print(f"'{cmd}' command received.")

    global process_manager

    # 'start_record' command received
    if cmd == 'start_record':
        # 여기에 실제 녹음 시작 코드를 넣으면 됩니다.
        print(">> start recording!")
        process_manager.rtsp_start(target)      # rtsp_server 프로세스 시작
        time.sleep(2)  # rtsp_server가 완전히 시작될 때까지 잠시 대기
        process_manager.f_estimator_start()   # estimator 프로세스 시작
        process_manager.p_estimator_start()   # estimator 프로세스 시작
        process_manager.recorder_start()      # recorder 프로세스 시작
        return f"'{cmd}' command executed successfully!", 200

    # 'stop_record' command received
    elif cmd == 'stop_record':
        print(">> stop recording!")
        process_manager.recorder_finish()     # recorder 프로세스 종료
        process_manager.p_estimator_finish()   # estimator 프로세스 종료
        process_manager.f_estimator_finish()   # estimator 프로세스 종료
        process_manager.rtsp_server_finish()  # rtsp_server 프로세스 종료
        process_manager.xml_mix()             # xml 믹스 프로세스 시작
        return f"'{cmd}' command executed successfully!", 200

    else:
        return "Unknown command.", 400

if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Pose Estimation with Hailo')
    parser.add_argument('--target', type=str, required=True,
                       help='target video device or file')
    
    args = parser.parse_args()

    global target

    target = args.target

    # host='0.0.0.0' allows access from all IPs.
    app.run(host='10.10.14.80', port=5000)
