import os
import requests

RASPBERRY_PI_IP = '10.10.14.80'
SERVER_URL = f"http://{RASPBERRY_PI_IP}:5000"

# 1. 파일 업로드 하기
def upload_file(filepath):
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f)}
            response = requests.post(f"{SERVER_URL}/upload", files=files)
            print("업로드 응답:", response.text)
    except Exception as e:
        print(f"업로드 실패: {e}")

# 2. 파일 다운로드 하기
def download_file(filename, save_path='.'):
    try:
        response = requests.get(f"{SERVER_URL}/download/{filename}", stream=True)
        if response.status_code == 200:
            with open(os.path.join(save_path, filename), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"'{filename}' 다운로드 성공!")
        else:
            print(f"다운로드 실패: {response.text}")
    except Exception as e:
        print(f"다운로드 실패: {e}")

def send_command(cmd):
    """라즈베리 파이로 간단한 명령어를 보냅니다."""
    try:
        response = requests.get(f"{SERVER_URL}/command/{cmd}")
        print(f"명령 전송 응답: {response.text}")
    except Exception as e:
        print(f"명령 전송 실패: {e}")


if __name__ == '__main__':
    # --- 파일 전송 테스트 (이전과 동일) ---
    # file_to_upload = 'my_document.txt'
    # ... (생략) ...

    # --- 명령어 전송 테스트 ---
    print("\n--- '녹음 시작' 명령어를 라즈베리 파이로 보냅니다 ---")
    send_command('start_record')
    
    # 예시: 5초 대기
    # import time
    # time.sleep(5) 
    
    print("\n--- '녹음 중지' 명령어를 라즈베리 파이로 보냅니다 ---")
    send_command('stop_record')