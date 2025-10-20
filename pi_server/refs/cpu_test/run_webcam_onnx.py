import cv2
import numpy as np
import onnxruntime as ort
import time

# ============================================
# 설정
# ============================================
ONNX_PATH = "best_model_nn_output.onnx"
IMG_SIZE = 128
NUM_CLASSES = 5
SELECTED_CLASSES = ['angry', 'disgust', 'happy', 'neutral', 'surprise']

CLASS_COLORS = {
    'angry': (0, 0, 255),      # Red
    'disgust': (0, 128, 0),    # Dark Green
    'happy': (0, 255, 255),    # Yellow
    'neutral': (255, 255, 255), # White
    'surprise': (255, 128, 0)   # Orange
}

# FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# ============================================
# 전처리 함수
# ============================================
def preprocess_frame(frame):
    """
    OpenCV 프레임을 모델 입력 형식으로 변환
    학습 코드의 전처리 로직과 동일하게 구현
    """
    # Grayscale 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 리사이즈 (bicubic interpolation)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
    # Grayscale -> RGB (채널 복제)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # 정규화: [0, 255] -> [0, 1] -> [-1, 1]
    normalized = rgb.astype(np.float32) / 255.0
    normalized = normalized * 2.0 - 1.0
    normalized = np.clip(normalized, -1.0, 1.0)
    
    # 배치 차원 추가: (128, 128, 3) -> (1, 128, 128, 3)
    input_data = np.expand_dims(normalized, axis=0)
    
    return input_data

# ============================================
# 후처리 함수
# ============================================
def softmax(logits):
    """Softmax 함수 (logits -> 확률)"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def postprocess_output(output):
    """
    모델 출력(logits)을 확률로 변환하고 예측 결과 반환
    
    Returns:
        predicted_class: int (클래스 인덱스)
        confidence: float (확률)
        probabilities: np.array (모든 클래스 확률)
    """
    # Logits를 확률로 변환
    probabilities = softmax(output[0])
    
    # 가장 높은 확률의 클래스
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return predicted_class, confidence, probabilities

# ============================================
# 시각화 함수
# ============================================
def draw_results(frame, predicted_class, confidence, probabilities, fps):
    """
    프레임에 예측 결과와 확률 막대 그래프 표시
    """
    h, w = frame.shape[:2]
    
    # 반투명 오버레이 생성
    overlay = frame.copy()
    
    # 상단 정보 박스
    emotion_name = SELECTED_CLASSES[predicted_class]
    emotion_color = CLASS_COLORS[emotion_name]
    
    cv2.rectangle(overlay, (10, 10), (w - 10, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # 감정 레이블
    cv2.putText(frame, f"Emotion: {emotion_name.upper()}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, emotion_color, 2)
    cv2.putText(frame, f"Confidence: {confidence:.2%}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 확률 막대 그래프 (우측)
    bar_width = 200
    bar_height = 25
    bar_x = w - bar_width - 20
    bar_y_start = 100
    
    # 배경
    cv2.rectangle(frame, (bar_x - 10, bar_y_start - 10), 
                  (w - 10, bar_y_start + (bar_height + 10) * NUM_CLASSES + 10), 
                  (0, 0, 0), -1)
    
    # 각 클래스별 확률 막대
    for i, (cls_name, prob) in enumerate(zip(SELECTED_CLASSES, probabilities)):
        y = bar_y_start + i * (bar_height + 10)
        
        # 막대 배경
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), 
                      (50, 50, 50), -1)
        
        # 확률 막대
        filled_width = int(bar_width * prob)
        color = CLASS_COLORS[cls_name]
        cv2.rectangle(frame, (bar_x, y), (bar_x + filled_width, y + bar_height), 
                      color, -1)
        
        # 레이블 및 퍼센트
        cv2.putText(frame, f"{cls_name}", (bar_x - 80, y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{prob:.1%}", (bar_x + bar_width + 10, y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # FPS 표시 (하단 좌측)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

# ============================================
# ONNX 추론 설정
# ============================================
def setup_onnx_inference(onnx_path):
    """ONNX Runtime 세션 설정"""
    
    # 세션 옵션 설정
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # CPU 스레드 수 설정 (성능 최적화)
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    
    # 프로바이더 선택 (CPU 또는 CUDA)
    providers = ['CPUExecutionProvider']
    
    # CUDA 사용 가능 여부 확인
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
        print("✅ CUDA available - using GPU acceleration")
    else:
        print("ℹ️  CUDA not available - using CPU")
    
    # 세션 생성
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    
    # 입력/출력 정보
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"   Input: {input_name}")
    print(f"   Output: {output_name}")
    
    return session, input_name, output_name

# ============================================
# 메인 실행
# ============================================
def main():
    print("="*70)
    print("EMOTION RECOGNITION - WEBCAM INFERENCE (ONNX)")
    print("="*70)
    
    # ONNX 모델 로드
    print("\n[1/3] Loading ONNX model...")
    try:
        session, input_name, output_name = setup_onnx_inference(ONNX_PATH)
        print(f" ONNX model loaded: {ONNX_PATH}")
    except Exception as e:
        print(f" Failed to load ONNX model: {e}")
        return
    
    # 웹캠 설정
    print("\n[2/3] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Failed to open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(" Webcam opened")
    
    # FPS 계산용
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    print("\n[3/3] Starting inference...")
    print("Press 'q' to quit, 's' to save screenshot\n")
    
    try:
        while True:
            # 프레임 캡처
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # 전처리
            input_data = preprocess_frame(frame)
            
            # ONNX 추론
            output_data = session.run([output_name], {input_name: input_data})[0]
            
            # 후처리
            predicted_class, confidence, probabilities = postprocess_output(output_data)
            
            # FPS 계산
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:
                current_fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()
            
            # 시각화
            display_frame = draw_results(frame, predicted_class, confidence, 
                                        probabilities, current_fps)
            
            # 화면 출력
            cv2.imshow('Emotion Recognition (ONNX)', display_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 스크린샷 저장
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f" Screenshot saved: {filename}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\n Error during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 정리
        cap.release()
        cv2.destroyAllWindows()
        print("\n Cleanup complete")
        print("="*70)

if __name__ == "__main__":
    main()