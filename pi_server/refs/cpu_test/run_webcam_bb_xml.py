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
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'happy': (0, 255, 255),
    'neutral': (255, 255, 255),
    'surprise': (255, 128, 0)
}

FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# 로깅 설정
SAMPLING_INTERVAL = 1.0  # 1초마다 샘플링 
LOG_FILE_PREFIX = "emotion_session"

# ============================================
# 데이터 로거 클래스
# ============================================
class EmotionLogger:
    def __init__(self, sampling_interval=1.0):
        self.sampling_interval = sampling_interval
        self.start_time = None
        self.end_time = None
        self.samples = []
        self.last_sample_time = 0
        
    def start_session(self):
        self.start_time = datetime.now()
        self.samples = []
        self.last_sample_time = time.time()
        print(f"\n📊 Session started: {self.start_time.isoformat()}")
    
    def add_sample(self, emotion, confidence, probabilities, num_faces):
        current_time = time.time()
        
        # 샘플링 간격 체크
        if current_time - self.last_sample_time >= self.sampling_interval:
            timestamp = datetime.now()
            elapsed = (timestamp - self.start_time).total_seconds()
            
            sample = {
                'timestamp': timestamp.isoformat(),
                'elapsed_seconds': round(elapsed, 3),
                'emotion': emotion,
                'confidence': round(confidence, 4),
                'probabilities': {
                    cls: round(prob, 4) 
                    for cls, prob in zip(SELECTED_CLASSES, probabilities)
                },
                'num_faces': num_faces
            }
            
            self.samples.append(sample)
            self.last_sample_time = current_time
            
            print(f"  Sample {len(self.samples)}: {emotion} ({confidence:.2%}) - {num_faces} face(s)")
    
    def end_session(self):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        print(f"\n Session ended: {self.end_time.isoformat()}")
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Samples: {len(self.samples)}")
        return duration
    
    def save_to_xml(self, filename=None):
        if not self.start_time or not self.end_time:
            print(" No session data to save")
            return
        
        if filename is None:
            filename = f"{LOG_FILE_PREFIX}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.xml"
        
        duration = (self.end_time - self.start_time).total_seconds()
        
        # XML 루트 생성
        root = ET.Element('emotion_session')
        root.set('start_time', self.start_time.isoformat())
        root.set('end_time', self.end_time.isoformat())
        root.set('duration_seconds', str(round(duration, 1)))
        root.set('total_samples', str(len(self.samples)))
        root.set('sampling_interval', str(self.sampling_interval))
        
        # 통계 정보
        stats = ET.SubElement(root, 'statistics')
        
        # 감정별 출현 횟수
        emotion_counts = {}
        for sample in self.samples:
            emotion = sample['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in emotion_counts.items():
            emotion_stat = ET.SubElement(stats, 'emotion_count')
            emotion_stat.set('name', emotion)
            emotion_stat.set('count', str(count))
            emotion_stat.set('percentage', str(round(count / len(self.samples) * 100, 1)))
        
        # 평균 신뢰도
        avg_confidence = sum(s['confidence'] for s in self.samples) / len(self.samples)
        confidence_stat = ET.SubElement(stats, 'average_confidence')
        confidence_stat.text = str(round(avg_confidence, 4))
        
        # 샘플 데이터
        samples_elem = ET.SubElement(root, 'samples')
        
        for i, sample in enumerate(self.samples):
            sample_elem = ET.SubElement(samples_elem, 'sample')
            sample_elem.set('id', str(i + 1))
            sample_elem.set('timestamp', sample['timestamp'])
            sample_elem.set('elapsed_seconds', str(sample['elapsed_seconds']))
            
            # 기본 정보
            ET.SubElement(sample_elem, 'emotion').text = sample['emotion']
            ET.SubElement(sample_elem, 'confidence').text = str(sample['confidence'])
            ET.SubElement(sample_elem, 'num_faces').text = str(sample['num_faces'])
            
            # 확률 분포
            probs_elem = ET.SubElement(sample_elem, 'probabilities')
            for emotion, prob in sample['probabilities'].items():
                prob_elem = ET.SubElement(probs_elem, 'probability')
                prob_elem.set('emotion', emotion)
                prob_elem.text = str(prob)
        
        # 예쁘게 포맷팅
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        
        # 파일 저장
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        print(f"\n Session saved to: {filename}")
        return filename
    

# ============================================
# 얼굴 검출기 초기화
# ============================================
def setup_face_detector():
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load face cascade from {FACE_CASCADE_PATH}")
    return face_cascade

# ============================================
# 얼굴 검출
# ============================================
def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces

# ============================================
# 전처리
# ============================================
def preprocess_face(frame, x, y, w, h):
    margin = int(w * 0.2)
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(frame.shape[1], x + w + margin)
    y2 = min(frame.shape[0], y + h + margin)
    
    face_roi = frame[y1:y2, x1:x2]
    
    if face_roi.size == 0:
        return None
    
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # 리사이즈
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    
    # 채널 복제 
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # 정규화
    normalized = rgb.astype(np.float32) / 255.0
    normalized = normalized * 2.0 - 1.0
    normalized = np.clip(normalized, -1.0, 1.0)
    
    input_data = np.expand_dims(normalized, axis=0)
    
    return input_data

# ============================================
# 후처리
# ============================================
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def postprocess_output(output):
    # logits -> 확률
    probabilities = softmax(output[0])
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return predicted_class, confidence, probabilities

# ============================================
# 시각화
# ============================================
def draw_face_box(frame, x, y, w, h, emotion_name, confidence):
    color = CLASS_COLORS[emotion_name]
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    label = f"{emotion_name.upper()} {confidence:.1%}"
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_w, label_h = label_size
    
    label_x = x
    label_y = y - 10 if y - 10 > label_h else y + h + 25
    
    cv2.rectangle(frame, 
                  (label_x, label_y - label_h - 5), 
                  (label_x + label_w + 10, label_y + 5), 
                  color, -1)
    
    cv2.putText(frame, label, (label_x + 5, label_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return frame

def draw_probability_bars(frame, probabilities):
    h, w = frame.shape[:2]
    
    bar_width = 200
    bar_height = 25
    bar_x = w - bar_width - 20
    bar_y_start = 20
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (bar_x - 10, bar_y_start - 10), 
                  (w - 10, bar_y_start + (bar_height + 10) * NUM_CLASSES + 10), 
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    for i, (cls_name, prob) in enumerate(zip(SELECTED_CLASSES, probabilities)):
        y = bar_y_start + i * (bar_height + 10)
        
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_width, y + bar_height), 
                      (50, 50, 50), -1)
        
        filled_width = int(bar_width * prob)
        color = CLASS_COLORS[cls_name]
        cv2.rectangle(frame, (bar_x, y), (bar_x + filled_width, y + bar_height), 
                      color, -1)
        
        cv2.putText(frame, f"{cls_name}", (bar_x - 80, y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{prob:.1%}", (bar_x + bar_width + 10, y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def draw_info(frame, num_faces, fps):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, f"Faces Detected: {num_faces}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

# ============================================
# ONNX 설정
# ============================================
def setup_onnx_inference(onnx_path):
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4
    
    providers = ['CPUExecutionProvider']
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        providers.insert(0, 'CUDAExecutionProvider')
        print(" CUDA available - using GPU acceleration")
    else:
        print(" CUDA not available - using CPU")
    
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"   Input name: {input_name}")
    print(f"   Output name: {output_name}")
    
    return session, input_name, output_name

# ============================================
# 메인
# ============================================
def main():
    print("="*70)
    print("EMOTION RECOGNITION - WEBCAM INFERENCE WITH FACE DETECTION")
    print("="*70)
    
    print("\n[1/4] Loading face detector...")
    try:
        face_cascade = setup_face_detector()
        print("Face detector loaded")
    except Exception as e:
        print(f"Failed to load face detector: {e}")
        return
    
    print("\n[2/4] Loading ONNX model...")
    try:
        session, input_name, output_name = setup_onnx_inference(ONNX_PATH)
        print(f"ONNX model loaded: {ONNX_PATH}")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
        return
    
    print("\n[3/4] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    print("Webcam opened")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    print("\n[4/4] Starting inference...")
    print("="*70)
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("="*70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            display_frame = frame.copy()
            
            faces = detect_faces(frame, face_cascade)
            
            last_probabilities = None
            
            for (x, y, w, h) in faces:
                input_data = preprocess_face(frame, x, y, w, h)
                
                if input_data is None:
                    continue
                
                output_data = session.run([output_name], {input_name: input_data})[0]
                
                predicted_class, confidence, probabilities = postprocess_output(output_data)
                emotion_name = SELECTED_CLASSES[predicted_class]
                
                display_frame = draw_face_box(display_frame, x, y, w, h, 
                                             emotion_name, confidence)
                
                last_probabilities = probabilities
            
            if last_probabilities is not None:
                display_frame = draw_probability_bars(display_frame, last_probabilities)
            
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:
                current_fps = fps_frame_count / elapsed_time
                fps_frame_count = 0
                fps_start_time = time.time()
            
            display_frame = draw_info(display_frame, len(faces), current_fps)
            
            cv2.imshow('Emotion Recognition', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"emotion_capture_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"📸 Screenshot saved: {filename}")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError during inference: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCleanup complete")
        print("="*70)

if __name__ == "__main__":
    main()