import cv2
import numpy as np
import os
from hailo_platform import VDevice, HailoSchedulingAlgorithm

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# output_metrix = ['positive', 'neutral', 'negative']
output_metrix = ['angry', 'disgust', 'happy', 'neutral', 'surprise']
# output_metrix = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# 추후 NPU 추론으로 바꿀 수도 있음
def face_detect(image):
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_face_roi(image, faces):
    if faces is None or len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    face_roi = image[y:y + h, x:x + w]

    return face_roi, (x, y, w, h)

def preprocess(image, target_size):
    # Resize the image to the target size
    resized_image = cv2.resize(image, (target_size[1], target_size[0]))

    # Convert the image to RGB (if needed)
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Expand dimensions to match model input shape (1, height, width, channels)
    input_tensor = np.expand_dims(rgb_image, axis=0).astype(np.uint8)

    return input_tensor

def postprocess(frame, outputs, bbox):
    x, y, w, h = bbox

    out = outputs
    probs = np.squeeze(out)

    #print(probs)

    if probs.ndim == 2 and probs.shape[0] == 1:
        probs = probs[0]
    
    pred_idx = int(np.argmax(probs))

    # Map to human label for common 2-class mask model, otherwise generic label
    emotion = output_metrix[pred_idx]

    # Draw rectangle around face and label with predicted emotion
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return frame

def run_camera_inference(hef_path):
    timeout_ms = 1000
    
    # 카메라 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다")
        return
    
    print("✅ 카메라 열림")
    
    # VDevice 설정
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    with VDevice(params) as vdevice:
        # InferModel 생성
        infer_model = vdevice.create_infer_model(hef_path)
        
        # 입출력 정보 확인
        input_shape = infer_model.input().shape
        output_shape = infer_model.output().shape
        
        print(f"입력 형태: {input_shape}")
        print(f"출력 형태: {output_shape}")
        
        # 모델 설정
        with infer_model.configure() as configured_infer_model:
            # Bindings 생성
            bindings = configured_infer_model.create_bindings()
            
            frame_count = 0
            
            while True:
                # 프레임 읽기
                # Capture frame-by-frame
                ret, frame = cap.read()

                # Convert frame to grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces in the frame
                faces = face_detect(gray_frame)

                try: 
                    face_roi, (x, y, w, h) = extract_face_roi(gray_frame, faces)
                except Exception as e:
                    face_roi = None
                    x, y, w, h = 0, 0, 0, 0

                if face_roi is None:
                    # no faces -> continue to next frame
                    frame_count += 1
                    cv2.imshow('Hailo Camera', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # Preprocess face_roi to match ONNX model input
                # Determine target size from model input shape if available
                ih = input_shape[1]
                iw = input_shape[0]

                input_tensor = preprocess(face_roi, (iw, ih))

                #print(input_shape)

                # NPU 추론

                # 입력 버퍼 설정
                bindings.input().set_buffer(input_tensor)

                # 출력 버퍼 준비
                output_buffer = np.empty(output_shape, dtype=np.uint8)
                bindings.output().set_buffer(output_buffer)

                # 동기 추론
                configured_infer_model.run([bindings], timeout_ms)

                # 결과 가져오기
                outputs = bindings.output().get_buffer()

                # NPU 추론 끝
                
                frame = postprocess(frame, outputs, (x, y, w, h))

                frame_count += 1
                
                # 화면 표시
                cv2.imshow('Hailo Camera', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 총 {frame_count} 프레임 처리")

if __name__ == "__main__":
    run_camera_inference("../../models/best_model_float32_5class.hef")
    # run_camera_inference("../../models/resmasking.hef")
