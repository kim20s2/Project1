import cv2
import numpy as np
import os
from hailo_platform import VDevice, HailoSchedulingAlgorithm

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
output_metrix = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

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

                # Convert grayscale frame to RGB format
                rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

                # Detect faces in the frame
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    # Extract the face ROI (Region of Interest)
                    face_roi = rgb_frame[y:y + h, x:x + w]

                    # Preprocess face_roi to match ONNX model input
                    # Determine target size from model input shape if available
                    try:
                        _, c, ih, iw = [int(s) if s is not None else None for s in input_shape]
                    except Exception:
                        ih, iw = 224, 224
                    if ih is None or iw is None:
                        ih, iw = 224, 224

                    resized = cv2.resize(face_roi, (iw, ih))
                    # Model likely expects floats in [0,1] and channel-first format
                    input_tensor = resized.astype(np.float32) / 255.0
                    input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
                    input_tensor = np.expand_dims(input_tensor, axis=0)  # add batch

                    # 입력 버퍼 설정
                    bindings.input().set_buffer(input_tensor)
                
                    # 출력 버퍼 준비
                    output_buffer = np.empty(output_shape, dtype=np.uint8)
                    bindings.output().set_buffer(output_buffer)
                
                    # 동기 추론
                    configured_infer_model.run([bindings], timeout_ms)
                
                    # 결과 가져오기
                    outputs = bindings.output().get_buffer()

                    # Interpret output: try to get probabilities and pick argmax
                    out = outputs[0]
                    probs = np.squeeze(out)
                    if probs.ndim == 2 and probs.shape[0] == 1:
                        probs = probs[0]
                    pred_idx = int(np.argmax(probs))

                    # Map to human label for common 2-class mask model, otherwise generic label
                    if probs.size == 2:
                        labels = ['NoMask', 'Mask']
                        emotion = labels[pred_idx]
                    else:
                        emotion = output_metrix[pred_idx]

                    # Draw rectangle around face and label with predicted emotion
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                frame_count += 1
                
                # 화면 표시
                cv2.imshow('Hailo Camera', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 총 {frame_count} 프레임 처리")

if __name__ == "__main__":
    run_camera_inference("resmasking.hef")
