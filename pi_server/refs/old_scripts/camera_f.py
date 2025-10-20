import cv2
import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm

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
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 전처리
                img = cv2.resize(frame, (input_shape[1], input_shape[0]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.uint8)
                
                # 입력 버퍼 설정
                bindings.input().set_buffer(img)
                
                # 출력 버퍼 준비
                output_buffer = np.empty(output_shape, dtype=np.uint8)
                bindings.output().set_buffer(output_buffer)
                
                # 동기 추론
                configured_infer_model.run([bindings], timeout_ms)
                
                # 결과 가져오기
                result = bindings.output().get_buffer()
                
                frame_count += 1
                
                # 첫 프레임에서 결과 확인
                result = result.reshape(40, 2)
                print(f"result : {result[:, 1]}")
                print(f"result : {result[:, 0]}")

                
                # 화면 표시
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Hailo Camera', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ 총 {frame_count} 프레임 처리")

if __name__ == "__main__":
    run_camera_inference("face_attr_resnet_v1_18.hef")
