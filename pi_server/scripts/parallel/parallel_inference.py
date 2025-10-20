import cv2
import numpy as np
import argparse
import time
from collections import deque
from functools import partial
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
from ..pose_est.pose_analyzer_kalman import KalmanPoseAnalyzer
# from pose_logger import create_pose_logger
from ..pose_est.pose_est_main import MSPNPostProcessor

class ParallelModelInference:
    def __init__(self, hef_path1, hef_path2, camera_device='/dev/video0', 
                 input_size=(192, 256), conf_threshold=0.3):
        """
        두 개의 모델을 병렬로 실행하는 클래스
        
        Args:
            hef_path1: 첫 번째 HEF 모델 파일 경로
            hef_path2: 두 번째 HEF 모델 파일 경로
            camera_device: 카메라 디바이스 경로
            input_size: 모델 입력 크기 (width, height)
            conf_threshold: keypoint confidence threshold
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.timeout_ms = 10000
        
        # Hailo 초기화
        print("Initializing Hailo device...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN  # 라운드 로빈 스케줄링
        
        self.vdevice = VDevice(params)
        
        # 두 개의 HEF 모델 로드
        print(f"Loading models: {hef_path1}, {hef_path2}")
        self.model1 = self.vdevice.create_infer_model(hef_path1)
        self.model2 = self.vdevice.create_infer_model(hef_path2)
        
        # 입력/출력 포맷 설정 (UINT8 사용)
        self.model1.input().set_format_type(FormatType.UINT8)
        self.model1.output().set_format_type(FormatType.UINT8)
        
        self.model2.input().set_format_type(FormatType.UINT8)
        self.model2.output().set_format_type(FormatType.UINT8)
            
        # 모델 configure
        self.configured_model1 = self.model1.configure()
        self.configured_model2 = self.model2.configure()
        
        # 카메라 초기화
        print(f"Opening camera: {camera_device}")
        self.cap = cv2.VideoCapture(camera_device)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {camera_device}")
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FPS 계산용
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # 결과 처리 및 시각화용
        self.results = {'model1': None, 'model2': None}
        self.frames_ready = 0
        self.postprocessor = MSPNPostProcessor()
        
        print("Initialization complete!")
    
    def preprocess(self, frame):
        """이미지 전처리"""
        img = cv2.resize(frame, self.input_size)
        img = np.expand_dims(img, axis=0)  # batch dimension 추가
        return img
    
    def run(self):
        """메인 루프 (병렬 실행)"""
        print("\nStarting parallel inference...")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # 프레임 전처리
                input_data = self.preprocess(frame)
                
                # 두 모델에 대해 추론 실행
                start_time = time.time()

                # configured 모델에서 바인딩 생성
                bindings1 = self.configured_model1.create_bindings()
                bindings2 = self.configured_model2.create_bindings()

                # 입력 데이터 설정
                input_data_uint8 = input_data.astype(np.uint8)
                bindings1.input().set_buffer(input_data_uint8)
                bindings2.input().set_buffer(input_data_uint8)

                # 결과를 저장할 버퍼 설정
                output_shape1 = self.model1.output().shape
                output_shape2 = self.model2.output().shape
                bindings1.output().set_buffer(np.zeros(output_shape1, dtype=np.uint8))
                bindings2.output().set_buffer(np.zeros(output_shape2, dtype=np.uint8))

                # 콜백 함수 정의
                def inference_callback(completion_info, model_name, bindings):
                    if completion_info.exception:
                        print(f"Error in {model_name}: {completion_info.exception}")
                        return
                    self.results[model_name] = bindings.output().get_buffer()

                # 비동기 파이프라인이 준비될 때까지 대기
                self.configured_model1.wait_for_async_ready(timeout_ms=self.timeout_ms)
                self.configured_model2.wait_for_async_ready(timeout_ms=self.timeout_ms)

                # 비동기 추론 시작
                job1 = self.configured_model1.run_async([bindings1], 
                    partial(inference_callback, model_name='model1', bindings=bindings1))
                job2 = self.configured_model2.run_async([bindings2], 
                    partial(inference_callback, model_name='model2', bindings=bindings2))

                # 두 작업이 완료될 때까지 대기
                try:
                    job1.wait(self.timeout_ms)
                    job2.wait(self.timeout_ms)
                except Exception as e:
                    print(f"Inference error: {e}")
                    return None
                
                inference_time = (time.time() - start_time) * 1000  # ms 단위
                
                # 결과 처리 및 시각화
                frame_vis2 = frame.copy()
                orig_h, orig_w = frame.shape[:2]
                if self.results['model1'] is not None:
                    keypoints1 = self.postprocessor.process(self.results['model1'], (orig_h, orig_w), self.conf_threshold)
                    frame = self.postprocessor.visualize(frame, keypoints1, self.conf_threshold)
                    cv2.putText(frame, "Model 1", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                if self.results['model2'] is not None:
                    keypoints2 = self.postprocessor.process(self.results['model2'], (orig_h, orig_w), self.conf_threshold)
                    frame_vis2 = self.postprocessor.visualize(frame_vis2, keypoints2, self.conf_threshold)
                    cv2.putText(frame_vis2, "Model 2", (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    
                    # 두 프레임을 나란히 표시
                    combined_frame = np.hstack((frame, frame_vis2))
                    frame = cv2.resize(combined_frame, (1280, 480))
                
                # 추론 시간 표시
                cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # FPS 계산 및 표시
                current_time = time.time()
                fps = 1.0 / (current_time - self.last_time)
                self.last_time = current_time
                self.fps_queue.append(fps)
                avg_fps = np.mean(self.fps_queue)
                
                cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # 결과 표시
                cv2.imshow('Parallel Inference', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error during inference: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done!")


def main():
    parser = argparse.ArgumentParser(description='Parallel Model Inference with Hailo')
    parser.add_argument('--hef1', type=str, required=True,
                       help='Path to first HEF model file')
    parser.add_argument('--hef2', type=str, required=True,
                       help='Path to second HEF model file')
    parser.add_argument('--camera', type=str, default='/dev/video0',
                       help='Camera device path (default: /dev/video0)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--width', type=int, default=192,
                       help='Model input width (default: 192)')
    parser.add_argument('--height', type=int, default=256,
                       help='Model input height (default: 256)')
    
    args = parser.parse_args()
    
    # 병렬 추론 시작
    inference = ParallelModelInference(
        hef_path1=args.hef1,
        hef_path2=args.hef2,
        camera_device=args.camera,
        input_size=(args.width, args.height),
        conf_threshold=args.conf
    )
    
    inference.run()


if __name__ == '__main__':
    main()