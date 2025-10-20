import cv2
import numpy as np
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
import time
from collections import deque
from functools import partial
from .pose_analyzer_kalman import KalmanPoseAnalyzer
from .pose_analyzer import PoseAnalyzer
from .pose_logger import create_pose_logger

import signal

class MSPNPostProcessor:
    def __init__(self):
        self.num_keypoints = 17
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        self.skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]
        
        self.colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
            (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
            (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
            (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
            (255, 0, 170)
        ]
    
    def dequantize(self, output_uint8):
        return output_uint8.astype(np.float32) / 255.0
    
    def get_keypoints_from_heatmap(self, heatmap, original_img_size):
        heatmap_h, heatmap_w, num_joints = heatmap.shape
        orig_h, orig_w = original_img_size
        
        keypoints = np.zeros((num_joints, 3))
        
        for i in range(num_joints):
            joint_heatmap = heatmap[:, :, i]
            max_val = np.max(joint_heatmap)
            max_idx = np.argmax(joint_heatmap)
            
            y = max_idx // heatmap_w
            x = max_idx % heatmap_w
            
            if 0 < x < heatmap_w - 1 and 0 < y < heatmap_h - 1:
                diff_x = joint_heatmap[y, x+1] - joint_heatmap[y, x-1]
                diff_y = joint_heatmap[y+1, x] - joint_heatmap[y-1, x]
                
                x += np.sign(diff_x) * 0.25 if abs(diff_x) > 0.01 else 0
                y += np.sign(diff_y) * 0.25 if abs(diff_y) > 0.01 else 0
            
            x_scaled = x * (orig_w / heatmap_w)
            y_scaled = y * (orig_h / heatmap_h)
            
            keypoints[i] = [x_scaled, y_scaled, max_val]
        
        return keypoints
    
    def process(self, hailo_output, original_img_size, conf_threshold=0.3):
        # Shape 확인 및 reshape
        if len(hailo_output.shape) == 1:
            # FCR format: flatten된 경우
            hailo_output = hailo_output.reshape(64, 48, 17)
        elif len(hailo_output.shape) == 4:
            # Batch dimension이 있는 경우
            hailo_output = hailo_output[0]
        elif len(hailo_output.shape) != 3:
            raise ValueError(f"Unexpected output shape: {hailo_output.shape}")
        
        # 이제 (64, 48, 17) 형태 보장
        heatmap = self.dequantize(hailo_output)
        keypoints = self.get_keypoints_from_heatmap(heatmap, original_img_size)
        keypoints[keypoints[:, 2] < conf_threshold, :] = 0
        
        return keypoints
    
    def visualize(self, image, keypoints, conf_threshold=0.3):
        img_vis = image.copy()
        
        for start_idx, end_idx in self.skeleton:
            if (keypoints[start_idx, 2] > conf_threshold and 
                keypoints[end_idx, 2] > conf_threshold):
                
                start_point = tuple(keypoints[start_idx, :2].astype(int))
                end_point = tuple(keypoints[end_idx, :2].astype(int))
                
                cv2.line(img_vis, start_point, end_point, (0, 255, 0), 3)
        
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > conf_threshold:
                color = self.colors[i % len(self.colors)]
                cv2.circle(img_vis, (int(x), int(y)), 5, color, -1)
                cv2.circle(img_vis, (int(x), int(y)), 7, (255, 255, 255), 2)
        
        return img_vis


class RealtimePoseEstimation:
    def __init__(self, hef_path, camera_device='/dev/video0', 
                 input_size=(192, 256), conf_threshold=0.3, batch_size=4, filtered=False):
        """
        실시간 Pose Estimation 클래스 (Async API)
        
        Args:
            hef_path: HEF 모델 파일 경로
            camera_device: 카메라 디바이스 경로 (/dev/video0 등)
            input_size: 모델 입력 크기 (width, height)
            conf_threshold: keypoint confidence threshold
            batch_size: 배치 크기
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.batch_size = batch_size
        self.timeout_ms = 10000
        self.filtered = filtered
        
        # Hailo 초기화
        print("Initializing Hailo device...")
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        params.group_id = "SHARED"
        params.multi_process_service = True
        
        self.vdevice = VDevice(params)
        
        # HEF 로드 및 InferModel 생성
        print(f"Loading model: {hef_path}")
        self.infer_model = self.vdevice.create_infer_model(hef_path)
        
        # 배치 크기 설정
        self.infer_model.set_batch_size(batch_size)
        
        # 입력/출력 포맷 설정 (UINT8 유지)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.infer_model.output().set_format_type(FormatType.UINT8)
        
        # 후처리 초기화
        self.post_processor = MSPNPostProcessor()
        
        # 카메라 초기화
        print(f"Opening camera: {camera_device}")
        self.cap = cv2.VideoCapture(camera_device)
        
        if not self.cap.isOpened():
            # 카메라 열기 실패 시 다른 인덱스 시도
            print(f"Failed to open {camera_device}, trying /dev/video1...")
            # self.cap = cv2.VideoCapture('/dev/video1')
            
            if not self.cap.isOpened():
                print("Warning: No camera found. Will try to continue anyway...")
                print("You can use a video file instead: --camera /path/to/video.mp4")
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FPS 계산용
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        # 결과 저장용
        self.latest_result = None
        self.result_ready = False

        if self.filtered:
            self.pose_analyzer = KalmanPoseAnalyzer(
                window_size=45,           # 이동 평균 윈도우 크기 증가
                process_noise=0.001,      # 프로세스 노이즈 대폭 감소 (더 부드럽게)
                measurement_noise=0.7     # 측정 노이즈 증가 (갑작스러운 변화 무시)
            )
        else:
            self.pose_analyzer = PoseAnalyzer(
                window_size=45,           # 이동 평균 윈도우 크기 증가
            )
        
        # Pose Analyzer 초기화 (Kalman Filter 적용)
        
        
        # 카메라 FPS 확인
        camera_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # XML 로거 초기화 (카메라 FPS 전달)
        self.pose_logger = create_pose_logger(fps=camera_fps)
        
        print(f"Initialization complete! Camera FPS: {camera_fps}")
    
    def preprocess(self, frame):
        """이미지 전처리"""
        img = cv2.resize(frame, self.input_size)
        # UINT8 유지, batch dimension 추가
        img = np.expand_dims(img, axis=0)  # (1, 256, 192, 3)
        return img
    
    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        
        return np.mean(self.fps_queue)
    
    def draw_info(self, frame, fps, inference_time, pose_analysis):
        """정보 오버레이"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # 기본 정보
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 분석 결과
        color_ok = (0, 255, 0)  # 녹색
        color_warning = (0, 0, 255)  # 빨간색
        color_gesture = (255, 165, 0)  # 주황색
        
        tilt, tilt_msg = pose_analysis['body_tilt']
        head_tilt, head_tilt_msg = pose_analysis['head_tilt']
        wrong_dist, dist_msg = pose_analysis['wrong_distance']
        gesture, gesture_msg = pose_analysis['gesture']
        
        cv2.putText(frame, dist_msg, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_warning if wrong_dist else color_ok, 2)
        cv2.putText(frame, tilt_msg, (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_warning if tilt else color_ok, 2)
        cv2.putText(frame, head_tilt_msg, (20, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_warning if head_tilt else color_ok, 2)
        # 제스처 감지 결과 표시
        cv2.putText(frame, gesture_msg, (20, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_gesture if gesture else color_ok, 2)
        
        # 종료 안내
        cv2.putText(frame, "Press 'q' to quit", (w - 250, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def inference_callback(self, completion_info, bindings, orig_size, start_time):
        """비동기 추론 완료 시 호출되는 콜백"""
        if completion_info.exception:
            print(f"Inference error: {completion_info.exception}")
            return
        
        try:
            # 추론 시간 계산
            inference_time = (time.time() - start_time) * 1000
            
            # Output 가져오기
            output_data = bindings.output().get_buffer()
            
            # Debug: Shape 확인
            # print(f"[DEBUG] Output shape: {output_data.shape}, dtype: {output_data.dtype}")
            
            # Batch dimension 처리
            if len(output_data.shape) == 4:
                # (batch, height, width, channels) → (height, width, channels)
                output_data = output_data[0]
            elif len(output_data.shape) == 1:
                # FCR format - flatten된 경우 reshape 필요
                # (64*48*17,) → (64, 48, 17)
                output_data = output_data.reshape(64, 48, 17)
            # else: 이미 (64, 48, 17) 형태면 그대로 사용
            
            # 후처리
            keypoints = self.post_processor.process(
                output_data,
                original_img_size=orig_size,
                conf_threshold=self.conf_threshold
            )
            
            # 결과 저장
            self.latest_result = {
                'keypoints': keypoints,
                'inference_time': inference_time
            }
            self.result_ready = True
            
        except Exception as e:
            print(f"Callback error: {e}")
            import traceback
            traceback.print_exc()
    
    def run(self):
        """메인 루프 (Async API)"""
        print("\nStarting real-time pose estimation...")
        print("Press 'q' to quit\n")
        
        try:
            # Configure 및 추론 루프
            with self.infer_model.configure() as configured_infer_model:
                
                while True:
                    # 프레임 읽기
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame")
                        break
                    
                    orig_h, orig_w = frame.shape[:2]
                    
                    # 전처리
                    input_data = self.preprocess(frame)

                    # print(input_data.shape)
                    
                    # Bindings 생성 및 버퍼 설정
                    bindings = configured_infer_model.create_bindings()
                    bindings.input().set_buffer(input_data)
                    bindings.output().set_buffer(
                        np.zeros(self.infer_model.output().shape, dtype=np.uint8)
                    )
                    
                    # 비동기 파이프라인 준비 대기
                    configured_infer_model.wait_for_async_ready(timeout_ms=self.timeout_ms)
                    
                    # 비동기 추론 시작
                    inference_start = time.time()
                    job = configured_infer_model.run_async(
                        [bindings],
                        partial(
                            self.inference_callback,
                            bindings=bindings,
                            orig_size=(orig_h, orig_w),
                            start_time=inference_start
                        )
                    )
                    
                    # 결과 대기
                    job.wait(self.timeout_ms)
                    
                    # 결과가 준비되면 시각화
                    if self.result_ready:
                        result = self.latest_result
                        result_frame = self.post_processor.visualize(
                            frame,
                            result['keypoints'],
                            self.conf_threshold
                        )
                        
                        # 포즈 분석
                        pose_analysis = self.pose_analyzer.analyze_pose(result['keypoints'])
                        
                        # XML로 포즈 데이터 저장
                        self.pose_logger.log_analysis(pose_analysis)
                        
                        # FPS 계산 및 정보 표시
                        fps = self.calculate_fps()
                        result_frame = self.draw_info(
                            result_frame,
                            fps,
                            result['inference_time'],
                            pose_analysis
                        )
                        
                        # 화면에 표시
                        cv2.imshow('Hailo Pose Estimation', result_frame)
                        
                        self.result_ready = False
                    
                    # 'q' 키로 종료
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'pose_logger'):
            self.pose_logger.close()
        print("Done!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Pose Estimation with Hailo')
    parser.add_argument('--hef', type=str, required=True,
                       help='Path to HEF model file')
    parser.add_argument('--camera', type=str, default='/dev/video0',
                       help='Camera device path (default: /dev/video0)')
    parser.add_argument('--conf', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--width', type=int, default=192,
                       help='Model input width (default: 192)')
    parser.add_argument('--height', type=int, default=256,
                       help='Model input height (default: 256)')
    parser.add_argument('--batch', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--filtered', type=bool, default=False,
                       help='Use Kalman filter for pose smoothing')
    
    args = parser.parse_args()
    
    # 실시간 추론 시작
    pose_estimator = RealtimePoseEstimation(
        hef_path=args.hef,
        camera_device=args.camera,
        input_size=(args.width, args.height),
        conf_threshold=args.conf,
        batch_size=args.batch,
        filtered=args.filtered
    )
    
    pose_estimator.run()


if __name__ == '__main__':
    main()
