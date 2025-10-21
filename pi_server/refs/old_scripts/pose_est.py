import cv2
import numpy as np
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType
import time
from collections import deque

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
        if len(hailo_output.shape) == 1:
            hailo_output = hailo_output.reshape(64, 48, 17)
        
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
                 input_size=(192, 256), conf_threshold=0.3):
        """
        실시간 Pose Estimation 클래스
        
        Args:
            hef_path: HEF 모델 파일 경로
            camera_device: 카메라 디바이스 경로 (/dev/video0 등)
            input_size: 모델 입력 크기 (width, height)
            conf_threshold: keypoint confidence threshold
        """
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        
        # Hailo 초기화
        print("Initializing Hailo device...")
        self.device = VDevice()
        
        # HEF 로드
        print(f"Loading model: {hef_path}")
        self.hef = self.device.create_infer_model(hef_path)
        
        # 후처리 초기화
        self.post_processor = MSPNPostProcessor()
        
        # 카메라 초기화
        print(f"Opening camera: {camera_device}")
        self.cap = cv2.VideoCapture(camera_device)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {camera_device}")
        
        # 카메라 설정
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # FPS 계산용
        self.fps_queue = deque(maxlen=30)
        self.last_time = time.time()
        
        print("Initialization complete!")
    
    def preprocess(self, frame):
        """이미지 전처리"""
        # 모델 입력 크기로 리사이즈
        img = cv2.resize(frame, self.input_size)
        
        # NHWC 형식으로 변환 (이미 NHWC면 그대로)
        # UINT8 유지 (모델이 UINT8 입력)
        img = np.expand_dims(img, axis=0)  # (1, 256, 192, 3)
        
        return img
    
    def calculate_fps(self):
        """FPS 계산"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time)
        self.last_time = current_time
        self.fps_queue.append(fps)
        
        return np.mean(self.fps_queue)
    
    def draw_info(self, frame, fps, inference_time):
        """정보 오버레이"""
        h, w = frame.shape[:2]
        
        # 반투명 배경
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # 텍스트
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # 종료 안내
        cv2.putText(frame, "Press 'q' to quit", (w - 250, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """메인 루프"""
        print("\nStarting real-time pose estimation...")
        print("Press 'q' to quit\n")
        
        try:
            while True:
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                orig_h, orig_w = frame.shape[:2]
                
                # 전처리
                input_data = self.preprocess(frame)
                
                # Hailo 추론
                inference_start = time.time()
                output = self.hef.run(input_data)
                inference_time = (time.time() - inference_start) * 1000
                
                # 후처리
                # Output key 찾기 (실제 모델의 output layer 이름 확인 필요)
                output_key = list(output.keys())[0]
                keypoints = self.post_processor.process(
                    output[output_key],
                    original_img_size=(orig_h, orig_w),
                    conf_threshold=self.conf_threshold
                )
                
                # 시각화
                result_frame = self.post_processor.visualize(
                    frame, keypoints, self.conf_threshold
                )
                
                # FPS 계산 및 정보 표시
                fps = self.calculate_fps()
                result_frame = self.draw_info(result_frame, fps, inference_time)
                
                # 화면에 표시
                cv2.imshow('Hailo Pose Estimation', result_frame)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
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
    
    args = parser.parse_args()
    
    # 실시간 추론 시작
    pose_estimator = RealtimePoseEstimation(
        hef_path=args.hef,
        camera_device=args.camera,
        input_size=(args.width, args.height),
        conf_threshold=args.conf
    )
    
    pose_estimator.run()


if __name__ == '__main__':
    main()
