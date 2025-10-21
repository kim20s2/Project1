import numpy as np
import time
from collections import deque
from .pose_analyzer import PoseAnalyzer

class SimpleKalmanFilter:
    def __init__(self, process_noise=0.1, measurement_noise=0.1):
        """
        Kalman Filter를 이용한 keypoint 추적기
        
        Args:
            process_noise: 시스템 프로세스 노이즈 (작을수록 움직임이 부드럽게)
            measurement_noise: 측정 노이즈 (작을수록 측정값을 더 신뢰)
        """
        self.num_keypoints = 17
        self.filters = []
        self.last_valid_positions = np.zeros((self.num_keypoints, 2))
        self.position_history = [deque(maxlen=10) for _ in range(self.num_keypoints)]  # 히스토리 길이 증가
        self.velocity_history = [deque(maxlen=10) for _ in range(self.num_keypoints)]  # 속도 히스토리 추가
        self.confidence_threshold = 0.3
        self.velocity_smoothing = 0.7  # 속도 스무딩 계수leKalmanFilter:
    def __init__(self, process_noise=0.1, measurement_noise=0.1):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initialized = False
        self.last_timestamp = None
        
        # 상태 벡터 [x, y, vx, vy, ax, ay]
        self.state = np.zeros(6)
        self.covariance = np.eye(6) * 1000  # 초기 불확실성
        
        # 이전 측정값 저장
        self.prev_measurement = None
        self.velocity_history = deque(maxlen=5)
    
    def predict(self, dt):
        # 상태 전이 행렬
        F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 상태 예측
        self.state = F @ self.state
        
        # 프로세스 노이즈 행렬
        Q = np.zeros((6, 6))
        Q[0:2, 0:2] = np.eye(2) * (dt**4 / 4) * self.process_noise  # 위치
        Q[2:4, 2:4] = np.eye(2) * (dt**2) * self.process_noise      # 속도
        Q[4:6, 4:6] = np.eye(2) * self.process_noise                # 가속도
        
        # 공분산 업데이트
        self.covariance = F @ self.covariance @ F.T + Q
    
    def update(self, measurement, timestamp=None):
        current_time = timestamp if timestamp is not None else time.time()
        
        if not self.initialized:
            self.state[:2] = measurement
            self.initialized = True
            self.last_timestamp = current_time
            self.prev_measurement = measurement
            return self.state[:2]
        
        # 시간 간격 계산
        dt = current_time - self.last_timestamp
        dt = max(0.001, min(dt, 0.1))  # 극단적인 dt 방지
        self.last_timestamp = current_time
        
        # 현재 속도 추정
        if self.prev_measurement is not None:
            velocity = (measurement - self.prev_measurement) / dt
            self.velocity_history.append(velocity)
        
        # 속도와 가속도 추정
        if len(self.velocity_history) > 1:
            avg_velocity = np.mean(list(self.velocity_history), axis=0)
            self.state[2:4] = avg_velocity
            if len(self.velocity_history) > 2:
                acceleration = (self.velocity_history[-1] - self.velocity_history[-2]) / dt
                self.state[4:6] = acceleration * 0.5  # 가속도는 약하게 반영
        
        self.predict(dt)
        
        # 측정 행렬
        H = np.zeros((2, 6))
        H[:2, :2] = np.eye(2)
        
        # 측정 노이즈
        R = np.eye(2) * self.measurement_noise
        
        # Kalman Gain 계산
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        # 측정값으로 상태 업데이트
        y = measurement - H @ self.state
        self.state = self.state + K @ y
        
        # 공분산 업데이트
        I = np.eye(6)
        self.covariance = (I - K @ H) @ self.covariance
        
        # 이전 측정값 저장
        self.prev_measurement = measurement
        
        return self.state[:2]

class KalmanPoseTracker:
    def __init__(self, process_noise=0.1, measurement_noise=0.1):
        """
        Kalman Filter를 이용한 keypoint 추적기
        
        Args:
            process_noise: 시스템 프로세스 노이즈 (작을수록 움직임이 부드러움)
            measurement_noise: 측정 노이즈 (작을수록 측정값을 더 신뢰)
        """
        self.num_keypoints = 17
        self.filters = []
        self.last_valid_positions = np.zeros((self.num_keypoints, 2))
        self.position_history = [deque(maxlen=5) for _ in range(self.num_keypoints)]
        self.velocity_history = [deque(maxlen=5) for _ in range(self.num_keypoints)]
        self.velocity_smoothing = 0.7  # 속도 스무딩 계수
        self.confidence_threshold = 0.3
        
        # 각 keypoint마다 별도의 Kalman Filter 생성
        for _ in range(self.num_keypoints):
            kf = SimpleKalmanFilter(process_noise, measurement_noise)
            self.filters.append(kf)
    
    def update(self, keypoints):
        """
        모든 keypoint에 대해 Kalman Filter 업데이트
        낮은 신뢰도의 키포인트는 이전 위치 기반으로 보간
        
        Args:
            keypoints: shape (num_keypoints, 3)의 배열, 각 행은 [x, y, confidence]
        
        Returns:
            filtered_keypoints: Kalman Filter로 필터링된 keypoints
        """
        filtered_keypoints = np.copy(keypoints)
        current_time = time.time()
        
        for i in range(self.num_keypoints):
            # 현재 키포인트의 신뢰도와 위치
            current_pos = keypoints[i, :2]
            confidence = keypoints[i, 2]
            
            # 현재 키포인트의 필터
            kf = self.filters[i]
            
            # 현재 위치와 이전 유효한 위치로부터 속도 계산
            if len(self.position_history[i]) > 0 and confidence >= self.confidence_threshold:
                dt = current_time - kf.last_timestamp if kf.last_timestamp else 0.033
                current_velocity = (current_pos - self.last_valid_positions[i]) / dt
                
                # 이전 속도들과 현재 속도를 혼합
                if len(self.velocity_history[i]) > 0:
                    avg_velocity = np.mean(list(self.velocity_history[i]), axis=0)
                    smoothed_velocity = self.velocity_smoothing * avg_velocity + \
                                      (1 - self.velocity_smoothing) * current_velocity
                    self.velocity_history[i].append(smoothed_velocity)
                else:
                    self.velocity_history[i].append(current_velocity)
            
            if confidence >= self.confidence_threshold:
                # 신뢰도가 높은 경우 칼만 필터 업데이트
                filtered_pos = kf.update(current_pos, current_time)
                self.last_valid_positions[i] = filtered_pos
                self.position_history[i].append(filtered_pos)
            else:
                # 신뢰도가 낮은 경우 예측값 사용
                if len(self.position_history[i]) > 0:
                    # 이전 위치와 속도를 고려한 예측
                    pos_prediction = np.mean(list(self.position_history[i]), axis=0)
                    if len(self.velocity_history[i]) > 0:
                        vel_prediction = np.mean(list(self.velocity_history[i]), axis=0)
                        dt = 0.033  # 30fps 기준
                        pos_prediction += vel_prediction * dt
                    filtered_pos = kf.update(pos_prediction, current_time)
                else:
                    # 히스토리가 없는 경우 마지막 유효한 위치 사용
                    filtered_pos = self.last_valid_positions[i]
            
            # 필터링된 위치 저장
            filtered_keypoints[i, :2] = filtered_pos
        
        return filtered_keypoints


class KalmanPoseAnalyzer(PoseAnalyzer):
    def __init__(self, window_size=30, process_noise=0.1, measurement_noise=0.1):
        """
        Kalman Filter를 적용한 Pose Analyzer
        
        Args:
            window_size: 이동 평균을 위한 윈도우 크기
            process_noise: Kalman Filter의 프로세스 노이즈
            measurement_noise: Kalman Filter의 측정 노이즈
        """
        super().__init__(window_size)
        self.tracker = KalmanPoseTracker(process_noise, measurement_noise)
    
    def analyze_pose(self, keypoints):
        """
        Kalman Filter로 필터링 후 포즈 분석 수행
        
        Args:
            keypoints: 원본 keypoints
        
        Returns:
            분석 결과 딕셔너리
        """
        # Kalman Filter로 keypoints 필터링
        filtered_keypoints = self.tracker.update(keypoints)
        
        # 부모 클래스의 분석 메서드 호출
        return super().analyze_pose(filtered_keypoints)