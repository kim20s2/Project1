import numpy as np
from collections import deque

class PoseAnalyzer:
    def __init__(self, window_size=30):
        # 키포인트 인덱스
        self.left_eye_idx = 1
        self.right_eye_idx = 2
        self.left_shoulder_idx = 5
        self.right_shoulder_idx = 6
        self.left_hip_idx = 11
        self.right_hip_idx = 12
        self.left_wrist_idx = 9
        self.right_wrist_idx = 10
        
        # 임계값 설정
        self.body_tilt_threshold = 1.1     # 몸 기울기 임계값 (길이 비율 차이)
        self.body_tilt_margin = 1.15
        self.head_tilt_threshold = 3.5     # 머리 기울기 임계값 (길이 비율)
        self.max_diff = 0
        
        # 제스처 관련 임계값
        self.gesture_distance_threshold = 5.5  # 손과 눈 사이의 최대 거리 (픽셀)
        self.hands_cross_threshold = 0.3      # 손이 교차된 것으로 판단할 거리 임계값 (픽셀)
        
        # 카메라 거리 관련 임계값
        self.eye_dist_min = 5.0     # 최소 눈 거리 (픽셀) - 너무 멀때
        self.eye_dist_max = 40.0    # 최대 눈 거리 (픽셀) - 너무 가까울때
        self.ideal_eye_dist = 20.0   # 이상적인 눈 거리 (픽셀)
    
    # 무릎 떨림과 얼굴 회전 감지 기능 제거됨
    
    def detect_body_tilt(self, keypoints):
        left_shoulder = keypoints[self.left_shoulder_idx]
        right_shoulder = keypoints[self.right_shoulder_idx]
        left_hip = keypoints[self.left_hip_idx]
        right_hip = keypoints[self.right_hip_idx]
        
        # 신뢰도가 낮으면 False 반환
        if (left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3 or
            left_hip[2] < 0.3 or right_hip[2] < 0.3):
            return False, "Low confidence in body keypoints"
        
        ## 앞/뒤 기울어짐 감지
        ud_bool = False

        mid_shoulder = (left_shoulder[:2]+right_shoulder[:2])/2
        mid_hip = (left_hip[:2]+right_hip[:2])/2

        diff = np.linalg.norm(mid_shoulder[:2]- mid_hip[:2])

        if self.max_diff < diff:
            self.max_diff = diff
            print(self.max_diff)
        
        if self.max_diff/diff > self.body_tilt_margin:
            ud_bool = True

        ud_radio = self.max_diff/diff

        ## 좌/우 기울어짐 감지
        # 왼쪽 길이 계산
        left_length = np.linalg.norm(
            left_shoulder[:2] - left_hip[:2]
        )
        
        # 오른쪽 길이 계산
        right_length = np.linalg.norm(
            right_shoulder[:2] - right_hip[:2]
        )

        # # 긴 쪽과 짧은 쪽의 비율 계산
        ratio = max(left_length, right_length)/min(left_length, right_length)
        
        is_tilted = ratio > self.body_tilt_threshold or ud_bool
        message = f"Body asymmetry detected ({ratio:.2f}, {ud_radio:.2f})" if is_tilted else f"Symmetric posture({ratio:.2f}, {ud_radio:.2f})"
        return is_tilted, message

    def detect_distance(self, keypoints):
        """카메라와의 거리 추정 (눈 사이 거리 기반)"""
        left_eye = keypoints[self.left_eye_idx]
        right_eye = keypoints[self.right_eye_idx]
        
        # 신뢰도가 낮으면 False 반환
        if left_eye[2] < 0.3 or right_eye[2] < 0.3:
            return False, "Low confidence in eye detection"
        
        # 두 눈 사이의 거리 계산
        eye_distance = np.linalg.norm(left_eye[:2] - right_eye[:2])

        self.eye_dist = eye_distance
        
        # 거리 상태 판단
        if eye_distance < self.eye_dist_min:
            message = f"Too far (eye dist: {eye_distance:.1f}px)"
            is_wrong_distance = True
        elif eye_distance > self.eye_dist_max:
            message = f"Too close (eye dist: {eye_distance:.1f}px)"
            is_wrong_distance = True
        else:
            # 이상적인 거리와의 차이 백분율 계산
            dist_diff_percent = abs(eye_distance - self.ideal_eye_dist) / self.ideal_eye_dist * 100
            message = f"Good distance (diff: {dist_diff_percent:.1f}%)"
            is_wrong_distance = False
        
        return is_wrong_distance, message
    
    def detect_head_tilt(self, keypoints):
        """머리 기울임 감지 (어깨선과 눈의 각도 비교)"""
        left_eye = keypoints[self.left_eye_idx]
        right_eye = keypoints[self.right_eye_idx]
        left_shoulder = keypoints[self.left_shoulder_idx]
        right_shoulder = keypoints[self.right_shoulder_idx]
        
        # 신뢰도가 낮으면 False 반환
        if (left_eye[2] < 0.3 or right_eye[2] < 0.3 or
            left_shoulder[2] < 0.3 or right_shoulder[2] < 0.3):
            return False, "Low confidence in head/shoulder detection"
        if (left_eye[0] == 0 or right_eye[0] == 0 or
            left_shoulder[0] == 0 or right_shoulder[0] == 0):
            return False, "zero head/shoulder detection"
        
        # 어깨와 눈까지의 거리
        left_s2e = abs(left_shoulder[0] - left_eye[0])
        right_s2e = abs(right_shoulder[0] - right_eye[0])

        if( left_s2e == 0 or right_s2e == 0):
            return False, "zero head/shoulder detection"

        d_s2e = max(left_s2e, right_s2e)/min(left_s2e, right_s2e)
        # print(f"d_s2e: {d_s2e}")
        
        is_tilted = d_s2e > self.head_tilt_threshold
        message = f"Head tilt detected ({d_s2e:.1f})" if is_tilted else f"Head aligned({d_s2e:.1f})"
        return is_tilted, message

    def detect_gesture(self, keypoints):
        """손과 눈의 거리를 기반으로 제스처 감지"""
        left_wrist = keypoints[self.left_wrist_idx]
        right_wrist = keypoints[self.right_wrist_idx]
        left_eye = keypoints[self.left_eye_idx]
        right_eye = keypoints[self.right_eye_idx]
        
        # 신뢰도가 낮으면 False 반환
        if (left_wrist[2] < 0.3 or right_wrist[2] < 0.3 or
            left_eye[2] < 0.3 or right_eye[2] < 0.3):
            return False, "Low confidence in hand/eye detection"
        
        # 양손이 교차되었는지 확인 (x좌표 비교)
        if self.eye_dist != 0:
            hands_crossed = (left_wrist[0] < right_wrist[0] and 
                        abs(left_wrist[1] - right_wrist[1])/self.eye_dist > self.hands_cross_threshold)
        else:
            hands_crossed = False
            self.eye_dist = 0.1  # 제로 디비전 방지
        
        # 각 손과 눈 사이의 최소 거리 계산
        left_hand_dist = min(
            np.linalg.norm(left_wrist[:2] - left_eye[:2]),
            np.linalg.norm(left_wrist[:2] - right_eye[:2])
        )
        right_hand_dist = min(
            np.linalg.norm(right_wrist[:2] - left_eye[:2]),
            np.linalg.norm(right_wrist[:2] - right_eye[:2])
        )
        
        # 제스처 감지 조건: 손이 눈 근처에 있고, 손이 교차되지 않은 상태
        gesture_detected = False
        message = "No gesture detected"
        
        if hands_crossed:
            message = "Hands are crossed"
        elif left_hand_dist/self.eye_dist < self.gesture_distance_threshold:
            gesture_detected = True
            message = f"Left hand gesture detected (dist: {left_hand_dist:.1f}px)"
        elif right_hand_dist/self.eye_dist < self.gesture_distance_threshold:
            gesture_detected = True
            message = f"Right hand gesture detected (dist: {right_hand_dist:.1f}px)"
        
        return gesture_detected, message

    def analyze_pose(self, keypoints):
        """모든 분석을 수행하고 결과를 반환"""
        tilt, tilt_msg = self.detect_body_tilt(keypoints)
        head_tilt, head_tilt_msg = self.detect_head_tilt(keypoints)
        wrong_dist, dist_msg = self.detect_distance(keypoints)
        gesture, gesture_msg = self.detect_gesture(keypoints)
        
        return {
            'body_tilt': (tilt, tilt_msg),
            'head_tilt': (head_tilt, head_tilt_msg),
            'wrong_distance': (False, ''),
            'gesture': (gesture, gesture_msg)
        }