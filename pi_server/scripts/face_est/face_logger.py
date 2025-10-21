import xml.etree.ElementTree as ET
import datetime
import os
import numpy as np

class PoseDataLogger:
    def __init__(self, save_dir=None, fps=15, cooldown_seconds=2.0):
        """
        얼굴 데이터를 XML 형식으로 저장하는 클래스
        
        Args:
            save_dir: 로그 파일을 저장할 디렉토리 경로
            fps: 영상의 FPS (초당 프레임 수)
            cooldown_seconds: 같은 분석 결과에 대한 재저장 대기 시간 (초)
        """
        # 기본 저장 경로 설정
        if save_dir is None:
            # 스크립트의 현재 디렉토리 기준으로 절대 경로 계산
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            save_dir = os.path.join(project_root, "srv_tmp", "xml")
        self.save_dir = save_dir
        self.fps = fps
        self.cooldown_seconds = cooldown_seconds
        self.frame_counter = 0
        self.start_time = None
        self.last_detection_time = {}  # 각 분석 유형별 마지막 감지 시간
        self.emotion_names = ['angry', 'disgust', 'happy', 'neutral', 'surprise']
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 현재 세션의 XML 파일 생성
        self.session_start = datetime.datetime.now()
        self.filename = os.path.join(
            save_dir, 
            f"face_log_{self.session_start.strftime('%Y%m%d_%H%M%S')}.xml"
        )
        
        # XML 루트 엘리먼트 생성
        self.root = ET.Element("face_session")
        self.root.set("start_time", self.session_start.isoformat())
        
        # XML 트리 저장
        self._save_xml()
        print(f"Created new face log file: {self.filename}")
    
    def log_analysis(self, analysis_results):
        """
        얼굴 분석 결과만을 XML에 추가
        분석 결과가 감지된 경우에만 저장
        
        Args:
            analysis_results: 포즈 분석 결과 딕셔너리
        """
        # 분석 결과가 없으면 무시
        if not analysis_results:
            return
            
        # 첫 프레임에서 시작 시간 설정
        if self.start_time is None:
            self.start_time = datetime.datetime.now()
        
        self.frame_counter += 1
        
        # 현재 시간 계산
        elapsed_seconds = 0.0 + (self.frame_counter / self.fps)
        current_time = self.start_time + datetime.timedelta(seconds=elapsed_seconds)
        
        # 감지된 분석 결과가 있는지 확인
        has_detection = any(is_detected for is_detected, _ in analysis_results.values())
        
        # 현재 시간
        now = datetime.datetime.now()
        
        # 기록할 결과가 있는지 확인
        results_to_log = {}
        for key, (is_detected, message) in analysis_results.items():
            if not is_detected:
                continue
                
            # 마지막 감지 시간 확인
            last_time = self.last_detection_time.get(key)
            if last_time is None or (now - last_time).total_seconds() >= self.cooldown_seconds:
                results_to_log[key] = (is_detected, message)
                self.last_detection_time[key] = now
        
        # 기록할 결과가 있을 때만 저장
        if results_to_log:
            # 현재 시간의 프레임 엘리먼트 생성
            frame = ET.SubElement(self.root, "frame")
            frame.set("timestamp", current_time.isoformat())
            frame.set("elapsed_seconds", f"{elapsed_seconds:.1f}")
            
            # 분석 결과 저장
            analysis = ET.SubElement(frame, "analysis")
            for key, (is_detected, message) in results_to_log.items():
                result = ET.SubElement(analysis, "result")
                result.set("type", key)
                result.set("message", message)
        
            # XML 파일 업데이트
            self._save_xml()
    
    def _save_xml(self):
        """XML 파일 저장"""
        tree = ET.ElementTree(self.root)
        tree.write(self.filename, encoding="utf-8", xml_declaration=True)
        tree.write(os.path.join(os.path.dirname(self.filename), "f_log.xml"), encoding="utf-8", xml_declaration=True)
    
    def close(self):
        """세션 종료 및 최종 저장"""
        session_end = datetime.datetime.now()
        self.root.set("end_time", session_end.isoformat())
        duration = (session_end - self.session_start).total_seconds()
        self.root.set("duration_seconds", f"{duration:.1f}")
        self._save_xml()
        print(f"\nface log saved to: {self.filename}")
        print(f"Session duration: {duration:.1f} seconds")


def create_pose_logger(fps=15):
    """PoseDataLogger 인스턴스 생성"""
    return PoseDataLogger(fps=fps)


# 사용 예시:
if __name__ == "__main__":
    # 테스트용 더미 데이터
    logger = create_pose_logger()
    
    # 더미 keypoints 생성
    dummy_keypoints = np.random.rand(17, 3)
    dummy_keypoints[:, 2] = 0.9  # confidence 값
    
    # 더미 분석 결과
    dummy_analysis = {
        'body_tilt': (True, "Body asymmetry detected (1.2, 1.1)"),
        'head_tilt': (True, "Head tilt detected (3.8)"),
        'wrong_distance': (False, "Good distance (diff: 5.2%)"),
        'gesture': (True, "Left hand gesture detected (dist: 4.5px)")
    }
    
    # 데이터 로깅
    logger.log_analysis(dummy_analysis)
    
    # 세션 종료
    logger.close()