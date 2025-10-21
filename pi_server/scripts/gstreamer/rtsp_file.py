import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

import os

# GStreamer 및 RTSP 서버 초기화
Gst.init(None)

class TestServerFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        
        # 버퍼링 설정
        self.set_latency(0)  # 최소 지연 시간
        self.set_shared(True)  # 여러 클라이언트가 미디어를 공유
        self.set_eos_shutdown(True)  # 정상적인 종료 보장

        self.port = 8554

        script_path = os.path.abspath(__file__)
        main_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))

        target_path = os.path.join(main_dir, 'scripts', 'gstreamer', 'examples', 'target.MP4')

        self.video_device = target_path
        self.max_clients = 5
        self.rotation = 90
        self.fps = 24
        
        # 미디어 설정
        self.set_transport_mode(GstRtspServer.RTSPTransportMode.PLAY)

    # 클라이언트가 접속할 때마다 호출되어 파이프라인을 생성하는 함수
    def do_create_element(self, url):
        rotation_method = {0: 0, 90: 1, 180: 2, 270: 3}
        method_value = rotation_method.get(self.rotation, 0)
        
        # 90도 또는 270도 회전 시 해상도 변경
        if self.rotation in [90, 270]:
            output_width, output_height = 360, 640
        else:
            output_width, output_height = 640, 360

        # 웹캠 영상과 마이크 음성을 함께 인코딩하여 RTP로 보내는 파이프라인
        pipeline_str = (
            "( "
            # 변환된 파일을 읽어 H.264와 Opus를 분리
            f"filesrc location={self.video_device} ! qtdemux name=demux "
            
            # 비디오: h264parse -> rtph264pay (CPU 사용 안함)
            "demux.video_0 ! queue ! h264parse ! "
            "rtph264pay name=pay0 pt=96 config-interval=1 "
            ") "
            
            "( "
            # 오디오: rtpopuspay (CPU 사용 안함)
            "demux.audio_0 ! queue ! "
            "rtpopuspay name=pay1 pt=97 "
            ")"
        )
        print(f"Server AV pipeline: {pipeline_str}")
        return Gst.parse_launch(pipeline_str)

class GstServer():
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        self.server.set_service("8554")
        
        # 서버 설정
        self.server.set_address("0.0.0.0")  # 모든 인터페이스에서 접속 허용
        self.server.set_backlog(20)  # 동시 연결 요청 대기열
        
        # 세션 관리 설정
        session_pool = GstRtspServer.RTSPSessionPool()
        session_pool.set_max_sessions(20)  # 최대 세션 수 설정
        self.server.set_session_pool(session_pool)
        
        # 마운트 포인트 설정
        mounts = self.server.get_mount_points()
        factory = TestServerFactory()
        mounts.add_factory("/test", factory)
        
        # 서버 시작
        self.server.attach(None)
        
        print("RTSP A/V stream ready at rtsp://0.0.0.0:8554/test")
        print("Press Ctrl+C to stop the server.")

if __name__ == '__main__':
    s = GstServer()
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\nStopping server...")
        loop.quit()