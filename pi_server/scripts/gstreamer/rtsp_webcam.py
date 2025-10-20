import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# GStreamer 및 RTSP 서버 초기화
Gst.init(None)

class TestServerFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        
        # 버퍼링 설정
        self.set_latency(0)  # 최소 지연 시간
        self.set_shared(True)  # 여러 클라이언트가 미디어를 공유
        self.set_eos_shutdown(True)  # 정상적인 종료 보장
        
        # 미디어 설정
        self.set_transport_mode(GstRtspServer.RTSPTransportMode.PLAY)

    # 클라이언트가 접속할 때마다 호출되어 파이프라인을 생성하는 함수
    def do_create_element(self, url):
        # 웹캠 영상과 마이크 음성을 함께 인코딩하여 RTP로 보내는 파이프라인
        pipeline_str = (
            # 메인 파이프라인
            "( "
            # 비디오 파이프라인 - 저사양 CPU에 최적화
            "v4l2src device=/dev/video0 do-timestamp=true ! "
            "videorate drop-only=true max-rate=15 ! "
            "video/x-raw,framerate=15/1,width=640,height=480 ! "
            "videoconvert ! "
            "video/x-raw,format=I420 ! "
            "x264enc tune=zerolatency speed-preset=ultrafast "
            "bitrate=1000 key-int-max=15 threads=2 ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "rtph264pay name=pay0 pt=96 config-interval=1 "
            ") "
            "( "
            # 오디오 파이프라인 - 저사양 환경 최적화
            "pulsesrc do-timestamp=true ! "
            "audio/x-raw,rate=16000,channels=1 ! "
            "audioconvert ! audioresample quality=2 ! "
            "audio/x-raw,rate=16000,channels=1 ! "
            "opusenc bitrate=20000 complexity=0 ! "
            "queue max-size-buffers=2 leaky=downstream ! "
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