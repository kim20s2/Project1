import gi

# GStreamer 및 RTSP 서버 관련 라이브러리 로드
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# GStreamer 초기화
Gst.init(None)

class SensorFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, **properties):
        super(SensorFactory, self).__init__(**properties)
        self.set_shared(True) # 여러 클라이언트가 동일한 스트림을 보도록 설정

    def do_create_element(self, url):
        # GStreamer 파이프라인 설정
        # v4l2src: /dev/video0에서 영상 소스를 가져옴
        # videoconvert: 색 공간 변환
        # x264enc: H.264로 영상을 인코딩 (CPU 사용)
        # rtph264pay: RTSP용 H.264 패킷 생성
        #
        # 라즈베리 파이 하드웨어 인코더 사용시 (더 효율적):
        # "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! v4l2h264enc ! rtph264pay name=pay0 pt=96"
        pipeline_str = ("v4l2src device=/dev/video0 ! "
                        "videoconvert ! "
                        "x264enc tune=zerolatency ! "
                        "rtph264pay name=pay0 pt=96")
        
        print(f"Server Pipeline: {pipeline_str}")
        return Gst.parse_launch(pipeline_str)

class GstServer():
    def __init__(self):
        self.server = GstRtspServer.RTSPServer()
        factory = SensorFactory()

        # 서버에 '/video'라는 주소로 스트림 마운트
        self.server.get_mount_points().add_factory("/video", factory)
        self.server.attach(None)

if __name__ == '__main__':
    server = GstServer()
    
    # 메인 루프 실행 (Ctrl+C로 종료)
    loop = GLib.MainLoop()
    try:
        loop.run()
        print("RTSP Server started at rtsp://10.10.14.80:8554/video")
    except KeyboardInterrupt:
        print("\nStopping server...")
        loop.quit()