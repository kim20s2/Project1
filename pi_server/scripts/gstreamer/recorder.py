import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys

Gst.init(None)

# 파이프라인 정의
pipeline_str = (
    "rtspsrc location=rtsp://10.10.14.80:8554/test name=r "
    "! rtpbin name=rtpbin "
    "mp4mux name=mux faststart=true ! filesink location=srv_tmp/mp4/video.mp4 "
    "r. ! queue max-size-buffers=4096 ! rtpbin.recv_rtp_sink_0 "
    "rtpbin. ! rtph264depay ! h264parse ! mux.video_0 "
    "r. ! queue max-size-buffers=4096 ! rtpbin.recv_rtp_sink_1 "
    "rtpbin. ! rtpopusdepay ! opusdec ! audioconvert ! audioresample ! tee name=t "
    "t. ! queue ! wavenc ! filesink location=srv_tmp/wav/audio.wav "
    "t. ! queue ! opusenc ! mux.audio_0"
)

# 파이프라인 생성
pipeline = Gst.parse_launch(pipeline_str)

# 버스 메시지 처리
bus = pipeline.get_bus()
bus.add_signal_watch()

def on_message(bus, message):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("recording finished.")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"error: {err}, {debug}")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

bus.connect("message", on_message)

# 녹화 시작
print("recording... (Ctrl+C to stop)")
pipeline.set_state(Gst.State.PLAYING)

loop = GLib.MainLoop()
try:
    loop.run()
except KeyboardInterrupt:
    print("\nstopping recording...")
    pipeline.send_event(Gst.Event.new_eos())
    loop.run()