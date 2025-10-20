import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import cv2
import numpy as np
import threading
import time
import requests
import select
import sys
import os

# Qt backend 설정
os.environ["QT_QPA_PLATFORM"] = "xcb"

Gst.init(None)

SERVER_URL = "http://127.0.0.1:5000"  # 서버 주소

class VideoAudioRecorderApp:
    def __init__(self):
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.recording = False

        self.record_thread = None
        self.video_pipeline = None
        self.audio_pipeline = None
        self.video_src = None

        # "gst-launch-1.0 rtspsrc location=rtsp://127.0.0.1:8554/test ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink"

        # OpenCV 화면용 pipeline
        pipeline_str = (
            "rtspsrc location=rtsp://127.0.0.1:8554/test latency=0 ! "
            "rtph264depay ! "
            "h264parse ! "
            "avdec_h264 max-threads=4 ! "
            "videoconvert n-threads=4 ! "
            "video/x-raw,format=BGR,width=640,height=480 ! "
            "queue max-size-buffers=3 ! "
            "appsink name=opencv_sink emit-signals=true max-buffers=3 drop=true sync=false"
        )
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name("opencv_sink")
        self.appsink.connect("new-sample", self.on_new_sample)
        self.pipeline.set_state(Gst.State.PLAYING)

        # OpenCV 창
        cv2.namedWindow("Camera Preview")

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")
            frame = np.ndarray(
                (height, width, 3),
                buffer=buf.extract_dup(0, buf.get_size()),
                dtype=np.uint8
            )
            with self.frame_lock:
                self.latest_frame = frame
        return Gst.FlowReturn.OK

    def start_recording(self, video_filename="recorded_video.mp4", audio_filename="recorded_audio.wav"):
        if self.recording:
            return
        print(f"Recording started: {video_filename}, {audio_filename}")
        self.recording = True
        self.video_filename = video_filename
        self.audio_filename = audio_filename

        # 비디오 pipeline
        video_pipeline_str = (
            "appsrc name=video_src is-live=true block=true format=time "
            "caps=video/x-raw,format=BGR,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "x264enc bitrate=2000 speed-preset=superfast tune=zerolatency key-int-max=30 ! "
            "mp4mux faststart=true ! filesink location={}".format(video_filename)
        )
        self.video_pipeline = Gst.parse_launch(video_pipeline_str)
        self.video_src = self.video_pipeline.get_by_name("video_src")
        self.video_pipeline.set_state(Gst.State.PLAYING)

        # 오디오 pipeline
        audio_pipeline_str = (
            "alsasrc device=hw:2,0 ! audioconvert ! audioresample ! "
            "wavenc ! filesink location={}".format(audio_filename)
        )
        self.audio_pipeline = Gst.parse_launch(audio_pipeline_str)
        self.audio_pipeline.set_state(Gst.State.PLAYING)

        # 영상 push 스레드
        self.record_thread = threading.Thread(target=self._push_frames, daemon=True)
        self.record_thread.start()

    def stop_recording(self):
        if not self.recording:
            return
        print("Recording stopped.")
        self.recording = False

        if self.record_thread:
            self.record_thread.join()
            self.record_thread = None

        # 영상 EOS 대기
        if self.video_pipeline:
            self.video_src.emit("end-of-stream")
            bus = self.video_pipeline.get_bus()
            while True:
                msg = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                if msg:
                    if msg.type == Gst.MessageType.EOS:
                        print("✅ Video EOS received, finalizing MP4 file.")
                        break
                    elif msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        print("Video Gst Error:", err, dbg)
                        break
            self.video_pipeline.set_state(Gst.State.NULL)
            self.video_pipeline = None
            self.video_src = None

        # 오디오 EOS 대기
        if self.audio_pipeline:
            self.audio_pipeline.send_event(Gst.Event.new_eos())
            bus = self.audio_pipeline.get_bus()
            while True:
                msg = bus.timed_pop_filtered(Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
                if msg:
                    if msg.type == Gst.MessageType.EOS:
                        print("✅ Audio EOS received, finalizing WAV file.")
                        break
                    elif msg.type == Gst.MessageType.ERROR:
                        err, dbg = msg.parse_error()
                        print("Audio Gst Error:", err, dbg)
                        break
            self.audio_pipeline.set_state(Gst.State.NULL)
            self.audio_pipeline = None

        # 서버 전송
        self.send_audio_to_server(self.audio_filename)

    def _push_frames(self):
        frame_count = 0
        last_push_time = time.time()
        target_fps = 30
        frame_interval = 1.0 / target_fps
        
        while self.recording:
            current_time = time.time()
            elapsed = current_time - last_push_time
            
            if elapsed < frame_interval:
                time.sleep(0.001)  # Short sleep to prevent CPU hogging
                continue
                
            with self.frame_lock:
                if self.latest_frame is None:
                    continue
                frame = self.latest_frame.copy()
                
            data = frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, target_fps)
            timestamp = frame_count * buf.duration
            buf.pts = buf.dts = timestamp
            
            if self.video_src:
                self.video_src.emit("push-buffer", buf)
                
            frame_count += 1
            last_push_time = current_time

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording("recorded_video.mp4", "recorded_audio.wav")

    def send_audio_to_server(self, filepath):
        try:
            with open(filepath, "rb") as f:
                files = {'file': f}
                resp = requests.post(SERVER_URL, files=files)
            print("Audio sent to server:", resp.text)
        except Exception as e:
            print("Failed to send audio:", e)

    def run(self):
        print("Camera preview running. Press 'r' to start/stop recording, 'q' to quit.")
        while True:
            # 터미널 입력 확인
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                key = sys.stdin.readline().strip().lower()
                if key == 'r':
                    self.toggle_recording()
                elif key == 'q':
                    if self.recording:
                        self.stop_recording()
                    break

            # 프레임 표시
            with self.frame_lock:
                frame = self.latest_frame
            if frame is not None:
                cv2.imshow("Camera Preview", frame)
                cv2.waitKey(1)

        cv2.destroyAllWindows()
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    app = VideoAudioRecorderApp()
    app.run()