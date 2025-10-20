import gi
import numpy as np
import threading
import time
import signal
import queue
import cv2

# Require specific GStreamer and GstApp versions
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GLib

# --- Settings ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_CHANNELS = 3

# --- Queue for inter-thread communication ---
# Set maxsize to avoid unbounded memory usage
frame_queue = queue.Queue(maxsize=30)
keep_running = True # Flag to control the main loop

def on_new_sample(appsink):
    """
    [Producer] GStreamer appsink: push new frames into the Queue.
    (Runs in the background thread)
    """
    sample = appsink.emit('pull-sample')
    if not sample:
        print("No sample received from appsink.")
        return Gst.FlowReturn.OK

    buf = sample.get_buffer()
    frame_data = buf.extract_dup(0, buf.get_size())
    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(FRAME_HEIGHT, FRAME_WIDTH, -1)

    try:
        # Use non-blocking put to avoid blocking; drop frame if queue is full
        frame_queue.put_nowait(frame)
    except queue.Full:
        # If the queue is full, drop the current frame to preserve real-time behavior
        pass
            
    return Gst.FlowReturn.OK

def start_pipeline():
    """Set up the GStreamer pipeline and run the main loop in a background thread."""
    Gst.init(None)
    # 시스템에 따라 v4l2src 대신 ksvideosrc (Windows), avfvideosrc (macOS) 사용
    pipeline_str = (
        f"v4l2src device=/dev/video0 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR,width={FRAME_WIDTH},height={FRAME_HEIGHT},framerate=30/1 ! "
        f"appsink name=sink emit-signals=true max-buffers=1 drop=true"
    )
    
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name('sink')
    appsink.connect('new-sample', on_new_sample)
    
    pipeline.set_state(Gst.State.PLAYING)
    print("GStreamer pipeline started.")

    loop = GLib.MainLoop()
    # daemon=True: thread will exit when the main program exits
    loop_thread = threading.Thread(target=loop.run, daemon=True)
    loop_thread.start()

    return pipeline, loop

def main():
    """
    [Consumer] Main function: take frames from the Queue and display via OpenCV.
    """
    # 1. GStreamer 파이프라인 시작
    pipeline, loop = start_pipeline()
    
    # 2. 메인 스레드에서 Queue를 확인하며 GUI 처리
    try:
        while keep_running:
            try:
                # Get a frame from the queue (use timeout to avoid infinite blocking)
                frame = frame_queue.get(timeout=1)
                
                # OpenCV 창에 프레임 표시
                cv2.imshow('Frame from Queue', frame)

                # Press 'q' to exit the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except queue.Empty:
                # If no frame received within 1 second, print waiting message
                print("Waiting for frames...")
                continue

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting.")
    finally:
        # 3. Cleanup resources
        print("Starting cleanup...")
        if loop.is_running():
            loop.quit()
        
        pipeline.set_state(Gst.State.NULL)
        print("GStreamer pipeline stopped.")
        
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")

def signal_handler(sig, frame):
    """Handle Ctrl+C signal to safely stop the main loop."""
    global keep_running
    keep_running = False

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    # X11을 사용하도록 환경 변수 설정 (Wayland 에러 방지용)
    # 이 부분은 스크립트 실행 전 터미널에서 export QT_QPA_PLATFORM=xcb 로 설정하는 것이 더 일반적입니다.
    # import os
    # os.environ['QT_QPA_PLATFORM'] = 'xcb'
    main()