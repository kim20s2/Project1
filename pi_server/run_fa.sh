#!/bin/bash

python -m scripts.face_est.face_est_main --camera 'rtsp://127.0.0.1:8554/test' --debug 1
# python -m scripts.face_est.face_est_main --camera './refs/test_videos/test.mp4' --debug 1