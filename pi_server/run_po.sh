#!/bin/bash

# python -m scripts.pose_est.pose_est_main --hef models/vit_pose_small.hef \
# --camera '/dev/video0' --conf 0.6 --width 192 --height 256

python -m scripts.pose_est.pose_est_main --hef models/vit_pose_small.hef \
--camera 'rtsp://127.0.0.1:8554/test' --conf 0.4 --width 192 --height 256