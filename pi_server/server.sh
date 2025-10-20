#!/bin/bash

# 스크립트에 전달된 첫 번째 인자(argument)를 TARGET 변수에 저장합니다.
TARGET=$1

export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

# TARGET 변수를 사용하여 파이썬 스크립트를 실행합니다.
python -m scripts.server --target $TARGET