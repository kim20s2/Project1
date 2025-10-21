#!/bin/bash

NAME=$1

ffmpeg -i $NAME.MP4 -vf "scale=640x480" -r 30 -c:a libopus target.MP4
