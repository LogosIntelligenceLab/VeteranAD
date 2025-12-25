#!/bin/bash
# killall -9 -r CarlaUE4-Linux
GPU_ID=$1
echo 'cleaning...'
ps -ef | grep "graphicsadapter=${GPU_ID}" | awk '{print $2}' | xargs kill > /dev/null 2>&1 &
ps -ef | grep "gpu-rank=${GPU_ID}" | awk '{print $2}' | xargs kill > /dev/null 2>&1 &
ps -ef | grep "Carla" | awk '{print $2}' | xargs kill > /dev/null 2>&1 &
