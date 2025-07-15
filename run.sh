#!/bin/bash

audio_path=$1
image_path=$2
output_path=$3
effects=$4

python circle_image.py --image "$image_path" --padding 32
python rotate_to_beat.py --image "./media/masked_image.png" --audio "$audio_path" --output "$output_path" --effects $effects
