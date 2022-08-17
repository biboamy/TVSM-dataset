#!/bin/bash

echo "[INFO] Preparing AVA Speech dataset"

stage=0
mkdir -p "avaspeech/audio/" && cd "$_"

if [ $stage -le 0 ]; then
	echo
	echo "[INFO] Downloading from AWS"
	ava_url="https://s3.amazonaws.com/ava-dataset/trainval/"
	while read p; do
	  echo $p
	  echo $ava_url$p -O;
	  if [ -e "{$ava_url$p}.mp4" ]; then
        echo 'File already exists' >&2
        exit 1
    fi
	  curl "{$ava_url$p}.mkv" -O;
	  curl "{$ava_url$p}.webm" -O;
	  curl "{$ava_url$p}.mp4" -O;
	done 
fi

if [ $stage -le 1 ]; then
	echo
	echo "[INFO] Demux audio from AV files"
	# demux each of the audio streams from the video: mkv, webm, mp4
	for vid in *.mkv; do ffmpeg -i "$vid" -vn -- "${vid%.mkv}.mkv.wav"; done
	for vid in *.webm; do ffmpeg -i "$vid" -vn -- "${vid%.webm}.webm.wav"; done
	for vid in *.mp4; do ffmpeg -i "$vid" -vn -- "${vid%.mp4}.mp4.wav"; done
fi