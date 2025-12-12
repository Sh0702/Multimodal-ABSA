import argparse
import os
import json
from pathlib import Path
import ffmpeg

def segment_video(input_video, subtitle_json, output_folder):
    with open(subtitle_json, 'r') as f:
        data = json.load(f)
    segments = data.get("segments", [])
    vidname = Path(input_video).stem
    os.makedirs(output_folder, exist_ok=True)
    for seg in segments:
        seg_start = seg['start']
        seg_end = seg['end']
        seg_id = seg['id']
        outpath = os.path.join(output_folder, f"{vidname}_seg{seg_id+1}.mp4")
        (
            ffmpeg
            .input(input_video, ss=seg_start, t=seg_end-seg_start)
            .output(outpath, vcodec='copy', acodec='copy')
            .run(overwrite_output=True, quiet=True)
        )
        print(f"Saved segment: {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Segment a video file using subtitle timestamp json from Whisper.")
    parser.add_argument('--input_video', required=True)
    parser.add_argument('--subtitles_json', required=True)
    parser.add_argument('--output_folder', required=True)
    args = parser.parse_args()
    segment_video(args.input_video, args.subtitles_json, args.output_folder)

if __name__ == '__main__':
    main()
