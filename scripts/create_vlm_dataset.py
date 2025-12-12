import argparse
import os
import pandas as pd
import cv2
import json
from pathlib import Path

def extract_frame(video_path, out_frame_path):
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(str(out_frame_path), frame)
    cap.release()
    return ret

def main():
    parser = argparse.ArgumentParser(description="Create CSV for VLM training: frame, subtitle, video name.")
    parser.add_argument('--segments_folder', type=str, required=True)
    parser.add_argument('--subtitles_folder', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    rows = []
    for seg_vid in os.listdir(args.segments_folder):
        if not seg_vid.endswith('.mp4'): continue
        seg_path = Path(args.segments_folder) / seg_vid
        # Video name should be everything before "_seg"
        base = seg_vid.split('_seg')[0]
        seg_id = int(seg_vid.split('_seg')[1].replace('.mp4','')) - 1
        # Subtitle JSON must exist
        json_path = Path(args.subtitles_folder) / f"{base}.json"
        if not json_path.exists(): continue
        with open(json_path) as f:
            data = json.load(f)
        # Find segment
        segments = data.get('segments', [])
        if seg_id < 0 or seg_id >= len(segments): continue
        subtitle = segments[seg_id]['text']
        # Extract frame (first frame)
        frame_path = Path(args.segments_folder) / f"{seg_vid.replace('.mp4', '.jpg')}"
        ret = extract_frame(seg_path, frame_path)
        if ret:
            rows.append({
                'frame_path': str(frame_path),
                'subtitle': subtitle,
                'video_name': base
            })
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

if __name__ == '__main__':
    main()
