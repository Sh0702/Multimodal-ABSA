import argparse
import os
from pathlib import Path
import json
import pandas as pd
from openpyxl import Workbook

def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{millis:03d}"

def parse_json_transcript(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("segments", [])

def get_segments_for_frame(segments, frame_time):
    for seg in segments:
        if seg["start"] <= frame_time < seg["end"]:
            return seg["text"], seg
    return "", None

def sliding_windows(words, window_size=5):
    if len(words) < window_size:
        return []
    return [words[i:i+window_size] for i in range(len(words)-window_size+1)]

def text_with_timestamp(json_path, fps, out_xlsx):
    segments = parse_json_transcript(json_path)
    # Flatten all segment words with (start, end, text)
    all_words = []
    word_times = []
    for seg in segments:
        seg_text = seg["text"].strip()
        seg_start = seg["start"]
        seg_end = seg["end"]
        for word in seg_text.split():
            all_words.append(word)
            word_times.append((seg_start, seg_end))

    # Sliding window
    rows = []
    for i, window in enumerate(sliding_windows(all_words, 5)):
        center = 2  # 5 words, the center is at index 2
        word = window[center]
        word_start, word_end = word_times[i+center]
        context_window = ' '.join(window)
        # For each frame inside this word's timestamp, output row
        start_frame = int(word_start * fps)
        end_frame = int(word_end * fps)
        for frame_number in range(start_frame, end_frame + 1):
            timestamp = seconds_to_timestamp(frame_number / fps)
            rows.append({
                "Frame_number": frame_number,
                "timestamp": timestamp,
                "subtitle_text": context_window,  # can also include segment text if desired
                "context_window": context_window,
                "word": word,
                "label": ""
            })
    df = pd.DataFrame(rows)
    df.to_excel(out_xlsx, index=False)
    print(f"✅ Wrote frame-by-frame context with windows: {out_xlsx}")

def text_only(json_path, output_txt_path):
    segments = parse_json_transcript(json_path)
    full_text = " ".join(seg["text"] for seg in segments)
    with open(output_txt_path, "w") as f:
        f.write(full_text.strip() + "\n")
    print(f"✅ Wrote plain text: {output_txt_path}")

def batch_text_with_timestamp(input_dir, output_dir, fps):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for json_file in input_dir.rglob("*.json"):
        relpath = json_file.relative_to(input_dir)
        out_xlsx = output_dir / relpath.with_suffix(".xlsx")
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        text_with_timestamp(json_file, fps=fps, out_xlsx=out_xlsx)

def batch_text_only(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for json_file in input_dir.rglob("*.json"):
        relpath = json_file.relative_to(input_dir)
        output_txt = output_dir / relpath.with_suffix(".txt")
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        text_only(json_file, output_txt)

def main():
    parser = argparse.ArgumentParser(description="Subtitle extraction & text export utilities.")
    parser.add_argument('--input_subtitles', type=str, required=True, help='Folder containing input Whisper JSONs')
    parser.add_argument('--output_folder', type=str, required=True, help='Where to write outputs')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate (default: 30)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--textOnly', action='store_true', help='Export only plain text for each subtitle')
    group.add_argument('--textWithTimeStamp', action='store_true', help='Export per-frame context window excel for each subtitle')
    args = parser.parse_args()

    if args.textOnly:
        batch_text_only(args.input_subtitles, args.output_folder)
    elif args.textWithTimeStamp:
        batch_text_with_timestamp(args.input_subtitles, args.output_folder, args.fps)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
