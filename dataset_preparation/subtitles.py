import argparse
import os
from pathlib import Path
import whisper
import pandas as pd
from tqdm import tqdm

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.wav', '.m4a', '.mp3'}

def seconds_to_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int((secs - int(secs)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{millis:03d}"

def collect_video_files(root):
    """Recursively collect all video/audio files"""
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if Path(fname).suffix.lower() in VIDEO_EXTS:
                all_files.append(Path(dirpath) / fname)
    return all_files

def sliding_windows(words, window_size=5):
    """Create sliding windows of words"""
    if len(words) < window_size:
        return []
    return [words[i:i+window_size] for i in range(len(words)-window_size+1)]

def process_video_to_excel(video_path, model, fps, output_path):
    """Transcribe video and create frame-by-frame Excel with required columns"""
    print(f"🎬 Processing: {video_path.name}")
    
    # Transcribe video
    result = model.transcribe(str(video_path), verbose=False)
    segments = result.get("segments", [])
    
    if not segments:
        print(f"⚠️  No subtitles found in {video_path.name}")
        return
    
    # Flatten all words with their timestamps
    all_words = []
    word_times = []
    segment_texts = []  # Store which segment each word belongs to
    
    for seg in segments:
        if "text" not in seg or "start" not in seg or "end" not in seg:
            continue
        
        seg_text = seg["text"].strip()
        seg_start = seg["start"]
        seg_end = seg["end"]
        
        if not seg_text:
            continue
        
        # Split segment into words and assign timestamps
        words_in_seg = seg_text.split()
        num_words = len(words_in_seg)
        if num_words == 0:
            continue
        
        # Distribute time evenly across words in segment
        time_per_word = (seg_end - seg_start) / num_words
        
        for i, word in enumerate(words_in_seg):
            word_start = seg_start + (i * time_per_word)
            word_end = seg_start + ((i + 1) * time_per_word)
            all_words.append(word)
            word_times.append((word_start, word_end))
            segment_texts.append(seg_text)  # Store full segment text for this word
    
    if not all_words:
        print(f"⚠️  No words extracted from {video_path.name}")
        return
    
    # Create sliding windows and generate frame-by-frame rows
    rows = []
    windows = sliding_windows(all_words, 5)
    
    for i, window in enumerate(windows):
        center_idx = 2  # Center word in 5-word window
        word_spoken = window[center_idx]
        
        # Get the word's timestamp
        word_idx = i + center_idx
        if word_idx >= len(word_times):
            continue
        
        word_start, word_end = word_times[word_idx]
        context_window = ' '.join(window)
        subtitle_text = segment_texts[word_idx]  # Full segment text for this word
        
        # Generate rows for each frame in this word's time range
        start_frame = int(word_start * fps)
        end_frame = int(word_end * fps)
        
        for frame_number in range(start_frame, end_frame + 1):
            timestamp = seconds_to_timestamp(frame_number / fps)
            rows.append({
                "frame_number": frame_number,
                "timestamp": timestamp,
                "subtitle_text": subtitle_text,
                "word_spoken": word_spoken,
                "context_window": context_window,
                "label": ""  # Empty label column
            })
    
    if not rows:
        print(f"⚠️  No rows generated for {video_path.name}")
        return
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"✅ Saved: {output_path} ({len(rows)} rows)")

def main():
    parser = argparse.ArgumentParser(
        description="Extract subtitles from videos frame-by-frame using Whisper and save as Excel."
    )
    parser.add_argument(
        '--input_videos',
        type=str,
        required=True,
        help='Path to folder containing video/audio files (recursively searched)'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Path to output folder where Excel files will be saved'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video frame rate (default: 30)'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='medium',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: medium)'
    )
    args = parser.parse_args()
    
    input_path = Path(args.input_videos)
    output_path = Path(args.output_folder)
    
    if not input_path.exists():
        print(f"❌ Input folder does not exist: {input_path}")
        return
    
    # Collect all video files
    print(f"🔍 Searching for video files in: {input_path}")
    video_files = collect_video_files(input_path)
    
    if not video_files:
        print(f"⚠️  No video/audio files found in: {input_path}")
        print(f"   Supported formats: {', '.join(VIDEO_EXTS)}")
        return
    
    print(f"📂 Found {len(video_files)} video file(s)")
    
    # Load Whisper model
    print(f"🤖 Loading Whisper model: {args.model_size}")
    model = whisper.load_model(args.model_size)
    print("✅ Model loaded")
    
    # Process each video
    output_path.mkdir(parents=True, exist_ok=True)
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        try:
            # Create output path preserving directory structure
            relpath = video_file.relative_to(input_path)
            out_xlsx = output_path / relpath.with_suffix('.xlsx')
            
            process_video_to_excel(video_file, model, args.fps, out_xlsx)
        except Exception as e:
            print(f"❌ Error processing {video_file}: {e}")
            continue
    
    print(f"\n✅ All videos processed! Output saved to: {output_path}")

if __name__ == '__main__':
    main()
