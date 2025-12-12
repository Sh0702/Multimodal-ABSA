import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess
import os
import re

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.flv'}
SENTENCE_ENDINGS = {'.', '!', '?'}

def find_word_in_context(context_window, word):
    """Check if word appears in context_window (case-insensitive)"""
    if pd.isna(context_window):
        return False
    context_lower = str(context_window).lower()
    word_lower = str(word).lower()
    # Check if word appears as whole word in context
    words_in_context = context_lower.split()
    return word_lower in words_in_context

def ends_with_sentence_punctuation(text):
    """Check if text ends with sentence-ending punctuation"""
    if pd.isna(text) or not text:
        return False
    text_str = str(text).strip()
    if not text_str:
        return False
    # Remove trailing quotes/whitespace and check last character
    text_clean = text_str.rstrip('"\'" \t\n')
    if not text_clean:
        return False
    # Check if ends with sentence punctuation
    return text_clean[-1] in SENTENCE_ENDINGS

def find_sentence_start(df, current_idx):
    """
    Find the start of the current sentence by going backwards
    until we find a row where subtitle_text ends with sentence punctuation.
    """
    if current_idx == 0:
        return 0
    
    # Go backwards to find sentence start
    for i in range(current_idx, -1, -1):
        subtitle_text = df.at[i, 'subtitle_text']
        if ends_with_sentence_punctuation(subtitle_text):
            # Found end of previous sentence, start is next row
            return min(i + 1, current_idx)
    
    # If no sentence end found, start from beginning
    return 0

def find_sentence_end(df, current_idx):
    """
    Find the end of the current sentence by going forwards
    until we find a row where subtitle_text ends with sentence punctuation.
    """
    if current_idx >= len(df) - 1:
        return len(df) - 1
    
    # Go forwards to find sentence end
    for i in range(current_idx, len(df)):
        subtitle_text = df.at[i, 'subtitle_text']
        if ends_with_sentence_punctuation(subtitle_text):
            return i
    
    # If no sentence end found, use last row
    return len(df) - 1

def count_sentences_in_range(df, start_idx, end_idx):
    """Count number of complete sentences in the given range"""
    count = 0
    for i in range(start_idx, min(end_idx + 1, len(df))):
        subtitle_text = df.at[i, 'subtitle_text']
        if ends_with_sentence_punctuation(subtitle_text):
            count += 1
    return count

def find_word_in_group_in_context(context_window, words_group):
    """Check if any word from the group appears in context_window"""
    if pd.isna(context_window):
        return False
    context_lower = str(context_window).lower()
    words_in_context = context_lower.split()
    for word in words_group:
        if word.lower() in words_in_context:
            return True
    return False

def find_padding_before(df, first_labeled_idx, word, words_group=None):
    """
    Find padding frames before the first labeled frame.
    Goes backwards checking context_window until word (or any word in group) no longer appears.
    """
    if first_labeled_idx == 0:
        return 0  # No rows before
    
    padding_start = first_labeled_idx
    
    # Go backwards from first_labeled_idx
    for i in range(first_labeled_idx - 1, -1, -1):
        context_window = df.at[i, 'context_window']
        # Check for the specific word or any word in the group
        if words_group:
            if find_word_in_group_in_context(context_window, words_group):
                padding_start = i
            else:
                break
        else:
            if find_word_in_context(context_window, word):
                padding_start = i
            else:
                break  # Stop when word no longer appears in context
    
    return padding_start

def find_padding_after(df, last_labeled_idx, word, words_group=None):
    """
    Find padding frames after the last labeled frame.
    Goes forwards checking context_window until word (or any word in group) no longer appears.
    """
    if last_labeled_idx >= len(df) - 1:
        return len(df) - 1  # No rows after
    
    padding_end = last_labeled_idx
    
    # Go forwards from last_labeled_idx
    for i in range(last_labeled_idx + 1, len(df)):
        context_window = df.at[i, 'context_window']
        # Check for the specific word or any word in the group
        if words_group:
            if find_word_in_group_in_context(context_window, words_group):
                padding_end = i
            else:
                break
        else:
            if find_word_in_context(context_window, word):
                padding_end = i
            else:
                break  # Stop when word no longer appears in context
    
    return padding_end

def normalize_label(label_value):
    """
    Normalize label value to integer (0 or 1) or None.
    Handles strings, integers, floats, NaN, empty strings.
    """
    if pd.isna(label_value):
        return None
    
    # Handle numeric types directly (int, float)
    if isinstance(label_value, (int, float)):
        if label_value == 0 or label_value == 0.0:
            return 0
        elif label_value == 1 or label_value == 1.0:
            return 1
        else:
            return None
    
    # Convert to string and strip whitespace
    label_str = str(label_value).strip().lower()
    
    # Check for various representations
    if label_str in ['0', '0.0', 'negative', 'neg']:
        return 0
    elif label_str in ['1', '1.0', 'positive', 'pos']:
        return 1
    
    # Try to convert to float then int (handles "0.0", "1.0" strings)
    try:
        label_float = float(label_str)
        label_int = int(label_float)
        if label_int in [0, 1] and label_float == label_int:
            return label_int
    except (ValueError, TypeError):
        pass
    
    return None

def segment_video_frames(excel_path):
    """
    Process a single Excel file and find frame segments for labeled words.
    Returns list of segments with (word, start_frame, end_frame, label) tuples.
    """
    try:
        # Load Excel file
        df = pd.read_excel(excel_path)
        
        # Check required columns
        required_cols = ['frame_number', 'word_spoken', 'context_window', 'label', 'subtitle_text']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing required columns in {excel_path.name}: {missing_cols}")
            return []
        
        # Normalize labels: convert to 0, 1, or None
        df['label_normalized'] = df['label'].apply(normalize_label)
        
        # Find rows with valid labels (0 or 1)
        labeled_rows = df[df['label_normalized'].isin([0, 1])]
        
        if labeled_rows.empty:
            return []
        
        # Find consecutive labeled words and group them together
        # Get all labeled rows with their indices
        labeled_indices = labeled_rows.index.tolist()
        
        if not labeled_indices:
            return []
        
        # Group consecutive labeled words
        # A word is "consecutive" if it appears within a reasonable frame distance
        # We'll consider words consecutive if they're within 30 frames (1 second at 30fps) of each other
        MAX_FRAME_GAP = 30
        
        word_groups = []
        current_group = [labeled_indices[0]]
        
        for i in range(1, len(labeled_indices)):
            prev_idx = labeled_indices[i-1]
            curr_idx = labeled_indices[i]
            
            # Check if current word is close to previous word
            prev_frame = df.at[prev_idx, 'frame_number']
            curr_frame = df.at[curr_idx, 'frame_number']
            frame_gap = curr_frame - prev_frame
            
            if frame_gap <= MAX_FRAME_GAP:
                # Consecutive - add to current group
                current_group.append(curr_idx)
            else:
                # Not consecutive - save current group and start new one
                word_groups.append(current_group)
                current_group = [curr_idx]
        
        # Add the last group
        if current_group:
            word_groups.append(current_group)
        
        # Process the first group (or all groups if you want multiple segments)
        # For now, we'll process the first group as requested
        if not word_groups:
            return []
        
        first_group = word_groups[0]
        first_word_idx = min(first_group)
        last_word_idx = max(first_group)
        
        # Get all words in this group for padding calculation
        words_in_group = set()
        for idx in first_group:
            word = str(df.at[idx, 'word_spoken']).lower()
            words_in_group.add(word)
        
        # Find padding frames (based on context window)
        # For padding before: check context_window for the first word and any word in group
        first_word = str(df.at[first_word_idx, 'word_spoken']).lower()
        padding_start = find_padding_before(df, first_word_idx, first_word, words_in_group)
        
        # For padding after: check context_window for the last word and any word in group
        last_word = str(df.at[last_word_idx, 'word_spoken']).lower()
        padding_end = find_padding_after(df, last_word_idx, last_word, words_in_group)
        
        # Extend to sentence boundaries
        sentence_start = find_sentence_start(df, padding_start)
        sentence_end = find_sentence_end(df, padding_end)
        
        # Count sentences in the range
        sentence_count = count_sentences_in_range(df, sentence_start, sentence_end)
        
        # Limit to 1-2 sentences
        if sentence_count > 2:
            # Find the second sentence end
            sentence_ends_found = 0
            for i in range(sentence_start, min(sentence_end + 1, len(df))):
                subtitle_text = df.at[i, 'subtitle_text']
                if ends_with_sentence_punctuation(subtitle_text):
                    sentence_ends_found += 1
                    if sentence_ends_found == 2:
                        sentence_end = i
                        break
        
        # Ensure we have at least 1 sentence (if possible)
        if sentence_count == 0:
            # No sentence endings found, use original padding
            # But try to extend to nearest sentence boundaries if they exist
            final_start = sentence_start
            final_end = sentence_end
        else:
            final_start = sentence_start
            final_end = sentence_end
        
        # Get frame numbers
        start_frame = int(df.at[final_start, 'frame_number'])
        end_frame = int(df.at[final_end, 'frame_number'])
        
        # Get all words in the group (preserve original case from first occurrence)
        words_list = []
        labels_list = []
        for idx in sorted(first_group):
            word_original = str(df.at[idx, 'word_spoken'])
            label_val = int(df.at[idx, 'label_normalized'])
            words_list.append(word_original)
            labels_list.append(label_val)
        
        # Use first word as primary, but include all words
        primary_word = words_list[0]
        primary_label = labels_list[0]
        
        segment = {
            'word': primary_word,  # Primary word (first in group)
            'words': ', '.join(words_list),  # All words in group
            'start_frame': start_frame,
            'end_frame': end_frame,
            'label': primary_label,  # Primary label (first in group)
            'num_frames': end_frame - start_frame + 1,
            'sentence_count': min(sentence_count, 2),  # Track sentence count
            'num_words': len(words_list)  # Number of words in segment
        }
        
        return [segment]
        
    except Exception as e:
        print(f"❌ Error processing {excel_path}: {e}")
        return []

def find_matching_video(excel_path, videos_folder, subtitles_base):
    """
    Find matching video file for an Excel subtitle file.
    Matches by label (alm/blm), platform folder, and video name.
    Structure: subtitles_base/label/platform/video_name.xlsx
              videos_folder/label/platform/video_name.mp4
    """
    excel_path = Path(excel_path)
    videos_path = Path(videos_folder)
    subtitles_base = Path(subtitles_base)
    
    # Get relative path from subtitles base
    try:
        rel_path = excel_path.relative_to(subtitles_base)
        parts = rel_path.parts
        
        # Expected structure: label/platform/video_name.xlsx
        if len(parts) < 3:
            return None
        
        label = parts[0]  # alm or blm
        platform = parts[1]  # TikTok or YouTube
        video_name = excel_path.stem  # Video name without extension
        
        # Construct video path: videos_folder/label/platform/video_name.ext
        platform_video_path = videos_path / label / platform
        
        if not platform_video_path.exists():
            return None
        
        # Try different video extensions
        for ext in VIDEO_EXTS:
            video_file = platform_video_path / f"{video_name}{ext}"
            if video_file.exists():
                return video_file
        
        return None
        
    except ValueError:
        # Excel file not relative to subtitles_base
        return None

def frames_to_timestamp(start_frame, end_frame, fps):
    """Convert frame numbers to start time and duration for ffmpeg"""
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps
    return start_time, duration

def extract_video_segment(video_path, start_frame, end_frame, output_path, fps=30):
    """
    Extract video segment using ffmpeg based on frame numbers.
    Ensures audio is preserved in the output .mp4 file.
    """
    try:
        start_time, duration = frames_to_timestamp(start_frame, end_frame, fps)
        
        cmd = [
            'ffmpeg',
            '-i', str(video_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'copy',  # Copy video stream without re-encoding
            '-c:a', 'copy',  # Copy audio stream without re-encoding (preserves audio)
            '-avoid_negative_ts', 'make_zero',
            '-y',  # Overwrite output file
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            print(f"❌ Error extracting segment: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting segment: {e}")
        return False

def process_all_subtitles(subtitles_folder, videos_folder, output_folder, fps=30):
    """
    Process all subtitle Excel files, find segments, and extract video segments.
    """
    subtitles_path = Path(subtitles_folder)
    videos_path = Path(videos_folder)
    output_path = Path(output_folder)
    
    if not subtitles_path.exists():
        print(f"❌ Subtitles folder does not exist: {subtitles_path}")
        return
    
    if not videos_path.exists():
        print(f"❌ Videos folder does not exist: {videos_path}")
        return
    
    # Find all Excel files
    excel_files = list(subtitles_path.rglob("*.xlsx")) + list(subtitles_path.rglob("*.xls"))
    
    if not excel_files:
        print(f"⚠️  No Excel files found in: {subtitles_path}")
        return
    
    print(f"📂 Found {len(excel_files)} subtitle Excel file(s)")
    
    # Process each Excel file
    all_segments = []
    extracted_count = 0
    no_labels_count = 0
    no_video_count = 0
    
    for excel_file in excel_files:
        print(f"\n🎬 Processing: {excel_file.name}")
        
        # Find matching video file
        video_file = find_matching_video(excel_file, videos_path, subtitles_path)
        
        if not video_file:
            print(f"   ⚠️  No matching video found")
            no_video_count += 1
            continue
        
        print(f"   📹 Video: {video_file.name}")
        
        # Find frame segments
        segments = segment_video_frames(excel_file)
        
        if not segments:
            print(f"   ⚠️  No segments found (no labeled words)")
            no_labels_count += 1
            continue
        
        print(f"   📊 Found {len(segments)} segment(s)")
        
        # Extract label, platform and video name for output structure
        # Expected: label/platform/video_name.xlsx
        excel_parts = excel_file.relative_to(subtitles_path).parts
        if len(excel_parts) < 3:
            print(f"   ⚠️  Invalid path structure, skipping")
            continue
        
        label = excel_parts[0]  # alm or blm
        platform = excel_parts[1]  # TikTok or YouTube
        video_name = excel_file.stem
        
        # Create output folder: output_folder/label/platform/video_name/
        video_output_dir = output_path / label / platform / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract each segment
        segments_saved = 0
        for seg_idx, segment in enumerate(segments, start=1):
            segment_filename = f"{video_name}_p{seg_idx}.mp4"
            segment_output = video_output_dir / segment_filename
            
            print(f"   🔄 Extracting segment {seg_idx}/{len(segments)}: {segment_filename} (frames {segment['start_frame']}-{segment['end_frame']})")
            
            # Extract video segment
            if extract_video_segment(
                video_file,
                segment['start_frame'],
                segment['end_frame'],
                segment_output,
                fps
            ):
                extracted_count += 1
                segments_saved += 1
                segment['video_file'] = video_name
                segment['label_folder'] = label  # alm or blm
                segment['platform'] = platform
                segment['segment_file'] = str(segment_output)
                segment['excel_path'] = str(excel_file)
                all_segments.append(segment)
                print(f"   ✅ Saved: {segment_output.name}")
            else:
                print(f"   ❌ Failed to extract segment {seg_idx}")
        
        print(f"   ✅ Completed: {segments_saved}/{len(segments)} segments saved")
    
    # Create summary DataFrame
    if all_segments:
        df_segments = pd.DataFrame(all_segments)
        
        # Reorder columns
        column_order = ['label_folder', 'platform', 'video_file', 'word', 'start_frame', 'end_frame', 
                       'num_frames', 'label', 'segment_file', 'excel_path']
        df_segments = df_segments[[col for col in column_order if col in df_segments.columns]]
        
        # Save summary CSV
        summary_csv = output_path / 'segments_summary.csv'
        df_segments.to_csv(summary_csv, index=False)
        
        print(f"\n✅ Processing complete!")
        print(f"   Total segments extracted: {extracted_count}")
        print(f"\n📊 Summary:")
        print(f"   Label 0 (negative): {len(df_segments[df_segments['label'] == 0])}")
        print(f"   Label 1 (positive): {len(df_segments[df_segments['label'] == 1])}")
        print(f"\n📈 File Statistics:")
        print(f"   Files processed: {len(excel_files)}")
        print(f"   Files with no labels: {no_labels_count}")
        print(f"   Files with no matching video: {no_video_count}")
        print(f"   Files successfully processed: {len(excel_files) - no_labels_count - no_video_count}")
        print(f"\n💾 Summary saved to: {summary_csv}")
        print(f"📁 Segments saved to: {output_path}")
        
        return df_segments
    else:
        print(f"\n⚠️  No segments extracted")
        print(f"\n📈 File Statistics:")
        print(f"   Files processed: {len(excel_files)}")
        print(f"   Files with no labels: {no_labels_count}")
        print(f"   Files with no matching video: {no_video_count}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Segment videos based on labeled words in subtitle Excel files."
    )
    parser.add_argument(
        '--subtitles_folder',
        type=str,
        required=True,
        help='Path to folder containing annotated subtitle Excel files'
    )
    parser.add_argument(
        '--videos_folder',
        type=str,
        required=True,
        help='Path to folder containing input video files (same structure as subtitles)'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='Path to output folder where video segments will be saved'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Video frame rate (default: 30)'
    )
    args = parser.parse_args()
    
    print(f"📂 Subtitles folder: {args.subtitles_folder}")
    print(f"📂 Videos folder: {args.videos_folder}")
    print(f"📂 Output folder: {args.output_folder}")
    print(f"🎬 Frame rate: {args.fps} fps")
    
    process_all_subtitles(args.subtitles_folder, args.videos_folder, args.output_folder, args.fps)

if __name__ == '__main__':
    main()
