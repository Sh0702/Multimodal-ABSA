import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

def normalize_platform_name(name):
    """Normalize platform name to lowercase for matching"""
    return name.lower()

def find_matching_annotation(excel_path, annotation_base, subtitles_base):
    """
    Find matching annotation CSV file for a given Excel subtitle file.
    Structure: subtitles/label/platform/video.xlsx -> annotation/platform/label/video.csv
    Returns path to annotation CSV or None if not found.
    """
    excel_path = Path(excel_path)
    annotation_base = Path(annotation_base)
    subtitles_base = Path(subtitles_base)
    
    # Get relative path from subtitles base
    try:
        rel_path = excel_path.relative_to(subtitles_base)
        parts = rel_path.parts
        
        # Expected structure: label/platform/video_name.xlsx
        # Example: alm/TikTok/vid1_tiktok_1.xlsx
        if len(parts) < 3:
            return None
        
        label = parts[0]  # alm or blm
        platform = normalize_platform_name(parts[1])  # TikTok or YouTube (normalized to lowercase)
        video_name = excel_path.stem  # Video name without extension
        
        # Construct annotation path: annotation_base/platform/label/video.csv
        # Example: subtitles_annotation/tiktok/alm/vid1_tiktok_1.csv
        annotation_path = annotation_base / platform / label / f"{video_name}.csv"
        
        if annotation_path.exists():
            return annotation_path
        
        # Debug: show what we're looking for
        # print(f"🔍 Looking for: {annotation_path} (from {excel_path.name}, label={label}, platform={platform})")
        
        return None
        
    except ValueError:
        # Excel file not relative to subtitles_base
        return None

def load_annotation_dict(annotation_csv_path):
    """
    Load annotation CSV and create a dictionary mapping word -> emotion.
    Returns dict with lowercase words as keys.
    """
    try:
        df = pd.read_csv(annotation_csv_path)
        
        # Check required columns
        if 'word' not in df.columns or 'emotion' not in df.columns:
            print(f"⚠️  Annotation file missing required columns (word, emotion): {annotation_csv_path}")
            return {}
        
        # Create dictionary: word (lowercase) -> emotion (lowercase)
        annotation_dict = {}
        for _, row in df.iterrows():
            word = str(row['word']).strip().lower()
            emotion = str(row['emotion']).strip().lower()
            annotation_dict[word] = emotion
        
        return annotation_dict
    except Exception as e:
        print(f"❌ Error loading annotation file {annotation_csv_path}: {e}")
        return {}

def emotion_to_label(emotion):
    """Convert emotion string to label (0 for negative, 1 for positive)"""
    emotion_lower = str(emotion).strip().lower()
    if emotion_lower == 'negative':
        return 0
    elif emotion_lower == 'positive':
        return 1
    return None

def annotate_subtitle_excel(excel_path, annotation_csv_path, output_path=None):
    """
    Annotate subtitle Excel file with labels from annotation CSV.
    Updates the 'label' column based on word_spoken matching.
    """
    try:
        # Load Excel file
        df = pd.read_csv(excel_path) if excel_path.suffix == '.csv' else pd.read_excel(excel_path)
        
        # Check required columns
        if 'word_spoken' not in df.columns:
            print(f"⚠️  Excel file missing 'word_spoken' column: {excel_path}")
            return False
        
        if 'label' not in df.columns:
            df['label'] = ""  # Add label column if missing
        
        # Load annotation dictionary
        annotation_dict = load_annotation_dict(annotation_csv_path)
        
        if not annotation_dict:
            print(f"⚠️  No annotations loaded from: {annotation_csv_path}")
            return False
        
        # Annotate labels
        labels_updated = 0
        for idx, row in df.iterrows():
            word_spoken = str(row['word_spoken']).strip().lower()
            
            # Check if word matches any annotation
            if word_spoken in annotation_dict:
                emotion = annotation_dict[word_spoken]
                label = emotion_to_label(emotion)
                if label is not None:
                    df.at[idx, 'label'] = label
                    labels_updated += 1
        
        # Save updated Excel
        output_path = output_path or excel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_excel(output_path, index=False)
        print(f"✅ Annotated {excel_path.name}: {labels_updated} labels updated")
        return True
        
    except Exception as e:
        print(f"❌ Error processing {excel_path}: {e}")
        return False

def process_all_subtitles(subtitles_folder, annotation_folder, output_folder=None):
    """
    Process all subtitle Excel files and annotate them with labels from annotation CSVs.
    """
    subtitles_path = Path(subtitles_folder)
    annotation_path = Path(annotation_folder)
    
    if not subtitles_path.exists():
        print(f"❌ Subtitles folder does not exist: {subtitles_path}")
        return
    
    if not annotation_path.exists():
        print(f"❌ Annotation folder does not exist: {annotation_path}")
        return
    
    # Find all Excel files in subtitles folder
    excel_files = list(subtitles_path.rglob("*.xlsx")) + list(subtitles_path.rglob("*.xls"))
    
    if not excel_files:
        print(f"⚠️  No Excel files found in: {subtitles_path}")
        return
    
    print(f"📂 Found {len(excel_files)} subtitle Excel file(s)")
    
    # Process each Excel file
    annotated_count = 0
    skipped_count = 0
    instagram_skipped = 0
    
    for excel_file in tqdm(excel_files, desc="Annotating subtitles"):
        # Check if it's in instagram folder (which we ignore)
        if 'instagram' in str(excel_file).lower():
            instagram_skipped += 1
            continue
        
        # Find matching annotation CSV
        annotation_csv = find_matching_annotation(excel_file, annotation_path, subtitles_path)
        
        if not annotation_csv:
            skipped_count += 1
            continue
        
        # Determine output path
        if output_folder:
            # Preserve relative structure in output folder
            relpath = excel_file.relative_to(subtitles_path)
            output_file = Path(output_folder) / relpath
        else:
            # Overwrite original file
            output_file = excel_file
        
        # Annotate the Excel file
        if annotate_subtitle_excel(excel_file, annotation_csv, output_file):
            annotated_count += 1
        else:
            skipped_count += 1
    
    print(f"\n✅ Processing complete!")
    print(f"   Total Excel files: {len(excel_files)}")
    print(f"   Annotated: {annotated_count} files")
    print(f"   Skipped (no annotation): {skipped_count} files")
    print(f"   Skipped (instagram): {instagram_skipped} files")

def main():
    parser = argparse.ArgumentParser(
        description="Annotate subtitle Excel files with emotion labels from annotation CSVs."
    )
    parser.add_argument(
        '--subtitles_folder',
        type=str,
        required=True,
        help='Path to folder containing subtitle Excel files (from subtitles.py)'
    )
    parser.add_argument(
        '--annotation_folder',
        type=str,
        required=True,
        help='Path to annotation folder with CSV files (structure: platform/label/video.csv)'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default=None,
        help='Optional: Output folder for annotated Excel files (default: overwrites originals)'
    )
    args = parser.parse_args()
    
    print(f"📂 Subtitles folder: {args.subtitles_folder}")
    print(f"📂 Annotation folder: {args.annotation_folder}")
    if args.output_folder:
        print(f"📂 Output folder: {args.output_folder}")
    else:
        print(f"📂 Output: Overwriting original files")
    
    process_all_subtitles(args.subtitles_folder, args.annotation_folder, args.output_folder)

if __name__ == '__main__':
    main()
