import argparse
import os
import numpy as np
import pandas as pd
import torch
from models.baselineVLM import extract_video_embeddings, extract_text_embeddings, align_embeddings

def main():
    parser = argparse.ArgumentParser(description="Run Baseline VLM Embedding Extraction and Alignment")
    parser.add_argument('--frames_csv', type=str, required=True, help='CSV file listing frame image paths, subtitles, and labels')
    parser.add_argument('--video_column', type=str, default='frame_path', help='CSV column for frame image path')
    parser.add_argument('--subtitle_column', type=str, default='subtitle', help='CSV column for subtitle')
    parser.add_argument('--label_column', type=str, default='label', help='CSV column for label')
    parser.add_argument('--output_dir', type=str, required=True, help='Output folder for embeddings')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    args = parser.parse_args()

    df = pd.read_csv(args.frames_csv)
    frame_paths = df[args.video_column].tolist()
    subtitle_texts = df[args.subtitle_column].tolist()
    labels = df[args.label_column].tolist()

    os.makedirs(args.output_dir, exist_ok=True)
    video_out = os.path.join(args.output_dir, 'ViLT_video_embedding.npy')
    text_out = os.path.join(args.output_dir, 'ViLT_text_embedding.npy')

    video_embeds = extract_video_embeddings(frame_paths, args.device, video_out)
    text_embeds = extract_text_embeddings(subtitle_texts, args.device, text_out)

    alignments = align_embeddings(labels, frame_paths, subtitle_texts, video_embeds, text_embeds)

    # Optionally save aligned structures for later loading
    np.save(os.path.join(args.output_dir, 'labels.npy'), alignments['labels'])
    np.save(os.path.join(args.output_dir, 'frame_names.npy'), alignments['frame_names'])
    np.save(os.path.join(args.output_dir, 'subtitle_texts.npy'), alignments['subtitle_texts'])
    print('✅ Baseline VLM dataset built and saved.')

if __name__ == '__main__':
    main()
