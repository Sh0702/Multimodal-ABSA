import argparse
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import ViltProcessor, ViltModel

def save_chunk(X, y, frame_paths, x_chunk_dir, y_chunk_dir, chunk_id):
    np.save(os.path.join(x_chunk_dir, f"X_chunk_{chunk_id}.npy"), np.array(X))
    np.save(os.path.join(y_chunk_dir, f"y_chunk_{chunk_id}.npy"), np.array(y))
    np.save(os.path.join(y_chunk_dir, f"frame_paths_chunk_{chunk_id}.npy"), np.array(frame_paths))
    print(f"💾 Saved chunk {chunk_id}: {len(X)} samples")

def load_all_chunks(chunk_dir, prefix):
    files = sorted(
        [f for f in os.listdir(chunk_dir) if f.startswith(prefix)],
        key=lambda x: int(x.split("chunk_")[-1].split(".npy")[0])
    )
    return np.concatenate([np.load(os.path.join(chunk_dir, f)) for f in files], axis=0)

def process_vilt_embeddings(args):
    x_chunk_dir = os.path.join(args.chunk_dir, "X_chunks")
    y_chunk_dir = os.path.join(args.chunk_dir, "y_chunks")
    os.makedirs(x_chunk_dir, exist_ok=True)
    os.makedirs(y_chunk_dir, exist_ok=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device).eval()

    # Dataset
    df = pd.read_csv(args.csv_path)
    df = df[df['word_label'].notna()].reset_index(drop=True)

    # Get already processed frames
    existing_chunks = sorted([
        f for f in os.listdir(y_chunk_dir) if f.startswith("frame_paths_chunk_")
    ])
    existing_frames = set()
    for chunk_file in existing_chunks:
        chunk_paths = np.load(os.path.join(y_chunk_dir, chunk_file))
        existing_frames.update(chunk_paths)
    print(f"🧠 Skipping {len(existing_frames)} already processed frames.")
    chunk_id = len(existing_chunks)
    chunk_X, chunk_y, chunk_frames = [], [], []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        frame_path = row['frame']
        subtitle_text = row['subtitle']
        label = row['word_label']
        try:
            dir_path = os.path.dirname(frame_path)
            base = os.path.basename(frame_path)
            frame_num = int(base.replace("frame_", "").replace(".jpg", ""))
            corrected_filename = f"frame_{frame_num}.jpg"
            corrected_path = os.path.join(dir_path, corrected_filename)
            if corrected_path in existing_frames:
                continue
            image = Image.open(corrected_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Error reading image {corrected_path}: {e}")
            continue
        try:
            inputs = processor(
                images=image,
                text=subtitle_text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            chunk_X.append(cls_embedding)
            chunk_y.append(label)
            chunk_frames.append(corrected_path)
        except Exception as e:
            print(f"⚠️ ViLT failed on {corrected_path}: {e}")
            continue
        if len(chunk_X) >= args.chunk_size:
            save_chunk(chunk_X, chunk_y, chunk_frames, x_chunk_dir, y_chunk_dir, chunk_id)
            chunk_id += 1
            chunk_X, chunk_y, chunk_frames = [], [], []
    if chunk_X:
        save_chunk(chunk_X, chunk_y, chunk_frames, x_chunk_dir, y_chunk_dir, chunk_id)

def merge_vilt_chunks(args):
    x_chunk_dir = os.path.join(args.chunk_dir, "X_chunks")
    y_chunk_dir = os.path.join(args.chunk_dir, "y_chunks")
    final_X_path = args.final_X_path
    final_y_path = args.final_y_path
    final_frame_path = args.final_frame_path
    print("🔄 Merging ViLT chunks...")
    X_all = load_all_chunks(x_chunk_dir, "X_chunk")
    y_all = load_all_chunks(y_chunk_dir, "y_chunk")
    frame_paths_all = load_all_chunks(y_chunk_dir, "frame_paths_chunk")
    np.save(final_X_path, X_all)
    np.save(final_y_path, y_all)
    np.save(final_frame_path, frame_paths_all)
    print(f"✅ Final shapes — X: {X_all.shape}, y: {y_all.shape}, frames: {frame_paths_all.shape}")
    print(f"📁 Saved to:\n- {final_X_path}\n- {final_y_path}\n- {final_frame_path}")

def main():
    parser = argparse.ArgumentParser(description="Create and merge ViLT embeddings from frames and subtitles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    process_parser = subparsers.add_parser("process", help="Extract ViLT embeddings and save in chunks.")
    process_parser.add_argument('--csv_path', required=True, help='CSV file with frame, subtitle, and label columns')
    process_parser.add_argument('--chunk_dir', required=True, help='Directory to save X_chunks and y_chunks')
    process_parser.add_argument('--chunk_size', type=int, default=1000, help='Number of samples per chunk')

    merge_parser = subparsers.add_parser("merge", help="Merge all ViLT chunks into single numpy files.")
    merge_parser.add_argument('--chunk_dir', required=True, help='Parent directory containing X_chunks and y_chunks')
    merge_parser.add_argument('--final_X_path', required=True, help='Output file for merged feature array')
    merge_parser.add_argument('--final_y_path', required=True, help='Output file for merged label array')
    merge_parser.add_argument('--final_frame_path', required=True, help='Output file for merged frame path array')

    args = parser.parse_args()

    if args.command == "process":
        process_vilt_embeddings(args)
    elif args.command == "merge":
        merge_vilt_chunks(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
