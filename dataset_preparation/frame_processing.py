import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 256
CHUNK_SIZE = 10000

def get_feature_model(model_path, projection_size=256):
    full_model = tf.keras.models.load_model(model_path)
    dense_512_output = full_model.get_layer("dense_4").output
    projected = tf.keras.layers.Dense(projection_size, activation='relu', name="projection_256")(dense_512_output)
    feature_model = tf.keras.Model(inputs=full_model.input, outputs=projected)
    feature_model.trainable = False
    return feature_model

@tf.function
def preprocess(path, label, name, img_size=IMG_SIZE):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0
    return img, label, name

def get_resume_info(chunk_dir, chunk_prefix):
    existing_chunks = sorted([
        int(f.split('chunk')[-1].split('.npy')[0])
        for f in os.listdir(chunk_dir) if f.startswith(chunk_prefix)
    ])
    completed_chunks = max(existing_chunks) + 1 if existing_chunks else 0
    return completed_chunks

def save_chunk(X, y, names, x_dir, y_dir, name_dir, i):
    np.save(os.path.join(x_dir, f'X_features_chunk{i}.npy'), X)
    np.save(os.path.join(y_dir, f'y_labels_chunk{i}.npy'), y)
    np.save(os.path.join(name_dir, f'X_videoname_chunk{i}.npy'), names)

def load_all_chunks(chunk_dir, prefix, decode_strings=False):
    files = sorted(
        [f for f in os.listdir(chunk_dir) if f.startswith(prefix)],
        key=lambda x: int(x.split('chunk')[-1].split('.npy')[0])
    )
    arrays = []
    for f in files:
        arr = np.load(os.path.join(chunk_dir, f), allow_pickle=True)
        if decode_strings and arr.dtype.type is np.bytes_:
            arr = np.array([s.decode('utf-8') for s in arr])
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)

def merge_chunks(
    X_chunk_dir,
    y_chunk_dir,
    name_chunk_dir,
    final_X_path,
    final_y_path,
    final_name_path
):
    print("🔄 Merging all chunks...")
    X_all = load_all_chunks(X_chunk_dir, 'X_features_chunk')
    y_all = load_all_chunks(y_chunk_dir, 'y_labels_chunk')
    name_all = load_all_chunks(name_chunk_dir, 'X_videoname_chunk', decode_strings=True)
    np.save(final_X_path, X_all)
    np.save(final_y_path, y_all)
    np.save(final_name_path, name_all)
    print(f"✅ Final merged shapes - X: {X_all.shape}, y: {y_all.shape}, names: {name_all.shape}")
    print(f"📁 Saved: {final_X_path}, {final_y_path}, {final_name_path}")

def main():
    parser = argparse.ArgumentParser(description="Frame feature extraction and chunk processing.")
    subparsers = parser.add_subparsers(dest="command")

    # Subparser for chunk processing (old main)
    process_parser = subparsers.add_parser("process", help="Process frames and save features in chunks.")
    process_parser.add_argument('--csv_path', required=True, help='CSV file with frame data')
    process_parser.add_argument('--frame_root', required=True, help='Root dir where frames are stored')
    process_parser.add_argument('--model_path', required=True, help='Pretrained .h5 model location')
    process_parser.add_argument('--output_dir', required=True, help='Where to save feature, label, name chunks')
    process_parser.add_argument('--img_size', type=int, nargs=2, default=(224,224))
    process_parser.add_argument('--batch_size', type=int, default=256)
    process_parser.add_argument('--chunk_size', type=int, default=10000)

    # Subparser for chunk merging
    merge_parser = subparsers.add_parser("merge", help="Merge all chunked .npy files into single arrays.")
    merge_parser.add_argument('--X_chunk_dir', required=True, help='Directory containing X_features_chunk*.npy files')
    merge_parser.add_argument('--y_chunk_dir', required=True, help='Directory containing y_labels_chunk*.npy files')
    merge_parser.add_argument('--name_chunk_dir', required=True, help='Directory containing X_videoname_chunk*.npy files')
    merge_parser.add_argument('--final_X_path', required=True, help='Output path for final merged features npy')
    merge_parser.add_argument('--final_y_path', required=True, help='Output path for final merged labels npy')
    merge_parser.add_argument('--final_name_path', required=True, help='Output path for final merged videonames npy')

    args = parser.parse_args()

    if args.command == "process":
        X_chunk_dir = os.path.join(args.output_dir, 'X_chunks')
        y_chunk_dir = os.path.join(args.output_dir, 'y_chunks')
        name_chunk_dir = os.path.join(args.output_dir, 'name_chunks')
        os.makedirs(X_chunk_dir, exist_ok=True)
        os.makedirs(y_chunk_dir, exist_ok=True)
        os.makedirs(name_chunk_dir, exist_ok=True)

        df = pd.read_csv(args.csv_path)
        df['full_path'] = df['frame_path'].apply(lambda x: os.path.join(args.frame_root, x))
        image_paths = df['full_path'].astype(str).values
        labels = df['label'].values
        names = df['frame_path'].astype(str).values
        total_samples = len(image_paths)
        total_chunks = (total_samples + args.chunk_size - 1) // args.chunk_size
        feature_model = get_feature_model(args.model_path, projection_size=256)
        completed_chunks = get_resume_info(X_chunk_dir, 'X_features_chunk')
        print(f"🔁 Total samples: {total_samples}, chunks: {total_chunks}")
        print(f"⏩ Resuming from chunk {completed_chunks + 1}...")
        for i in range(completed_chunks, total_chunks):
            start = i * args.chunk_size
            end = min((i + 1) * args.chunk_size, total_samples)
            print(f"\n🔹 Processing chunk {i + 1}/{total_chunks} - [{start}:{end}]")
            chunk_paths = image_paths[start:end]
            chunk_labels = labels[start:end]
            chunk_names = names[start:end]
            ds = tf.data.Dataset.from_tensor_slices((chunk_paths, chunk_labels, chunk_names))
            ds = ds.map(lambda p, l, n: preprocess(p, l, n, img_size=tuple(args.img_size)), num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
            chunk_X, chunk_y, chunk_names_out = [], [], []
            for batch_imgs, batch_lbls, batch_names in tqdm(ds):
                feats = feature_model(batch_imgs, training=False)
                chunk_X.append(feats.numpy())
                chunk_y.append(batch_lbls.numpy())
                chunk_names_out.extend(batch_names.numpy())
            X_chunk = np.concatenate(chunk_X, axis=0)
            y_chunk = np.concatenate(chunk_y, axis=0)
            name_chunk = np.array([n.decode("utf-8") for n in chunk_names_out])
            save_chunk(X_chunk, y_chunk, name_chunk, X_chunk_dir, y_chunk_dir, name_chunk_dir, i)
            print(f"✅ Chunk {i + 1} saved: X {X_chunk.shape}, y {y_chunk.shape}, names {name_chunk.shape}")
        print("\n✅ All available chunks processed.")
    elif args.command == "merge":
        merge_chunks(
            args.X_chunk_dir,
            args.y_chunk_dir,
            args.name_chunk_dir,
            args.final_X_path,
            args.final_y_path,
            args.final_name_path
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
