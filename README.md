# Emotion Recognition via Multimodal Deep Learning

This repository contains code and experiments for emotion recognition using text, video, and vision-language models. It implements multiple approaches including text-only models, video-only models, and multimodal fusion models.

## 📋 Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Experiments](#experiments)
- [Project Structure](#project-structure)

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📊 Dataset Preparation

### Step 1: Extract Subtitles from Videos

Extract subtitles using Whisper and save as JSON files:

```bash
python dataset_preparation/subtitles.py \
  --input_videos /path/to/video/folder \
  --output_subtitles /path/to/output/subtitles
```

**Options:**
- `--textOnly`: Export only plain text subtitles (.txt files)
- `--textWithTimeStamp`: Export frame-by-frame subtitles with timestamps (.xlsx files)

### Step 2: Extract Frame Features (EmoAffectNet)

Extract visual features from video frames using a pretrained model:

```bash
# Process frames and save as chunks
python dataset_preparation/frame_processing.py process \
  --csv_path /path/to/frames.csv \
  --frame_root /path/to/frames \
  --model_path /path/to/pretrained_model.h5 \
  --output_dir /path/to/output/features \
  --batch_size 256 \
  --chunk_size 10000

# Merge chunks into final arrays
python dataset_preparation/frame_processing.py merge \
  --X_chunk_dir /path/to/output/features/X_chunks \
  --y_chunk_dir /path/to/output/features/y_chunks \
  --name_chunk_dir /path/to/output/features/name_chunks \
  --final_X_path /path/to/X_features.npy \
  --final_y_path /path/to/y_labels.npy \
  --final_name_path /path/to/X_videoname.npy
```

### Step 3: Extract ViLT Embeddings (Optional)

For baseline VLM experiments, extract ViLT embeddings from frames and subtitles:

```bash
# Extract embeddings in chunks
python dataset_preparation/vilt_embeddings.py process \
  --csv_path /path/to/data.csv \
  --chunk_dir /path/to/output/vilt_chunks \
  --chunk_size 1000

# Merge chunks
python dataset_preparation/vilt_embeddings.py merge \
  --chunk_dir /path/to/output/vilt_chunks \
  --final_X_path /path/to/X_features_vilt.npy \
  --final_y_path /path/to/y_labels_vilt.npy \
  --final_frame_path /path/to/X_videoname_vilt.npy
```

## 🧪 Experiments

### 1. EMOAffectNet (Video-Only LSTM)

Train an LSTM model on EmoAffectNet video features:

```bash
python experiments/run_emoaffectnet.py \
  --features_path /path/to/X_features.npy \
  --labels_path /path/to/y_labels.npy \
  --epochs 20 \
  --batch_size 64 \
  --seq_len 10 \
  --feature_dim 512 \
  --lstm_units 128
```

**Output:** Trained model with TensorBoard logs in `logs/fit/`

---

### 2. MaskedABSA (Text-Only)

Train a T5-based text-only sentiment classification model:

```bash
python experiments/run_masked_absa.py \
  --subtitle_base /path/to/subtitles \
  --annotation_base /path/to/annotations \
  --model_id "Anshul99/ALM_BLM_Narratives_Stance_using" \
  --batch_size 8 \
  --epochs 10 \
  --lr 3e-5
```

**Expected directory structure:**
```
subtitles/
  ├── instagram/
  │   ├── alm/
  │   └── blm/
  ├── tiktok/
  └── youtube/
annotations/
  ├── instagram/
  │   ├── alm/
  │   └── blm/
  ├── tiktok/
  └── youtube/
```

---

### 3. Baseline VLM (ViLT + BERT)

Extract embeddings and build baseline vision-language model:

```bash
python experiments/run_vlm_baseline.py \
  --frames_csv /path/to/frames_data.csv \
  --video_column frame_path \
  --subtitle_column subtitle \
  --label_column label \
  --output_dir /path/to/embeddings \
  --device cuda
```

**Output:** 
- `ViLT_video_embedding.npy`
- `ViLT_text_embedding.npy`
- `labels.npy`
- `frame_names.npy`
- `subtitle_texts.npy`

---

### 4. Proposed VLM (MaskedABSA + EmoAffectNet)

Train the proposed multimodal fusion model combining MaskedABSA text embeddings and EmoAffectNet video embeddings:

```bash
python experiments/run_proposed_vlm.py \
  --csv_path /path/to/data.csv \
  --video_embeddings_path /path/to/X_features.npy \
  --text_column text \
  --label_column label \
  --extract_text_embeddings \
  --embeddings_output_dir ./embeddings \
  --maskedabsa_model "Anshul99/ALM_BLM_Narratives_Stance_using" \
  --keras_lstm_path /path/to/pretrained_lstm.h5 \
  --batch_size 512 \
  --epochs 3 \
  --lr 2e-5 \
  --device cuda \
  --save_dir ./checkpoints
```

**Key Features:**
- Uses MaskedABSA T5EncoderModel for text embeddings
- Uses pre-computed EmoAffectNet features for video
- Optional LSTM weight initialization from pretrained Keras model
- Automatic train/val/test split (60/20/20)
- Checkpoint saving and resuming support

**Output:**
- Model checkpoints in `--save_dir`
- Text embeddings saved to `--embeddings_output_dir/MaskedABSA_text_embedding.npy`
- Training metrics (Accuracy, F1, AUC) printed after each epoch

---

## 📁 Project Structure

```
emotion-recognition/
├── dataset_preparation/     # Data preprocessing scripts
│   ├── frame_processing.py  # Extract EmoAffectNet features
│   ├── vilt_embeddings.py   # Extract ViLT embeddings
│   └── subtitles.py         # Extract subtitles from videos
├── models/                   # Model architectures and helpers
│   ├── emo_affectnet.py     # EMOAffectNet helpers
│   ├── maskedabsa.py        # MaskedABSA dataset and utilities
│   ├── baselineVLM.py       # Baseline VLM helpers
│   └── proposedVLM.py       # Proposed VLM model and helpers
├── experiments/              # Experiment scripts
│   ├── run_emoaffectnet.py  # EMOAffectNet experiment
│   ├── run_masked_absa.py   # MaskedABSA experiment
│   ├── run_vlm_baseline.py # Baseline VLM experiment
│   └── run_proposed_vlm.py # Proposed VLM experiment
├── notebooks/                # Original Jupyter notebooks
├── scripts/                  # Utility scripts
└── requirements.txt          # Python dependencies
```

## 📝 Notes

- All experiments use command-line arguments for flexibility
- Checkpoints are saved automatically during training
- Use `--help` flag with any script to see all available options
- Original notebooks are preserved in `notebooks/` for reference
- GPU is recommended for all experiments (especially VLM models)

## 🔧 Troubleshooting

**CUDA out of memory:** Reduce `--batch_size` in the experiment scripts.

**Missing dependencies:** Ensure all packages in `requirements.txt` are installed.

**Path errors:** Use absolute paths or ensure relative paths are correct from the project root.
