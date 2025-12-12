import os, re, torch
import pandas as pd
import numpy as np
from sklearn.utils import resample
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration

def normalize_phrases(text):
    text = re.sub(r'\bblack\s*lives\s*matter\b', 'black-lives-matter', text, flags=re.IGNORECASE)
    text = re.sub(r'\ball\s*lives\s*matter\b', 'all-lives-matter', text, flags=re.IGNORECASE)
    return text

def load_data(subtitle_path, annotation_path):
    data = []
    skipped = 0
    for platform in ['instagram', 'tiktok', 'youtube']:
        for label_folder in ['alm', 'blm']:
            sub_dir = os.path.join(subtitle_path, platform, label_folder)
            ann_dir = os.path.join(annotation_path, platform, label_folder)
            if not os.path.exists(sub_dir): continue
            for fname in os.listdir(sub_dir):
                if not fname.endswith('.txt'): continue
                txt_path = os.path.join(sub_dir, fname)
                csv_path = os.path.join(ann_dir, fname.replace('.txt', '.csv'))
                if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
                    skipped += 1
                    continue
                try:
                    with open(txt_path, 'r') as f:
                        text = normalize_phrases(f.read().strip())
                    ann_df = pd.read_csv(csv_path)
                    ann_df.columns = [c.strip().lower() for c in ann_df.columns]
                    if 'word' not in ann_df or 'emotion' not in ann_df: continue
                    ann_df['word'] = ann_df['word'].astype(str).apply(normalize_phrases)
                    for word, emotion in zip(ann_df['word'], ann_df['emotion']):
                        data.append({
                            'text': word,
                            'label': 1 if emotion.lower() == 'positive' else 0
                        })
                except: skipped += 1
    df = pd.DataFrame(data)
    # Oversampling
    df_majority = df[df.label == df.label.mode()[0]]
    df_minority = df[df.label != df.label.mode()[0]]
    df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1).reset_index(drop=True)
    return df_balanced

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=32):
        self.tokenizer = tokenizer
        self.data = df
        self.max_len = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_enc = self.tokenizer(
            row['input_text'], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        target_enc = self.tokenizer(
            row['target_text'], max_length=5, truncation=True, padding="max_length", return_tensors="pt"
        )
        labels = target_enc["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_enc["input_ids"].squeeze(),
            "attention_mask": input_enc["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "label_binary": row["label"]
        }
