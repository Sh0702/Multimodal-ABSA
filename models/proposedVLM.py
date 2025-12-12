import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import ast
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5EncoderModel
from tensorflow.keras.models import load_model

def extract_text_embeddings_maskedabsa(text_list, device, model_id="Anshul99/ALM_BLM_Narratives_Stance_using", output_path=None):
    """
    Extract text embeddings using MaskedABSA (T5EncoderModel).
    Returns numpy array of embeddings.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_id)
    encoder = T5EncoderModel.from_pretrained(model_id).to(device)
    encoder.eval()
    
    embeddings = []
    for text in tqdm(text_list, desc="Extracting MaskedABSA text embeddings"):
        enc = tokenizer(text, return_tensors='pt', max_length=64, padding='max_length', truncation=True).to(device)
        with torch.no_grad():
            text_emb = encoder(**enc).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(text_emb)
    
    embeddings = np.stack(embeddings)
    if output_path:
        np.save(output_path, embeddings)
    return embeddings

def load_video_embeddings(embeddings_path):
    """
    Load pre-computed EmoAffectNet video embeddings from .npy file.
    These should be extracted using frame_processing.py.
    """
    return np.load(embeddings_path)

class MultimodalTextFrameDataset(Dataset):
    """Dataset class for proposed VLM combining text and frame embeddings."""
    def __init__(self, df, text_embeddings, frame_embeddings):
        self.df = df.reset_index(drop=True)
        self.text_embeddings = text_embeddings
        self.frame_embeddings = frame_embeddings
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_emb = torch.tensor(self.text_embeddings[idx], dtype=torch.float32)
        frame_emb = torch.tensor(self.frame_embeddings[idx], dtype=torch.float32)
        label = int(row['label'])
        return {
            'text_embedding': text_emb,
            'frame_embedding': frame_emb,
            'label': torch.tensor(label, dtype=torch.long)
        }

class SimpleFusionModel(nn.Module):
    """Proposed VLM model combining text and video embeddings with LSTM temporal modeling."""
    def __init__(self, text_dim=1024, frame_dim=256, hidden_dim=256):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.frame_proj = nn.Linear(frame_dim, hidden_dim)
        
        # Temporal Model (LSTM layers)
        self.lstm1 = nn.LSTM(input_size=hidden_dim * 2, hidden_size=512, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        
        # Classifier on top of LSTM output
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, text_embed, frame_embed):
        t = self.text_proj(text_embed)  # (B, H)
        f = self.frame_proj(frame_embed)  # (B, H)
        x = torch.cat([t, f], dim=1)  # (B, H*2)
        x = x.unsqueeze(1)  # (B, 1, H*2) – adding sequence dimension
        
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)
        out = out2[:, -1, :]  # (B, 256) – take last hidden state
        
        return self.classifier(out)  # (B, 2)

def convert_lstm_weights(keras_weights):
    """Convert Keras LSTM weights to PyTorch format."""
    W, U, b = keras_weights
    return torch.from_numpy(W.T), torch.from_numpy(U.T), torch.from_numpy(b)

def init_lstm_weights(pytorch_model, keras_model_path):
    """
    Initialize LSTM weights from pretrained Keras model.
    Loads the Keras model and transfers LSTM weights to PyTorch model.
    """
    keras_model = load_model(keras_model_path)
    lstm1_weights = keras_model.layers[1].get_weights()
    lstm2_weights = keras_model.layers[2].get_weights()
    
    W_ih1, W_hh1, b1 = convert_lstm_weights(lstm1_weights)
    W_ih2, W_hh2, b2 = convert_lstm_weights(lstm2_weights)
    
    pytorch_model.lstm1.weight_ih_l0.data.copy_(W_ih1)
    pytorch_model.lstm1.weight_hh_l0.data.copy_(W_hh1)
    pytorch_model.lstm1.bias_ih_l0.data.copy_(b1)
    pytorch_model.lstm1.bias_hh_l0.data.zero_()
    
    pytorch_model.lstm2.weight_ih_l0.data.copy_(W_ih2)
    pytorch_model.lstm2.weight_hh_l0.data.copy_(W_hh2)
    pytorch_model.lstm2.bias_ih_l0.data.copy_(b2)
    pytorch_model.lstm2.bias_hh_l0.data.zero_()

def evaluate(model, loader, device, return_preds=False):
    """Evaluate model on a dataloader."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    
    model.eval()
    preds, probs, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            text = batch["text_embedding"].to(device)
            frame = batch["frame_embedding"].to(device)
            label = batch["label"].to(device)
            logits = model(text, frame)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1)
            preds.extend(pred.cpu().numpy())
            probs.extend(prob[:, 1].cpu().numpy())
            targets.extend(label.cpu().numpy())
    
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds)
    auc = roc_auc_score(targets, probs)
    return (acc, f1, auc, preds, targets) if return_preds else (acc, f1, auc)
