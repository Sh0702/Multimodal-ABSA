import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
from models.proposedVLM import (
    extract_text_embeddings_maskedabsa,
    load_video_embeddings,
    MultimodalTextFrameDataset,
    SimpleFusionModel,
    init_lstm_weights,
    evaluate
)

def main():
    parser = argparse.ArgumentParser(description="Run Proposed VLM experiment with MaskedABSA text and EmoAffectNet video embeddings.")
    
    # Data paths
    parser.add_argument('--csv_path', type=str, required=True, help='CSV file with text, frame paths, and labels')
    parser.add_argument('--video_embeddings_path', type=str, required=True, help='Path to pre-computed EmoAffectNet video embeddings (.npy)')
    parser.add_argument('--text_column', type=str, default='text', help='CSV column name for text/subtitle')
    parser.add_argument('--label_column', type=str, default='label', help='CSV column name for labels')
    
    # Model paths
    parser.add_argument('--maskedabsa_model', type=str, default='Anshul99/ALM_BLM_Narratives_Stance_using', 
                        help='MaskedABSA model ID for text embeddings')
    parser.add_argument('--keras_lstm_path', type=str, default=None, 
                        help='Optional: Path to pretrained Keras LSTM model for weight initialization')
    
    # Embedding extraction
    parser.add_argument('--extract_text_embeddings', action='store_true', 
                        help='Extract text embeddings (otherwise assume they exist)')
    parser.add_argument('--text_embeddings_path', type=str, default=None,
                        help='Path to save/load text embeddings (.npy)')
    parser.add_argument('--embeddings_output_dir', type=str, default='./embeddings',
                        help='Output directory for embeddings')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    
    # Model save/load
    parser.add_argument('--save_dir', type=str, default='./checkpoints', 
                        help='Directory to save model checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.embeddings_output_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv(args.csv_path)
    df = df[df[args.label_column].isin([0, 1])].copy()
    df = df.dropna(subset=[args.text_column])
    print(f"✅ Loaded {len(df)} samples")
    
    # Load video embeddings (pre-computed from frame_processing.py)
    print("📹 Loading video embeddings...")
    video_embeddings = load_video_embeddings(args.video_embeddings_path)
    print(f"✅ Loaded video embeddings: {video_embeddings.shape}")
    
    # Extract or load text embeddings
    if args.extract_text_embeddings or args.text_embeddings_path is None:
        text_emb_path = args.text_embeddings_path or os.path.join(args.embeddings_output_dir, 'MaskedABSA_text_embedding.npy')
        print("📝 Extracting text embeddings using MaskedABSA...")
        text_list = df[args.text_column].tolist()
        text_embeddings = extract_text_embeddings_maskedabsa(
            text_list, device, model_id=args.maskedabsa_model, output_path=text_emb_path
        )
        print(f"✅ Extracted text embeddings: {text_embeddings.shape}")
    else:
        print(f"📝 Loading text embeddings from {args.text_embeddings_path}...")
        text_embeddings = np.load(args.text_embeddings_path)
        print(f"✅ Loaded text embeddings: {text_embeddings.shape}")
    
    # Ensure alignment
    min_len = min(len(df), len(video_embeddings), len(text_embeddings))
    df = df.iloc[:min_len].reset_index(drop=True)
    video_embeddings = video_embeddings[:min_len]
    text_embeddings = text_embeddings[:min_len]
    print(f"✅ Aligned to {min_len} samples")
    
    # Train/Val/Test split
    print("🔄 Splitting data...")
    train_idx, temp_idx = train_test_split(
        np.arange(len(df)), test_size=0.4, stratify=df[args.label_column], random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=df.loc[temp_idx, args.label_column], random_state=42
    )
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    
    train_video = video_embeddings[train_idx]
    val_video = video_embeddings[val_idx]
    test_video = video_embeddings[test_idx]
    
    train_text = text_embeddings[train_idx]
    val_text = text_embeddings[val_idx]
    test_text = text_embeddings[test_idx]
    
    # Create datasets
    train_set = MultimodalTextFrameDataset(train_df, train_text, train_video)
    val_set = MultimodalTextFrameDataset(val_df, val_text, val_video)
    test_set = MultimodalTextFrameDataset(test_df, test_text, test_video)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)
    
    print(f"✅ Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    # Initialize model
    print("🏗️  Building model...")
    model = SimpleFusionModel(text_dim=text_embeddings.shape[1], frame_dim=video_embeddings.shape[1])
    
    # Initialize LSTM weights from pretrained Keras model if provided
    if args.keras_lstm_path:
        print(f"🔄 Initializing LSTM weights from {args.keras_lstm_path}...")
        init_lstm_weights(model, args.keras_lstm_path)
        print("✅ LSTM weights initialized")
    
    model = model.to(device)
    
    # Training
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_from:
        print(f"🔄 Resuming from {args.resume_from}...")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        start_epoch = int(os.path.basename(args.resume_from).split('_')[-1].split('.')[0])
        print(f"✅ Resumed from epoch {start_epoch}")
    
    print("🚀 Starting training...")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            text = batch['text_embedding'].to(device)
            frame = batch['frame_embedding'].to(device)
            label = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(text, frame)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Save checkpoint
        ckpt_path = os.path.join(args.save_dir, f"proposed_model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")
        
        # Validation
        acc, f1, auc = evaluate(model, val_loader, device)
        print(f"✅ Epoch {epoch+1}: Val Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
    
    # Final evaluation on test set
    print("\n📊 Final Test Evaluation:")
    acc, f1, auc = evaluate(model, test_loader, device)
    print(f"Test Results - Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

if __name__ == '__main__':
    main()
