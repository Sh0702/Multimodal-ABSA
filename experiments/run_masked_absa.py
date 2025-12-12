import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.optim import AdamW
import matplotlib.pyplot as plt
from models.maskedabsa import load_data, SentimentDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Run MaskedABSA (text-only) sentiment experiment.")
    parser.add_argument('--subtitle_base', type=str, required=True, help='Path to subtitles root')
    parser.add_argument('--annotation_base', type=str, required=True, help='Path to annotation root')
    parser.add_argument('--model_id', type=str, default="Anshul99/ALM_BLM_Narratives_Stance_using")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-5)
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and process data
    df = load_data(args.subtitle_base, args.annotation_base)
    df = df.dropna().reset_index(drop=True)
    df['input_text'] = df['text'].apply(lambda x: f"classify sentiment: {x}")
    df['target_text'] = df['label'].map({1: "positive", 0: "negative"})
    # Split (60-20-20)
    from sklearn.model_selection import train_test_split
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        df[['input_text', 'target_text']],
        df['label'],
        test_size=0.4,
        stratify=df['label'],
        random_state=42
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)
    train_df = train_texts.copy(); train_df['label'] = train_labels
    val_df = val_texts.copy(); val_df['label'] = val_labels
    test_df = test_texts.copy(); test_df['label'] = test_labels

    # Tokenizer/model
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(DEVICE)
    train_dataset = SentimentDataset(train_df, tokenizer)
    val_dataset = SentimentDataset(val_df, tokenizer)
    test_dataset = SentimentDataset(test_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    train_losses = []

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                binary_targets = batch["label_binary"]
                output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=5)
                decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                decoded_preds = [1 if x.strip().lower() == "positive" else 0 for x in decoded_preds]
                preds.extend(decoded_preds)
                targets.extend(binary_targets)
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)
        try:
            auc = roc_auc_score(targets, preds)
        except Exception:
            auc = 0.0
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

    # Final evaluation
    model.eval()
    final_preds, final_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            binary_targets = batch["label_binary"]
            output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=5)
            decoded_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            decoded_preds = [1 if x.strip().lower() == "positive" else 0 for x in decoded_preds]
            final_preds.extend(decoded_preds)
            final_targets.extend(binary_targets)
    final_acc = accuracy_score(final_targets, final_preds)
    final_f1 = f1_score(final_targets, final_preds)
    final_auc = roc_auc_score(final_targets, final_preds)
    print(f"\nTest Set Results: Acc: {final_acc:.4f} | F1: {final_f1:.4f} | AUC: {final_auc:.4f}")
    # Plot Loss
    plt.plot(range(1, args.epochs + 1), train_losses, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
