import numpy as np
import torch
from transformers import ViltProcessor, ViltModel, AutoTokenizer, AutoModel
from tqdm import tqdm
from PIL import Image
import os

# ================== Embedding Extraction ===================
def extract_video_embeddings(frame_paths, device, output_path):
    '''Extract ViLT embeddings for a list of frame image paths and save as .npy.'''
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    model = ViltModel.from_pretrained("dandelin/vilt-b32-mlm").to(device).eval()
    video_embeddings = []
    for path in tqdm(frame_paths, desc="Extracting ViLT video embeddings"):
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        video_embeddings.append(cls_embedding)
    video_embeddings = np.stack(video_embeddings)
    np.save(output_path, video_embeddings)
    return video_embeddings

def extract_text_embeddings(text_list, device, output_path, model_name='bert-base-uncased'):
    '''Extract BERT text embeddings for a list of texts (subtitles) and save as .npy.'''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    text_embeddings = []
    for text in tqdm(text_list, desc="Extracting text embeddings"):
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens)
            # Get CLS embedding (first position)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
        text_embeddings.append(embedding)
    text_embeddings = np.stack(text_embeddings)
    np.save(output_path, text_embeddings)
    return text_embeddings

# ============= Helper for Alignment ===============
def align_embeddings(labels, frame_names, subtitle_texts, video_embeds, text_embeds):
    '''Aligns video frames, subtitles, and labels to ensure order matches for baseline VLM dataset.'''
    # Placeholder: Modify this depending on your annotation and matching structure
    # Here we assume all lists are in perfect order (frame, subtitle, label match)
    return {
        'video_embeddings': video_embeds,
        'text_embeddings': text_embeds,
        'labels': np.array(labels),
        'frame_names': np.array(frame_names),
        'subtitle_texts': np.array(subtitle_texts)
    }