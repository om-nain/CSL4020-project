#!/usr/bin/env python3
"""
Machine Translation using Convolutional Sequence-to-Sequence (ConvSeq2Seq)

This script downloads the WMT-2014 English-German dataset from Kaggle,
builds vocabularies, processes data, defines a ConvSeq2Seq model with
positional encoding, gated convolutional layers with GLU and residual connections,
and a simple dot-product attention mechanism. It includes training, evaluation,
and a greedy decoding inference function.

Before running:
  - Install required packages:
      pip install kaggle pandas spacy torch matplotlib
  - Install spaCy models:
      python3 -m spacy download de_core_news_sm
      python3 -m spacy download en_core_web_sm
  - Ensure your kaggle.json is set up correctly (placed in ~/.kaggle/)
"""
#Importing libraries 
import os
import math
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# -----------------------------
# Setup: Seed, Device, and Dataset Download
# -----------------------------
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Download dataset using Kaggle API (ensure kaggle is installed and configured)
# This command will download and unzip the dataset into the ./data directory.
# Note: In a non-notebook environment, use os.system or subprocess.
os.system("kaggle datasets download -d mohamedlotfy50/wmt-2014-english-german -p ./data --unzip")

# Set the dataset path based on your Kaggle output (adjust if needed)
dataset_path = "/Users/amolgaur/.cache/kagglehub/datasets/mohamedlotfy50/wmt-2014-english-german/versions/1"
print("Dataset downloaded to:", dataset_path)

# -----------------------------
# Section 2: Load spaCy Models and Define Tokenizers
# -----------------------------
import spacy
spacy_de = spacy.load("de_core_news_sm")
spacy_en = spacy.load("en_core_web_sm")

def tokenize_de(text):
    if not isinstance(text, str):
        return []
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    if not isinstance(text, str):
        return []
    return [tok.text for tok in spacy_en.tokenizer(text)]

# -----------------------------
# Section 3: Load the WMT-2014 Dataset Files (CSV Version)
# -----------------------------
def load_translation_pairs(split="train"):
    # Files are named: wmt14_translate_de-en_train.csv, etc.
    filename = f"wmt14_translate_de-en_{split}.csv"
    csv_file = os.path.join(dataset_path, filename)
    # Use the python engine and skip bad lines
    df = pd.read_csv(csv_file, engine="python", on_bad_lines="skip")
    pairs = df.to_dict(orient="records")
    return pairs

train_pairs = load_translation_pairs("train")
valid_pairs = load_translation_pairs("validation")
test_pairs  = load_translation_pairs("test")
print("Number of training pairs:", len(train_pairs))

# -----------------------------
# Section 4: Build Vocabularies from the Dataset
# -----------------------------
from collections import Counter

SPECIALS = ["<sos>", "<eos>", "<pad>"]

def build_vocab(tokenizer, pairs, key, specials=SPECIALS, max_size=10000):
    counter = Counter()
    for pair in pairs:
        text = pair.get(key)
        if text is None:
            continue
        tokens = tokenizer(text)
        counter.update(tokens)
    common_tokens = [token for token, _ in counter.most_common(max_size - len(specials))]
    vocab = {token: idx for idx, token in enumerate(specials)}
    for token in common_tokens:
        vocab[token] = len(vocab)
    inv_vocab = {idx: token for token, idx in vocab.items()}
    return vocab, inv_vocab

# Build vocab: Source is German ("de") and target is English ("en")
vocab_src, inv_vocab_src = build_vocab(tokenize_de, train_pairs, key="de", specials=SPECIALS, max_size=10000)
vocab_trg, inv_vocab_trg = build_vocab(tokenize_en, train_pairs, key="en", specials=SPECIALS, max_size=10000)

print("German vocab size (source):", len(vocab_src))
print("English vocab size (target):", len(vocab_trg))

# -----------------------------
# Section 5: Data Processing and Collate Function
# -----------------------------
def process_example(example):
    # For translation from German to English
    src_tokens = ["<sos>"] + tokenize_de(example["de"]) + ["<eos>"]
    trg_tokens = ["<sos>"] + tokenize_en(example["en"]) + ["<eos>"]
    src_indices = [vocab_src.get(token, vocab_src["<pad>"]) for token in src_tokens]
    trg_indices = [vocab_trg.get(token, vocab_trg["<pad>"]) for token in trg_tokens]
    src_tensor = torch.tensor(src_indices, dtype=torch.long)
    trg_tensor = torch.tensor(trg_indices, dtype=torch.long)
    return src_tensor, trg_tensor

def process_split(pairs):
    processed = [process_example(item) for item in pairs]
    # Filter out None values if any
    return [item for item in processed if item is not None]

train_data = process_split(train_pairs)
valid_data = process_split(valid_pairs)
test_data  = process_split(test_pairs)

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, padding_value=vocab_src["<pad>"], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=vocab_trg["<pad>"], batch_first=True)
    return src_batch, trg_batch

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# Section 6.1: Positional Encoding Module
# -----------------------------
class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=5000):
        super(PositionalEncoding, self)._init_()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -----------------------------
# Section 7: Define the Convolutional Encoder
# -----------------------------
class ConvEncoder(nn.Module):
    def _init_(self, vocab_size, embed_size, hidden_size, num_layers, kernel_size=3, dropout=0.1):
        super(ConvEncoder, self)._init_()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = PositionalEncoding(embed_size)
        self.fc = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2,
                      kernel_size=kernel_size, padding=(kernel_size - 1))
            for _ in range(num_layers)
        ])
    def forward(self, src):
        emb = self.embedding(src)
        emb = self.pos_embedding(emb)
        emb = self.fc(emb)
        emb = self.dropout(emb)
        conv_input = emb.transpose(1, 2)
        for conv in self.conv_layers:
            conv_out = conv(conv_input)
            conv_out = conv_out[:, :, :conv_input.size(2)]
            glu_out = F.glu(conv_out, dim=1)
            conv_input = (glu_out + conv_input) * math.sqrt(0.5)
        encoder_outputs = conv_input.transpose(1, 2)
        return encoder_outputs

# Quick test
print("Encoder test:")
src_batch, _ = next(iter(train_loader))
encoder = ConvEncoder(len(vocab_src), 256, 256, 4, kernel_size=3, dropout=0.1).to(device)
enc_out = encoder(src_batch.to(device))
print("Encoder output shape:", enc_out.shape)

# -----------------------------
# Section 8: Define the Convolutional Decoder with Attention
# -----------------------------
class ConvDecoder(nn.Module):
    def _init_(self, vocab_size, embed_size, hidden_size, num_layers, kernel_size=3, dropout=0.1):
        super(ConvDecoder, self)._init_()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = PositionalEncoding(embed_size)
        self.fc = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size * 2,
                      kernel_size=kernel_size, padding=(kernel_size - 1))
            for _ in range(num_layers)
        ])
        self.attn_linear = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)
    def forward(self, tgt, encoder_outputs):
        emb = self.embedding(tgt)
        emb = self.pos_embedding(emb)
        emb = self.fc(emb)
        emb = self.dropout(emb)
        conv_input = emb.transpose(1, 2)
        for conv in self.conv_layers:
            conv_out = conv(conv_input)
            conv_out = conv_out[:, :, :conv_input.size(2)]
            glu_out = F.glu(conv_out, dim=1)
            conv_input = (glu_out + conv_input) * math.sqrt(0.5)
        conv_output = conv_input.transpose(1, 2)
        queries = self.attn_linear(conv_output)
        attn_scores = torch.bmm(queries, encoder_outputs.transpose(1,2))
        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, encoder_outputs)
        combined = conv_output + context
        output = self.out(combined)
        return output, attn_weights

# Quick test
print("Decoder test:")
_, trg_batch = next(iter(train_loader))
decoder = ConvDecoder(len(vocab_trg), 256, 256, 4, kernel_size=3, dropout=0.1).to(device)
dec_out, attn_w = decoder(trg_batch.to(device), enc_out)
print("Decoder output shape:", dec_out.shape)
print("Attention weights shape:", attn_w.shape)

# -----------------------------
# Section 9: Define the Full ConvSeq2Seq Model
# -----------------------------
class ConvSeq2Seq(nn.Module):
    def _init_(self, src_vocab_size, tgt_vocab_size, embed_size, hidden_size, num_layers, kernel_size=3, dropout=0.1):
        super(ConvSeq2Seq, self)._init_()
        self.encoder = ConvEncoder(src_vocab_size, embed_size, hidden_size, num_layers, kernel_size, dropout)
        self.decoder = ConvDecoder(tgt_vocab_size, embed_size, hidden_size, num_layers, kernel_size, dropout)
    def forward(self, src, tgt):
        encoder_outputs = self.encoder(src)
        decoder_outputs, attn_weights = self.decoder(tgt, encoder_outputs)
        return decoder_outputs, attn_weights

INPUT_DIM = len(vocab_src)
OUTPUT_DIM = len(vocab_trg)
model = ConvSeq2Seq(INPUT_DIM, OUTPUT_DIM, 256, 256, 4, kernel_size=3, dropout=0.1).to(device)
print(model)

# -----------------------------
# Section 10: Define Optimizer and Loss Function
# -----------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)
PAD_IDX = vocab_trg["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# -----------------------------
# Section 11: Define Training and Evaluation Functions
# -----------------------------
def train_epoch(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, trg in iterator:
        optimizer.zero_grad()
        # Teacher forcing: input to decoder is trg[:, :-1], target is trg[:, 1:]
        output, _ = model(src, trg[:, :-1])
        output = output.contiguous().view(-1, output.shape[-1])
        trg_target = trg[:, 1:].contiguous().view(-1)
        loss = criterion(output, trg_target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in iterator:
            output, _ = model(src, trg[:, :-1])
            output = output.contiguous().view(-1, output.shape[-1])
            trg_target = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg_target)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs

# -----------------------------
# Section 12: Full Training Loop
# -----------------------------
N_EPOCHS = 2  # Change to desired epochs (e.g., 10) for full training
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    valid_loss = evaluate_epoch(model, valid_loader, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'convseq2seq-model.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

# -----------------------------
# Section 13: Greedy Decoding for Inference
# -----------------------------
def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()
    # Tokenize the input sentence using the German tokenizer
    tokens = [token.lower() for token in tokenize_de(sentence)]
    tokens = ["<sos>"] + tokens + ["<eos>"]
    src_indices = [src_vocab.get(token, src_vocab["<pad>"]) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs = model.encoder(src_tensor)
    
    trg_indices = [trg_vocab["<sos>"]]
    for i in range(max_len - 1):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output, _ = model.decoder(trg_tensor, encoder_outputs)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        trg_indices.append(next_token)
        if next_token == trg_vocab["<eos>"]:
            break
    trg_tokens = [inv_vocab_trg[i] for i in trg_indices]
    return trg_tokens[1:]  # remove <sos>

# Test translation on a sample German sentence
example_sentence = "Ein kleines Haus mit einem Garten ."
translation = translate_sentence(example_sentence, vocab_src, vocab_trg, model, device)
print("Translated:", " ".join(translation))
