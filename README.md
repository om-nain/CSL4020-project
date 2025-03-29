# Neural Machine Translation Project

We're doing a deep learning project titled **"Design a deep learning model for Deep Learning-based Language Translation."** For the dataset, we'll be using the **WMT 2014 English-German** dataset. We need to train three models:
- Basic encoder-decoder RNN LSTM
- Base Transformer
- One additional approach

**Deliverables:**
1. A comprehensive report (4 pages) including all the results.
2. A demonstration UI (using the best model weights for the demo)
3. All the code files in a repository

---

## Dataset

**Dataset Name:** WMT 2014 English-German

**Description:**  
A parallel corpus for machine translation tasks, containing English and German sentence pairs.

**Languages:**  
- English (en)  
- German (de)

**Size:**
- **Training Data:** Approximately 4.5 million sentence pairs.
- **Validation Data:** Commonly uses "newstest2013" with 3,000 sentence pairs.
- **Test Data:** "newstest2014" comprising 3,003 sentence pairs.

**Data Format:**  
Plain text files with aligned sentences; English and German sentences are provided in separate files.

**Source:**  
Derived from the Europarl corpus, News Commentary corpus, and Common Crawl corpus.

**Access:**
- Official Website: [http://www.statmt.org/wmt14/translation-task.html](http://www.statmt.org/wmt14/translation-task.html)
- Kaggle: [https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german)

**Preprocessing Recommendations:**
- **Tokenization:** Segment text into tokens using tools like the Moses tokenizer.
- **Truecasing:** Normalize text casing to maintain consistency.
- **Subword Segmentation:** Apply Byte Pair Encoding (BPE) to handle rare and compound words.

**Usage:**  
Widely used for training and evaluating machine translation models, including RNNs, LSTMs, and Transformers.

**Licensing:**  
Usage terms are specified on the official website; users should review and comply with these terms.

**References:**
- Official WMT 2014 Translation Task Overview: [http://www.statmt.org/wmt14/translation-task.html](http://www.statmt.org/wmt14/translation-task.html)
- Kaggle Dataset Page: [https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german)

---

## Project Overview

Great! I’ll design optimal model architectures and full training pipelines for both an RNN/LSTM encoder-decoder and a base Transformer model using the WMT 2014 English-German dataset. These configurations will target BLEU scores of:
- **~20 for the LSTM model**
- **25–30 for the Transformer model**

They will be tuned for 4–5 hour training on an RTX 4090 using PyTorch. The full pipeline includes architecture design, hyperparameters, preprocessing, training strategy, and efficiency tips.

---

# Optimal Architectures and Training Pipelines for WMT14 En–De Translation

To meet the BLEU score targets within a ~5-hour training budget on a single NVIDIA RTX 4090 (24 GB), we design two models – an **Encoder-Decoder LSTM (Seq2Seq)** and a **Transformer (base)** – with optimized architectures and training pipelines. Both models use the WMT 2014 English-German dataset (~4.5 million sentence pairs) and share similar preprocessing (tokenization + subword encoding) to handle the large vocabulary. We emphasize memory-efficient settings (suitable for 24 GB VRAM) and techniques like mixed precision to accelerate training.

---

## Model 1: Encoder-Decoder LSTM (Seq2Seq)

### Architecture Design
- **Encoder-Decoder with Attention:**  
  A classic sequence-to-sequence RNN architecture with a **bidirectional LSTM** encoder and a unidirectional LSTM decoder. Includes a Luong-style global attention mechanism.
- **Layer Depth:**  
  Stack **4 LSTM layers** for both encoder and decoder (encoder is bi-directional on the bottom layer).  
  _Goal: ~20 BLEU_
- **Hidden Size and Embeddings:**  
  - Embedding size: **512**  
  - LSTM hidden state size: **512**  
  Optionally, tie input and output embeddings.
- **Vocabulary Size:**  
  After applying BPE, the vocabulary is ~32k tokens (shared across source and target).
- **Dropout:**  
  Apply dropout (≈0.2) on embeddings and between LSTM layers.

### Data Preprocessing
- **Text Normalization & Tokenization:**  
  Normalize punctuation and tokenize using tools like Moses.
- **Byte-Pair Encoding (BPE):**  
  Apply BPE with ~32k merge operations to handle rare words.
- **Train/Val/Test Split:**  
  - Training: ~4.5M sentence pairs  
  - Validation: "newstest2013" (≈3,000 sentence pairs)  
  - Test: "newstest2014" (≈3,003 sentence pairs)
- **Length Filtering & Batching:**  
  Remove or truncate very long sentences (e.g., >100 tokens). Bucket by length to minimize padding.

### Training Pipeline
- **Batching:**  
  Use ~64 sentence pairs per batch (or batch by token count, ~4096–8192 tokens).  
- **Forward Pass:**  
  - **Encoder:** Embed source tokens and process through the 4-layer bi-LSTM.  
  - **Decoder:** Use teacher forcing; feed `<s>` start token and ground truth tokens. Apply attention to encoder outputs.
- **Loss Function:**  
  Standard cross-entropy loss with optional label smoothing.
- **Optimizer:**  
  Adam optimizer with initial learning rate ~0.001 (β₁=0.9, β₂=0.999).  
  Use gradient clipping (clip norm ≈5).
- **Training Duration:**  
  Train for ~3–5 epochs (~4–5 hours) to reach ~20 BLEU.
- **Inference:**  
  Use beam search (beam size 5–10) for final translation and BLEU evaluation.

### Efficiency & Memory Optimization
- **Mixed Precision (FP16) Training:**  
  Use FP16 (with FP32 master weights) to utilize RTX 4090 Tensor Cores.
- **Efficient CuDNN LSTMs:**  
  Use PyTorch’s native `nn.LSTM` with packed sequences.
- **Gradient Accumulation:**  
  Optionally accumulate gradients over 2–4 mini-batches.
- **Data Loading:**  
  Utilize multiple workers and pre-fetching.
- **Teacher Forcing:**  
  Stick with teacher forcing for faster convergence.
- **Memory Footprint:**  
  With 24 GB VRAM, the model and activations fit comfortably in FP16.

---

## Model 2: Transformer (Attention-Based) – Base Model

### Architecture Design
- **Transformer Encoder-Decoder:**  
  Based on Vaswani et al. (2017) with self-attention and feed-forward layers; no recurrence.
- **Layer Depth:**  
  - Encoder: 6 layers  
  - Decoder: 6 layers
- **Model Dimensions:**  
  - Model dimension: **512**  
  - Feed-forward network (FFN) hidden size: **2048**
- **Multi-Head Attention:**  
  8 attention heads (each with 64-dim projections).
- **Positional Encoding:**  
  Fixed sinusoidal positional encodings are added.
- **Embeddings & Output Projection:**  
  Share source and target embeddings (512-dim) and optionally tie decoder projection.
- **Dropout and Regularization:**  
  Apply dropout = 0.1 and label smoothing = 0.1.
- **Parameter Count:**  
  Approximately 65 million parameters.

### Data Preprocessing
- **Same Pipeline as LSTM:**  
  Use the same tokenized and BPE-encoded data (~32k joint subword vocabulary).
- **Batches by Token Count:**  
  Create batches around 4096 tokens per batch.
- **Additional Processing:**  
  Insert `<s>` and `</s>` tokens. Shift decoder inputs for teacher forcing. Create encoder and decoder attention masks.
- **Parallel Data Loading:**  
  Use PyTorch DataLoader with multiple workers and pinning memory.

### Training Pipeline
- **Optimizer:**  
  Adam optimizer with recommended settings: β₁=0.9, β₂=0.98, ε=1e-9.
- **Learning Rate Schedule:**  
  Use a warm-up phase (~4000 steps) followed by inverse square root decay.
- **Forward Pass:**  
  - **Encoder:** Process source embeddings + positional encodings through 6 encoder layers.  
  - **Decoder:** Process shifted target tokens through 6 decoder layers with masked self-attention and encoder-decoder attention.
  - **Prediction:** Linear projection + softmax over ~32k vocabulary.
- **Loss Computation:**  
  Cross-entropy loss with label smoothing (ε=0.1). Apply mask to ignore `<pad>` tokens.
- **Backpropagation & Parameter Update:**  
  Compute gradients, optionally clip, and update via Adam.
- **Training Duration:**  
  Train for around 100k steps on a single GPU, targeting ~25–30 BLEU.
- **Training Techniques:**  
  Use gradient accumulation to simulate larger batch sizes if needed.
- **Inference:**  
  Use beam search (beam size 4–8) and post-process translations (BPE inversion, detokenization).

### Efficiency & Training Tricks
- **Mixed Precision & Tensor Cores:**  
  Use FP16/FP32 mixed precision with PyTorch’s `autocast` and `GradScaler` to speed up training.
- **Large Batch via Gradient Accumulation:**  
  Accumulate gradients to simulate a batch size of ~32k tokens.
- **Optimized Attention:**  
  Use fused attention routines (e.g., FlashAttention) to reduce memory and improve speed.
- **Parallelization:**  
  Fully utilize parallel operations in multi-head attention and FFN layers.
- **Regularization:**  
  Apply label smoothing (0.1) and dropout (0.1) to prevent overfitting.
- **Memory Considerations:**  
  Cap sequence lengths (e.g., 100 tokens) and consider gradient checkpointing if needed.
- **Throughput:**  
  Aim for >10,000 tokens/sec on RTX 4090 using FP16.

---

By following these pipelines and optimizations, the LSTM model should reach around **~20 BLEU** and the Transformer model should achieve **25–30 BLEU** on the WMT 2014 English-German dataset—all within a 4–5 hour training window on a single RTX 4090 using PyTorch.
