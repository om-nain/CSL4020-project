# Convolutional Sequence-to-Sequence (ConvSeq2Seq) for Machine Translation

This project implements a Convolutional Sequence-to-Sequence (ConvSeq2Seq) model for machine translation. The model processes German–English translation pairs using a convolutional encoder and decoder architecture with positional encoding, gated linear units (GLU), residual connections, and a simple dot‑product attention mechanism.

The project is provided as a single Jupyter Notebook (convseq2seq.ipynb) containing the complete implementation, training pipeline, and inference code. A trained model checkpoint (convseq2seq-model.pt) is also included.

## File Structure

- *convseq2seq.ipynb*  
  A Jupyter Notebook that:
  - Downloads and extracts the WMT‑2014 English–German dataset from Kaggle.
  - Loads and preprocesses data (tokenization, vocabulary building, batching).
  - Defines the ConvSeq2Seq model (including positional encoding, encoder, decoder, and attention).
  - Contains training and evaluation loops.
  - Provides a greedy decoding function for inference.

- *convseq2seq-model.pt*  
  A model checkpoint containing the trained ConvSeq2Seq model weights.

## Installation

Ensure you have Python 3.7 or later installed. Then, install the required packages:

bash
pip install kaggle pandas spacy torch matplotlib
## Download the Required spaCy Language Models

To download the necessary spaCy language models, execute the following commands:

bash
python3 -m spacy download de_core_news_sm
python3 -m spacy download en_core_web_sm


## Requirements

- *Python 3.7 or later*
- *PyTorch*
- *Kaggle API* (with your credentials set up)
- *pandas*
- *spaCy* (with de_core_news_sm and en_core_web_sm models)
- *matplotlib*

## Notes

# Dataset Download

The notebook utilizes the Kaggle API to download the WMT‑2014 English–German dataset. Ensure that the dataset path in the notebook aligns with your directory structure; adjust it if necessary.&#8203;:contentReference[oaicite:0]{index=0}

# Data Processing

:contentReference[oaicite:1]{index=1}&#8203;:contentReference[oaicite:2]{index=2}

### Model Architecture

:contentReference[oaicite:3]{index=3}&#8203;:contentReference[oaicite:4]{index=4}

### Checkpointing

:contentReference[oaicite:5]{index=5}&#8203;:contentReference[oaicite:6]{index=6}

---


Happy training and translating!
