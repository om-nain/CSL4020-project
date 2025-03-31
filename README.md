
---


# Machine Translation using Deep Learning

This repository contains implementations of various deep learning approaches for English-to-German machine translation using the WMT14 dataset. The project compares three methods:

1. **Transformer-based Model**  
2. **LSTM-based Seq2Seq Model with Attention**  
3. **Convolutional Sequence-to-Sequence (ConvSeq2Seq) Model**

## Repository Structure

```
.
├── ConvSeq2Seq-Model
│   ├── Readme.md          # Instructions specific to the ConvSeq2Seq model
│   ├── convS2S.py         # Main code file for the ConvSeq2Seq model
│   └── convseq2seq-model.pt  # Pretrained ConvSeq2Seq model checkpoint
├── LSTM-model
│   ├── LSTM-model.py      # Main code file for the LSTM-based model
│   └── README.md          # Instructions specific to the LSTM model
├── LSTM_Enc_Dec.ipynb       # Jupyter Notebook for the LSTM Encoder-Decoder model
├── NMT_using_Transformer.ipynb  # Jupyter Notebook for the Transformer model (also available on Google Colab)
├── convS2S.ipynb            # Jupyter Notebook for the ConvSeq2Seq model
├── dataset                # Folder containing dataset files (or instructions to download them)
├── report                 # LaTeX source files and PDF of the project report
└── transformer-model
    ├── README.md          # Instructions specific to the Transformer model
    ├── components.py      # Model component definitions (e.g., attention, encoder, decoder)
    ├── config.py          # Configuration settings for training/inference
    ├── data.py            # Data loading and preprocessing code
    ├── inference.py       # Inference code for translating new sentences
    ├── main.py            # Main training script for the Transformer model
    ├── model.py           # Model definition for the Transformer
    ├── requirements.txt   # List of Python dependencies for the Transformer model
    └── utils.py           # Utility functions (e.g., learning rate scheduler, evaluation metrics)
```

## Overview

This project explores different architectures for neural machine translation (NMT):

- **Transformer-based Model:** Implements the encoder-decoder architecture with self-attention mechanisms, positional encoding, and multi-head attention. This model demonstrates superior performance and faster inference compared to traditional RNN-based models.

- **LSTM-based Seq2Seq Model with Attention:** Uses a 4-layer bidirectional LSTM encoder and a 4-layer LSTM decoder with Luong-style attention. This model serves as a robust baseline, highlighting the evolution from traditional recurrent approaches to modern architectures.

- **Convolutional Sequence-to-Sequence (ConvSeq2Seq) Model:** Explores the use of convolutional networks for sequence modeling. This approach, based on convolutional layers, offers an alternative perspective on handling sequence-to-sequence tasks.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Recommended libraries: PyTorch, NumPy, Matplotlib, Hugging Face Datasets, and others as specified in the respective `requirements.txt` files.

### Installation

Clone the repository:
```bash
git clone https://github.com/om-nain/CSL4020-project.git
cd csl4020-project
```

For each model, follow the instructions in the respective README files:
- **Transformer Model:** See `transformer-model/README.md`
- **LSTM Model:** See `LSTM-model/README.md`
- **ConvSeq2Seq Model:** See `ConvSeq2Seq-Model/Readme.md`

### Running the Models

#### Transformer Model
- To train or evaluate the Transformer model, open the `NMT_using_Transformer.ipynb` notebook.
- For quick inference, you can run the notebook on [Google Colab](https://colab.research.google.com/) to leverage free GPU resources.
- Alternatively, you can use the scripts in `transformer-model/` (e.g., run `main.py` for training and `inference.py` for inference).

#### LSTM-based Seq2Seq Model
- Open `LSTM_Enc_Dec.ipynb` to train or evaluate the LSTM model.
- Follow the instructions in the `LSTM-model/README.md` for more details.

#### ConvSeq2Seq Model
- Open `convS2S.ipynb` for a demonstration of the ConvSeq2Seq model.
- Detailed instructions are provided in the `ConvSeq2Seq-Model/Readme.md`.

## Dataset

The models are trained on the WMT14 English-German dataset. Preprocessing steps (tokenization, filtering, Byte Pair Encoding) are implemented in the code, ensuring a consistent setup across models. Refer to the respective code files for details.

## Report

The project report is available in the `report/` directory. It includes:
- A comprehensive description of the dataset and preprocessing steps.
- Detailed methodology for each model.
- Quantitative and qualitative evaluations (including BLEU scores and attention visualizations).
- Future directions and conclusions.

## Contact

For any questions or suggestions, please contact:
- Omprakash Nain (B22AI062)
- Abhishek Yadav (B22ES020)
- Amol Gaur (B22CS008)
- Sai Vignesh (B22ES023)


---