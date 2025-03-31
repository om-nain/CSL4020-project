# Transformer for Machine Translation

This directory contains an implementation of a Transformer-based model for machine translation. The project has been modularized into multiple Python files to simplify training, evaluation, and inference. It also includes utilities for data preprocessing, model optimization, visualization (including attention heatmaps), and more.

## File Structure

- **config.py**  
  Contains common imports, constant definitions, and directory setup (for saving models and data).

- **components.py**  
  Implements helper components such as the sublayer logic, position-wise feed-forward network, embeddings, positional encoding, and multi-head attention.

- **model.py**  
  Defines the core Transformer architecture, including encoder, decoder, and the generator.

- **data.py**  
  Contains functions and classes for data processing, dataset caching, vocabulary building, and DataLoader creation.

- **utils.py**  
  Provides training utilities such as a custom learning rate scheduler, label smoothing, BLEU score calculation, greedy decoding, and helper functions for saving/loading checkpoints and binary files.

- **main.py**  
  The main training script that prepares data loaders, initializes the Transformer model, sets up training utilities, and runs the training and validation loops.

- **inference.py**  
  A standalone script for running inference on a single sentence. It downloads pretrained model weights and vocabulary files (if not already available) from Google Drive, performs translation, and visualizes attention (if desired).

## Installation

Make sure you have Python 3.7 or later installed. Then, install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

Additionally, download the SpaCy language models:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Usage

### Training the Model

To train the Transformer model, simply run:

```bash
python main.py
```

This script will:
- Download and preprocess the training dataset (using Hugging Face's `datasets` library).
- Build vocabularies and save them as `.pkl` files.
- Train the Transformer model while logging training metrics via TensorBoard.
- Save checkpoints and the final trained model binary to the specified directories.

### Running Inference

To perform translation using a pretrained model, run:

```bash
python inference.py --source_sentence "How are you doing today?"
```

The `inference.py` script will:
- Download pretrained model weights and vocabulary files from Google Drive if not already available.
- Initialize the Transformer model and load the pretrained weights.
- Tokenize and numericalize the input sentence.
- Perform translation using greedy decoding.
- Optionally, visualize the encoder and decoder attention heatmaps.

You can adjust inference parameters (such as `--visualize_attention`) via command-line arguments.

## Requirements

The project requires the following packages (see `requirements.txt` for details):
- `torch`
- `datasets`
- `matplotlib`
- `seaborn`
- `tensorboard`
- `spacy`
- `nltk`

## Notes

- **Data Caching:** Processed datasets and vocabularies are cached in the specified directories. Checkpoints and model binaries are saved to `models/checkpoints` and `models/binaries`, respectively.
- **Visualization:** The attention visualization functions display heatmaps for the multi-head attention weights.
- **Customization:** Feel free to modify hyperparameters (e.g., number of epochs, batch size) and the model architecture as needed.

---