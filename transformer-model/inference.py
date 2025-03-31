import matplotlib.pyplot as plt
import seaborn
import torch
import os
import time
import pickle
import re
from torch.hub import download_url_to_file

from config import BINARIES_PATH, DATA_DIR_PATH, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from data import get_datasets_and_vocabs, get_masks_and_count_tokens_src, get_src_and_trg_batches
from utils import (greedy_decoding, calculate_bleu_score, get_training_state, print_model_metadata)
from main import *

#########################################
# Visualization functions
#########################################

def plot_attention_heatmap(data, x, y, head_id, ax):
    seaborn.heatmap(data, xticklabels=x, yticklabels=y, square=True, vmin=0.0, vmax=1.0,
                      cbar=False, annot=True, fmt=".2f", ax=ax)
    ax.set_title(f'MHA head id = {head_id}')

def visualize_attention_helper(attention_weights, source_sentence_tokens=None, target_sentence_tokens=None, title=''):
    num_columns = 4
    num_rows = 2
    fig, axs = plt.subplots(num_rows, num_columns, figsize=(20, 10))

    assert source_sentence_tokens is not None or target_sentence_tokens is not None, \
        'Either source or target sentence must be passed in.'

    target_sentence_tokens = source_sentence_tokens if target_sentence_tokens is None else target_sentence_tokens
    source_sentence_tokens = target_sentence_tokens if source_sentence_tokens is None else source_sentence_tokens

    for head_id, head_attention_weights in enumerate(attention_weights):
        row_index = head_id // num_columns
        column_index = head_id % num_columns
        # For clarity, we pass target labels only for the first column
        tick_labels = target_sentence_tokens if head_id % num_columns == 0 else []
        plot_attention_heatmap(head_attention_weights, source_sentence_tokens, tick_labels, head_id, axs[row_index, column_index])

    fig.suptitle(title)
    plt.show()

def visualize_attention(baseline_transformer, source_sentence_tokens, target_sentence_tokens):
    encoder = baseline_transformer.encoder
    decoder = baseline_transformer.decoder

    # Remove EOS token from target tokens (we do not attend to it)
    target_sentence_tokens = target_sentence_tokens[0][:-1]

    # Visualize encoder attention weights
    for layer_id, encoder_layer in enumerate(encoder.encoder_layers):
        mha = encoder_layer.multi_headed_attention
        attention_weights = mha.attention_weights.cpu().numpy()[0]
        title = f'Encoder layer {layer_id + 1}'
        visualize_attention_helper(attention_weights, source_sentence_tokens, title=title)

    # Visualize decoder attention weights
    for layer_id, decoder_layer in enumerate(decoder.decoder_layers):
        mha_trg = decoder_layer.trg_multi_headed_attention
        mha_src = decoder_layer.src_multi_headed_attention
        attention_weights_trg = mha_trg.attention_weights.cpu().numpy()[0]
        attention_weights_src = mha_src.attention_weights.cpu().numpy()[0]
        title_trg = f'Decoder layer {layer_id + 1}, self-attention MHA'
        visualize_attention_helper(attention_weights_trg, target_sentence_tokens=target_sentence_tokens, title=title_trg)
        title_src = f'Decoder layer {layer_id + 1}, source-attending MHA'
        visualize_attention_helper(attention_weights_src, source_sentence_tokens, target_sentence_tokens, title=title_src)

#########################################
# Pretrained files URLs (Google Drive links converted)
#########################################

IWSLT_ENGLISH_TO_GERMAN_MODEL_URL = "https://drive.google.com/uc?id=1BxnqcktsPDR738mVQuKc6hS1XBwPObfg&export=download"
IWSLT_GERMAN_TO_ENGLISH_MODEL_URL = None
WMT14_ENGLISH_TO_GERMAN_MODEL_URL = None
WMT14_GERMAN_TO_ENGLISH_MODEL_URL = None

DOWNLOAD_DICT = {
    'iwslt_e2g': IWSLT_ENGLISH_TO_GERMAN_MODEL_URL,
    'iwslt_g2e': IWSLT_GERMAN_TO_ENGLISH_MODEL_URL,
    'wmt14_e2g': WMT14_ENGLISH_TO_GERMAN_MODEL_URL,
    'wmt14_g2e': WMT14_GERMAN_TO_ENGLISH_MODEL_URL
}

#########################################
# Download and loading functions for model and vocabs
#########################################

def download_models(translation_config):
    # Form key based on dataset and language direction
    language_direction = translation_config['language_direction'].lower()
    dataset_name = translation_config['dataset_name'].lower()
    key = f'{dataset_name}_{language_direction}'

    model_name = f'{key}.pth'
    model_path = os.path.join(BINARIES_PATH, model_name)
    if os.path.exists(model_path):
        print(f'Found model {model_path} trained on {dataset_name} for {language_direction}.')
        return model_path

    remote_resource_path = DOWNLOAD_DICT[key]
    if remote_resource_path is None:
        print(f'No pretrained model available for {dataset_name} and {language_direction}.')
        exit(0)

    print(f'Downloading pretrained model from {remote_resource_path}. This may take a while.')
    download_url_to_file(remote_resource_path, model_path)
    return model_path

def load_vocab(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def try_load_vocab_inference(dataset_path, binaries_path):
    dataset_src_path = os.path.join(dataset_path, "src_vocab.pkl")
    dataset_trg_path = os.path.join(dataset_path, "trg_vocab.pkl")
    bin_src_path = os.path.join(binaries_path, "src_vocab.pkl")
    bin_trg_path = os.path.join(binaries_path, "trg_vocab.pkl")

    if os.path.exists(dataset_src_path) and os.path.exists(dataset_trg_path):
        return load_vocab(dataset_src_path), load_vocab(dataset_trg_path), dataset_path
    if os.path.exists(bin_src_path) and os.path.exists(bin_trg_path):
        return load_vocab(bin_src_path), load_vocab(bin_trg_path), binaries_path

    # If vocab files are not found, download from the provided Google Drive URLs.
    SRC_VOCAB_URL = "https://drive.google.com/uc?id=1s2IpXPhm3aTwNO8KSCjD-_0CiWY_hha9&export=download"
    TRG_VOCAB_URL = "https://drive.google.com/uc?id=1ZN6B0Sq9xviDu5tti4YX5Q59cQy1hIjV&export=download"
    download_src_path = os.path.join(binaries_path, "src_vocab.pkl")
    download_trg_path = os.path.join(binaries_path, "trg_vocab.pkl")
    print("Downloading pretrained vocabulary files...")
    download_url_to_file(SRC_VOCAB_URL, download_src_path)
    download_url_to_file(TRG_VOCAB_URL, download_trg_path)
    return load_vocab(download_src_path), load_vocab(download_trg_path), binaries_path

def get_latest_model_path(translation_config):
    all_files = [f for f in os.listdir(BINARIES_PATH) if f.endswith('.pth')]
    trained_files = [f for f in all_files if f.startswith("transformer_")]
    if trained_files:
        latest_file = sorted(trained_files)[-1]
        return os.path.join(BINARIES_PATH, latest_file)
    return os.path.join(BINARIES_PATH, translation_config['model_name'])

#########################################
# Translation function
#########################################

def translate_a_single_sentence(translation_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare field processors
    _, _, src_field_processor, trg_field_processor = get_datasets_and_vocabs(
        translation_config['dataset_path'],
        translation_config['language_direction'],
        translation_config['dataset_name'] == "IWSLT"
    )

    # Attempt to load saved vocabs; if not found, download the pretrained ones.
    src_vocab_dict, trg_vocab_dict, loaded_from = try_load_vocab_inference(translation_config['dataset_path'], BINARIES_PATH)
    if src_vocab_dict is not None and trg_vocab_dict is not None:
        src_field_processor.vocab = src_vocab_dict
        trg_field_processor.vocab = trg_vocab_dict
        print(f"Loaded vocabulary from: {loaded_from}")
    else:
        print("Warning: No saved vocabulary found. Building a new one may lead to a mismatch with pretrained weights.")

    # Ensure that the PAD token indices match.
    assert src_field_processor.stoi[PAD_TOKEN] == trg_field_processor.stoi[PAD_TOKEN], \
        "PAD token indices differ between source and target vocabs!"
    pad_token_id = src_field_processor.stoi[PAD_TOKEN]

    # Prepare the model
    from model import Transformer  # Import here if not already imported
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=len(src_field_processor.stoi),
        trg_vocab_size=len(trg_field_processor.stoi),
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB,
        log_attention_weights=True
    ).to(device)

    # Load latest trained weights or download pretrained if needed
    model_path = get_latest_model_path(translation_config)
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found, attempting to download pretrained weights.")
        model_path = download_models(translation_config)
    model_state = torch.load(model_path, map_location=device)
    print_model_metadata(model_state)
    baseline_transformer.load_state_dict(model_state["state_dict"], strict=True)
    baseline_transformer.eval()

    # Process the input sentence
    source_sentence = translation_config['source_sentence']
    source_sentence_tokens = src_field_processor.tokenize(source_sentence)
    print(f"Source sentence tokens: {source_sentence_tokens}")
    src_ids = src_field_processor.numericalize(source_sentence_tokens)
    src_token_ids_batch = torch.tensor([src_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        src_mask, _ = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
        src_representations_batch = baseline_transformer.encode(src_token_ids_batch, src_mask)
        target_sentence_tokens = greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor)
        print(f"Translation | Target sentence tokens: {target_sentence_tokens}")

        if translation_config.get('visualize_attention', False):
            visualize_attention(baseline_transformer, source_sentence_tokens, target_sentence_tokens)

#########################################
# Argument parsing and configuration for inference
#########################################

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_sentence", type=str, help="Source sentence to translate", default="How are you doing today?")
parser.add_argument("--model_name", type=str, help="Transformer model name", default='iwslt_e2g.pth')
parser.add_argument("--dataset_name", type=str, choices=['IWSLT', 'WMT14'], help="Dataset to use", default="IWSLT")
parser.add_argument("--language_direction", type=str, choices=["E2G", "G2E"], help="Translation direction", default="E2G")
parser.add_argument("--dataset_path", type=str, help="Path to download dataset", default=DATA_DIR_PATH)
parser.add_argument("--beam_size", type=int, help="Beam size (if using beam search)", default=4)
parser.add_argument("--length_penalty_coefficient", type=float, help="Length penalty for beam search", default=0.6)
parser.add_argument("--visualize_attention", type=bool, help="Visualize attention", default=False)
args = parser.parse_args("")

translation_config = {arg: getattr(args, arg) for arg in vars(args)}
translate_a_single_sentence(translation_config)
