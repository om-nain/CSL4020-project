# utils.py

import torch
import time
import os
import math
import re
from config import BINARIES_PATH, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from nltk.translate.bleu_score import corpus_bleu
from data import get_masks_and_count_tokens_src, get_masks_and_count_tokens_trg

class CustomLRAdamOptimizer:
    """
    Linear ramp learning rate for the warm-up number of steps and then start decaying
    according to the inverse square root law of the current training step number.
    """
    def __init__(self, optimizer, model_dimension, num_of_warmup_steps):
        self.optimizer = optimizer
        self.model_size = model_dimension
        self.num_of_warmup_steps = num_of_warmup_steps
        self.current_step_number = 0

    def step(self):
        self.current_step_number += 1
        current_learning_rate = self.get_current_learning_rate()

        for p in self.optimizer.param_groups:
            p['lr'] = current_learning_rate

        self.optimizer.step()  # apply gradients

    def get_current_learning_rate(self):
        step = self.current_step_number
        warmup = self.num_of_warmup_steps
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


class LabelSmoothingDistribution(torch.nn.Module):
    """
    Instead of a one-hot target distribution, set the target word's probability to a 'confidence_value'
    (typically 0.9) and distribute the remaining 'smoothing_value' mass (typically 0.1) over the rest of the vocab.
    """
    def __init__(self, smoothing_value, pad_token_id, trg_vocab_size, device):
        assert 0.0 <= smoothing_value <= 1.0

        super(LabelSmoothingDistribution, self).__init__()

        self.confidence_value = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value

        self.pad_token_id = pad_token_id
        self.trg_vocab_size = trg_vocab_size
        self.device = device

    def forward(self, trg_token_ids_batch):
        batch_size = trg_token_ids_batch.shape[0]
        smooth_target_distributions = torch.zeros((batch_size, self.trg_vocab_size), device=self.device)

        # Distribute smoothing mass over all tokens except pad and the ground truth token
        smooth_target_distributions.fill_(self.smoothing_value / (self.trg_vocab_size - 2))

        smooth_target_distributions.scatter_(1, trg_token_ids_batch, self.confidence_value)
        smooth_target_distributions[:, self.pad_token_id] = 0.

        # Set distribution to all 0s for pad tokens
        smooth_target_distributions.masked_fill_(trg_token_ids_batch == self.pad_token_id, 0.)

        return smooth_target_distributions


def calculate_bleu_score(transformer, token_ids_loader, trg_field_processor):
    with torch.no_grad():
        pad_token_id = trg_field_processor.stoi[PAD_TOKEN]

        gt_sentences_corpus = []
        predicted_sentences_corpus = []

        ts = time.time()
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            # DataLoader now returns a tuple (src, trg)
            src_token_ids_batch, trg_token_ids_batch = token_ids_batch
            if batch_idx % 10 == 0:
                print(f'batch={batch_idx}, time elapsed = {time.time()-ts:.3f} seconds.')

            # Compute source representations only once
            src_mask, _ = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
            src_representations_batch = transformer.encode(src_token_ids_batch, src_mask)

            predicted_sentences = greedy_decoding(transformer, src_representations_batch, src_mask, trg_field_processor)
            predicted_sentences_corpus.extend(predicted_sentences)

            # Convert ground truth token IDs to tokens using the Field's reverse mapping
            trg_token_ids_batch = trg_token_ids_batch.cpu().numpy()
            for target_sentence_ids in trg_token_ids_batch:
                target_sentence_tokens = [trg_field_processor.itos[id] for id in target_sentence_ids if id != pad_token_id]
                gt_sentences_corpus.append([target_sentence_tokens])  # Each sentence as a single-element list

        bleu_score = corpus_bleu(gt_sentences_corpus, predicted_sentences_corpus)
        print(f'BLEU-4 corpus score = {bleu_score}, corpus length = {len(gt_sentences_corpus)}, time elapsed = {time.time()-ts:.3f} seconds.')
        return bleu_score


def greedy_decoding(baseline_transformer, src_representations_batch, src_mask, trg_field_processor, max_target_tokens=100):
    """
    Batch greedy decoding.
    Decoding is performed by sequentially predicting tokens until all sentences generate an EOS token
    or the maximum number of tokens is reached.
    """
    device = next(baseline_transformer.parameters()).device
    pad_token_id = trg_field_processor.stoi[PAD_TOKEN]

    # Initial prompt: start-of-sentence token for each instance (shape: B x 1)
    target_sentences_tokens = [[BOS_TOKEN] for _ in range(src_representations_batch.shape[0])]
    trg_token_ids_batch = torch.tensor([[trg_field_processor.stoi[tokens[0]]] for tokens in target_sentences_tokens], device=device)

    is_decoded = [False] * src_representations_batch.shape[0]

    while True:
        trg_mask, _ = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
        # Shape: (B*T, V) where T is the current sequence length and V is the target vocab size
        predicted_log_distributions = baseline_transformer.decode(trg_token_ids_batch, src_representations_batch, trg_mask, src_mask)

        num_of_trg_tokens = len(target_sentences_tokens[0])
        # Extract logits corresponding to the last token in each sequence
        predicted_log_distributions = predicted_log_distributions[num_of_trg_tokens-1::num_of_trg_tokens]

        # Greedy decoding: select the token with the highest probability
        most_probable_last_token_indices = torch.argmax(predicted_log_distributions, dim=-1).cpu().numpy()

        predicted_words = [trg_field_processor.itos[index] for index in most_probable_last_token_indices]

        for idx, predicted_word in enumerate(predicted_words):
            target_sentences_tokens[idx].append(predicted_word)
            if predicted_word == EOS_TOKEN:
                is_decoded[idx] = True

        if all(is_decoded) or num_of_trg_tokens == max_target_tokens:
            break

        trg_token_ids_batch = torch.cat(
            (trg_token_ids_batch, torch.unsqueeze(torch.tensor(most_probable_last_token_indices, device=device), 1)),
            1
        )

    # Post-process: truncate sequences after the EOS token
    target_sentences_tokens_post = []
    for tokens in target_sentences_tokens:
        try:
            target_index = tokens.index(EOS_TOKEN) + 1
        except ValueError:
            target_index = None
        target_sentences_tokens_post.append(tokens[:target_index])
    return target_sentences_tokens_post


def get_available_binary_name():
    prefix = 'transformer'

    def valid_binary_name(binary_name):
        # Regex to match names like "transformer_000001.pth"
        pattern = re.compile(rf'{prefix}_[0-9]{{6}}\.pth')
        return re.fullmatch(pattern, binary_name) is not None

    valid_binary_names = list(filter(valid_binary_name, os.listdir(BINARIES_PATH)))
    if valid_binary_names:
        last_binary_name = sorted(valid_binary_names)[-1]
        new_suffix = int(last_binary_name.split('.')[0][-6:]) + 1  # Increment suffix
        return f'{prefix}_{str(new_suffix).zfill(6)}.pth'
    else:
        return f'{prefix}_000000.pth'


def get_training_state(training_config, model):
    training_state = {
        "dataset_name": training_config['dataset_name'],
        "language_direction": training_config['language_direction'],
        "num_of_epochs": training_config['num_of_epochs'],
        "batch_size": training_config['batch_size'],
        "state_dict": model.state_dict()
    }
    return training_state


def print_model_metadata(training_state):
    header = f'\n{"*"*5} Model training metadata: {"*"*5}'
    print(header)
    for key, value in training_state.items():
        if key != 'state_dict':
            if key == 'language_direction':
                value = 'English to German' if value == 'E2G' else 'German to English'
            print(f'{key}: {value}')
    print(f'{"*" * len(header)}\n')
