# main.py

import os
import time
import pickle
import argparse
import re
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from config import (
    CHECKPOINTS_PATH,
    BINARIES_PATH,
    DATA_DIR_PATH,
    BASELINE_MODEL_DIMENSION,
    BASELINE_MODEL_NUMBER_OF_HEADS,
    BASELINE_MODEL_NUMBER_OF_LAYERS,
    BASELINE_MODEL_DROPOUT_PROB,
    BASELINE_MODEL_LABEL_SMOOTHING_VALUE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)
from data import get_data_loaders, get_masks_and_count_tokens, get_src_and_trg_batches, DatasetType, LanguageDirection
from model import Transformer
from utils import (
    get_training_state,
    calculate_bleu_score,
    CustomLRAdamOptimizer,
    LabelSmoothingDistribution,
    get_available_binary_name,
)

# Global variables for logging
num_of_trg_tokens_processed = 0
bleu_scores = []
global_train_step, global_val_step = [0, 0]
writer = SummaryWriter()  # TensorBoard writer (outputs to ./runs/ by default)

# Decorator function to create the training/validation loop without passing redundant arguments
def get_train_val_loop(baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time_start):
    def train_val_loop(is_train, token_ids_loader, epoch):
        global num_of_trg_tokens_processed, global_train_step, global_val_step, writer, training_config

        if is_train:
            baseline_transformer.train()
        else:
            baseline_transformer.eval()

        device = next(baseline_transformer.parameters()).device

        # Core training/validation loop
        for batch_idx, token_ids_batch in enumerate(token_ids_loader):
            # Unpack the batch into source, target input, and target ground truth tokens.
            src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt = get_src_and_trg_batches(token_ids_batch)
            src_mask, trg_mask, num_src_tokens, num_trg_tokens = get_masks_and_count_tokens(
                src_token_ids_batch, trg_token_ids_batch_input, pad_token_id, device
            )

            predicted_log_distributions = baseline_transformer(
                src_token_ids_batch, trg_token_ids_batch_input, src_mask, trg_mask
            )
            smooth_target_distributions = label_smoothing(trg_token_ids_batch_gt)

            if is_train:
                custom_lr_optimizer.zero_grad()

            loss = kl_div_loss(predicted_log_distributions, smooth_target_distributions)

            if is_train:
                loss.backward()
                custom_lr_optimizer.step()

            # Logging and metrics
            if is_train:
                global_train_step += 1
                num_of_trg_tokens_processed += num_trg_tokens

                if training_config['enable_tensorboard']:
                    writer.add_scalar('training_loss', loss.item(), global_train_step)

                if training_config['console_log_freq'] is not None and batch_idx % training_config['console_log_freq'] == 0:
                    print(
                        f'Transformer training: time elapsed= {(time.time() - time_start):.2f} [s] '
                        f'| epoch= {epoch + 1} | batch= {batch_idx + 1} '
                        f'| target tokens/batch= {num_of_trg_tokens_processed / training_config["console_log_freq"]}'
                    )
                    num_of_trg_tokens_processed = 0

                # Save checkpoint at specified epochs (at the beginning of the epoch)
                if (training_config['checkpoint_freq'] is not None
                    and (epoch + 1) % training_config['checkpoint_freq'] == 0
                    and batch_idx == 0):
                    ckpt_model_name = f"transformer_ckpt_epoch_{epoch + 1}.pth"
                    torch.save(
                        get_training_state(training_config, baseline_transformer),
                        os.path.join(CHECKPOINTS_PATH, ckpt_model_name)
                    )
            else:
                global_val_step += 1
                if training_config['enable_tensorboard']:
                    writer.add_scalar('val_loss', loss.item(), global_val_step)

    return train_val_loop

def train_transformer(training_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Prepare data loaders (vocabularies will be saved in dataset_path as *.pkl files)
    train_token_ids_loader, val_token_ids_loader, src_field_processor, trg_field_processor = get_data_loaders(
        training_config['dataset_path'],
        training_config['language_direction'],
        training_config['dataset_name'],
        training_config['batch_size'],
        device
    )

    pad_token_id = src_field_processor.stoi[PAD_TOKEN]
    src_vocab_size = len(src_field_processor.stoi)
    trg_vocab_size = len(trg_field_processor.stoi)

    # Step 2: Initialize the Transformer model and push it to the selected device.
    baseline_transformer = Transformer(
        model_dimension=BASELINE_MODEL_DIMENSION,
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        number_of_heads=BASELINE_MODEL_NUMBER_OF_HEADS,
        number_of_layers=BASELINE_MODEL_NUMBER_OF_LAYERS,
        dropout_probability=BASELINE_MODEL_DROPOUT_PROB
    ).to(device)

    # Step 3: Setup training utilities.
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    label_smoothing = LabelSmoothingDistribution(
        BASELINE_MODEL_LABEL_SMOOTHING_VALUE, pad_token_id, trg_vocab_size, device
    )
    custom_lr_optimizer = CustomLRAdamOptimizer(
        Adam(baseline_transformer.parameters(), betas=(0.9, 0.98), eps=1e-9),
        BASELINE_MODEL_DIMENSION,
        training_config['num_warmup_steps']
    )

    train_val_loop = get_train_val_loop(
        baseline_transformer, custom_lr_optimizer, kl_div_loss, label_smoothing, pad_token_id, time.time()
    )

    # Step 4: Run training and validation loops.
    for epoch in range(training_config['num_of_epochs']):
        # Training loop
        train_val_loop(is_train=True, token_ids_loader=train_token_ids_loader, epoch=epoch)

        # Validation loop and BLEU evaluation
        with torch.no_grad():
            train_val_loop(is_train=False, token_ids_loader=val_token_ids_loader, epoch=epoch)
            bleu_score = calculate_bleu_score(baseline_transformer, val_token_ids_loader, trg_field_processor)
            if training_config['enable_tensorboard']:
                writer.add_scalar('bleu_score', bleu_score, epoch)

    # Save the final trained model binary.
    final_model_path = os.path.join(BINARIES_PATH, get_available_binary_name())
    torch.save(get_training_state(training_config, baseline_transformer), final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Save final vocab files alongside the trained model.
    src_vocab_final_path = os.path.join(BINARIES_PATH, "src_vocab.pkl")
    trg_vocab_final_path = os.path.join(BINARIES_PATH, "trg_vocab.pkl")
    with open(src_vocab_final_path, "wb") as f:
        pickle.dump(src_field_processor.vocab, f)
    with open(trg_vocab_final_path, "wb") as f:
        pickle.dump(trg_field_processor.vocab, f)
    print(f"Final vocab files saved to: {src_vocab_final_path} and {trg_vocab_final_path}")

def main():
    # Fixed argument: warm-up steps.
    num_warmup_steps = 4000

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_epochs", type=int, help="Number of training epochs", default=10)
    parser.add_argument("--batch_size", type=int, help="Target number of tokens in a batch", default=200)
    parser.add_argument(
        "--dataset_name",
        choices=[el.name for el in DatasetType],
        help="Which dataset to use for training",
        default=DatasetType.IWSLT.name
    )
    parser.add_argument(
        "--language_direction",
        choices=[el.name for el in LanguageDirection],
        help="Which direction to translate",
        default=LanguageDirection.E2G.name
    )
    parser.add_argument("--dataset_path", type=str, help="Path to download dataset", default=DATA_DIR_PATH)
    parser.add_argument("--enable_tensorboard", type=bool, help="Enable TensorBoard logging", default=True)
    parser.add_argument("--console_log_freq", type=int, help="Console log frequency (in batches)", default=10)
    parser.add_argument("--checkpoint_freq", type=int, help="Checkpoint saving frequency (in epochs)", default=1)

    args = parser.parse_args()
    global training_config
    training_config = {arg: getattr(args, arg) for arg in vars(args)}
    training_config['num_warmup_steps'] = num_warmup_steps

    train_transformer(training_config)

if __name__ == "__main__":
    main()
