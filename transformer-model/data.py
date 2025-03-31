# data.py

import pickle  # For caching
import os
import time
import enum
import torch
import spacy
from datasets import load_dataset
from config import PAD_TOKEN, BOS_TOKEN, EOS_TOKEN

# Define a simple Field class to mimic torchtext's Field behavior
class Field:
    def __init__(self, tokenize, init_token=None, eos_token=None, pad_token="<pad>", batch_first=True):
        self.tokenize = tokenize
        self.init_token = init_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.batch_first = batch_first
        self.vocab = {}

    def build_vocab(self, examples, min_freq=1):
        freq = {}
        for ex in examples:
            for token in ex:
                freq[token] = freq.get(token, 0) + 1
        # Start vocab with special tokens
        self.vocab = {}
        for token in [self.pad_token] + ([self.init_token] if self.init_token else []) + ([self.eos_token] if self.eos_token else []):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        # Add tokens meeting the frequency threshold
        for token, count in freq.items():
            if count >= min_freq and token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def numericalize(self, tokens):
        # Unknown tokens default to index 0 (usually the pad token)
        return [self.vocab.get(token, 0) for token in tokens]

    @property
    def stoi(self):
        """String to index mapping."""
        return self.vocab

    @property
    def itos(self):
        """Index to string mapping."""
        return {index: token for token, index in self.vocab.items()}


# Define enum classes for dataset types and language directions
class DatasetType(enum.Enum):
    IWSLT = 0,
    WMT14 = 1

class LanguageDirection(enum.Enum):
    E2G = 0,
    G2E = 1


def get_datasets_and_vocabs(dataset_path, language_direction, use_iwslt=True, use_caching_mechanism=True):
    # Determine language direction: if German-to-English, then german_to_english is True.
    german_to_english = language_direction == LanguageDirection.G2E.name
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    src_tokenizer = tokenize_de if german_to_english else tokenize_en
    trg_tokenizer = tokenize_en if german_to_english else tokenize_de

    # Create Field processors for source and target
    src_field_processor = Field(tokenize=src_tokenizer, pad_token=PAD_TOKEN, batch_first=True)
    trg_field_processor = Field(tokenize=trg_tokenizer, init_token=BOS_TOKEN, eos_token=EOS_TOKEN, pad_token=PAD_TOKEN, batch_first=True)

    MAX_LEN = 100  # Filter out examples with more than MAX_LEN tokens
    filter_pred = lambda x: len(x["src"]) <= MAX_LEN and len(x["trg"]) <= MAX_LEN

    # Define cache file paths
    prefix = 'de_en' if german_to_english else 'en_de'
    prefix += '_iwslt' if use_iwslt else '_wmt14'
    train_cache_path = os.path.join(dataset_path, f'{prefix}_train_cache.pkl')
    val_cache_path = os.path.join(dataset_path, f'{prefix}_val_cache.pkl')
    test_cache_path = os.path.join(dataset_path, f'{prefix}_test_cache.pkl')

    ts = time.time()
    if not use_caching_mechanism or not (os.path.exists(train_cache_path) and os.path.exists(val_cache_path)):
        # Load raw dataset from Hugging Face
        if use_iwslt:
            builder_config = "iwslt2017-de-en" if german_to_english else "iwslt2017-en-de"
            raw_datasets = load_dataset("iwslt2017", builder_config, cache_dir=dataset_path, trust_remote_code=True)
        else:
            raw_datasets = load_dataset("wmt14", "de-en", cache_dir=dataset_path)

        # Preprocess each example
        def preprocess(example):
            if german_to_english:
                src_text = example["translation"]["de"]
                trg_text = example["translation"]["en"]
            else:
                src_text = example["translation"]["en"]
                trg_text = example["translation"]["de"]
            src_tokens = src_tokenizer(src_text)
            trg_tokens = trg_tokenizer(trg_text)
            if trg_field_processor.init_token is not None:
                trg_tokens = [trg_field_processor.init_token] + trg_tokens
            if trg_field_processor.eos_token is not None:
                trg_tokens = trg_tokens + [trg_field_processor.eos_token]
            return {"src": src_tokens, "trg": trg_tokens}

        # Process dataset splits
        train_dataset = raw_datasets["train"].map(preprocess)
        val_dataset = raw_datasets["validation"].map(preprocess) if "validation" in raw_datasets else None
        test_dataset = raw_datasets["test"].map(preprocess) if "test" in raw_datasets else None

        # Filter examples exceeding MAX_LEN
        train_dataset = train_dataset.filter(lambda x: filter_pred(x))
        if val_dataset is not None:
            val_dataset = val_dataset.filter(lambda x: filter_pred(x))
        if test_dataset is not None:
            test_dataset = test_dataset.filter(lambda x: filter_pred(x))

        # Cache the processed datasets
        with open(train_cache_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        if val_dataset is not None:
            with open(val_cache_path, 'wb') as f:
                pickle.dump(val_dataset, f)
        if test_dataset is not None:
            with open(test_cache_path, 'wb') as f:
                pickle.dump(test_dataset, f)
    else:
        with open(train_cache_path, 'rb') as f:
            train_dataset = pickle.load(f)
        with open(val_cache_path, 'rb') as f:
            val_dataset = pickle.load(f)
        if os.path.exists(test_cache_path):
            with open(test_cache_path, 'rb') as f:
                test_dataset = pickle.load(f)
        else:
            test_dataset = None

    print(f'Time it took to prepare the data: {time.time() - ts:.3f} seconds.')

    MIN_FREQ = 2
    # Build the vocabulary using tokenized training data
    src_field_processor.build_vocab(train_dataset["src"], min_freq=MIN_FREQ)
    trg_field_processor.build_vocab(train_dataset["trg"], min_freq=MIN_FREQ)

    # Save the vocabularies to disk
    src_vocab_path = os.path.join(dataset_path, "src_vocab.pkl")
    trg_vocab_path = os.path.join(dataset_path, "trg_vocab.pkl")
    with open(src_vocab_path, "wb") as f:
        pickle.dump(src_field_processor.vocab, f)
    with open(trg_vocab_path, "wb") as f:
        pickle.dump(trg_field_processor.vocab, f)
    print(f"Vocab saved to {src_vocab_path} and {trg_vocab_path}")

    return train_dataset, val_dataset, src_field_processor, trg_field_processor


def get_data_loaders(dataset_path, language_direction, dataset_name, batch_size, device):
    train_dataset, val_dataset, src_field_processor, trg_field_processor = get_datasets_and_vocabs(
        dataset_path, language_direction, use_iwslt=(dataset_name == DatasetType.IWSLT.name)
    )

    # Convert tokens to indices
    def numericalize(example):
        example["src"] = src_field_processor.numericalize(example["src"])
        example["trg"] = trg_field_processor.numericalize(example["trg"])
        return example

    train_dataset = train_dataset.map(numericalize)
    if val_dataset is not None:
        val_dataset = val_dataset.map(numericalize)

    # Create a simple PyTorch Dataset
    class TranslationDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            return item["src"], item["trg"]

    train_data = TranslationDataset(train_dataset)
    val_data = TranslationDataset(val_dataset) if val_dataset is not None else None

    # Collate function to pad sequences in the batch
    def collate_fn(batch):
        src_seqs, trg_seqs = zip(*batch)
        src_lengths = [len(seq) for seq in src_seqs]
        trg_lengths = [len(seq) for seq in trg_seqs]
        max_src = max(src_lengths)
        max_trg = max(trg_lengths)
        pad_idx_src = src_field_processor.vocab[src_field_processor.pad_token]
        pad_idx_trg = trg_field_processor.vocab[trg_field_processor.pad_token]
        padded_src = [seq + [pad_idx_src] * (max_src - len(seq)) for seq in src_seqs]
        padded_trg = [seq + [pad_idx_trg] * (max_trg - len(seq)) for seq in trg_seqs]
        return (torch.tensor(padded_src, dtype=torch.long, device=device),
                torch.tensor(padded_trg, dtype=torch.long, device=device))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    if val_data is not None:
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        val_loader = None

    return train_loader, val_loader, src_field_processor, trg_field_processor


global longest_src_sentence, longest_trg_sentence

def batch_size_fn(new_example, count, sofar):
    """
    If used in a BucketIterator, batch size is defined as the total number of tokens.
    (Note: This function is not used since we now rely on PyTorch DataLoader with a custom collate_fn)
    """
    global longest_src_sentence, longest_trg_sentence

    if count == 1:
        longest_src_sentence = 0
        longest_trg_sentence = 0

    longest_src_sentence = max(longest_src_sentence, len(new_example["src"]))
    longest_trg_sentence = max(longest_trg_sentence, len(new_example["trg"]) + 2)

    num_of_tokens_in_src_tensor = count * longest_src_sentence
    num_of_tokens_in_trg_tensor = count * longest_trg_sentence

    return max(num_of_tokens_in_src_tensor, num_of_tokens_in_trg_tensor)


def get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id):
    batch_size = src_token_ids_batch.shape[0]
    src_mask = (src_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)
    num_src_tokens = torch.sum(src_mask.long())
    return src_mask, num_src_tokens


def get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id):
    batch_size = trg_token_ids_batch.shape[0]
    device = trg_token_ids_batch.device
    trg_padding_mask = (trg_token_ids_batch != pad_token_id).view(batch_size, 1, 1, -1)
    sequence_length = trg_token_ids_batch.shape[1]
    trg_no_look_forward_mask = torch.triu(torch.ones((1, 1, sequence_length, sequence_length), device=device) == 1).transpose(2, 3)
    trg_mask = trg_padding_mask & trg_no_look_forward_mask
    num_trg_tokens = torch.sum(trg_padding_mask.long())
    return trg_mask, num_trg_tokens


def get_masks_and_count_tokens(src_token_ids_batch, trg_token_ids_batch, pad_token_id, device):
    src_mask, num_src_tokens = get_masks_and_count_tokens_src(src_token_ids_batch, pad_token_id)
    trg_mask, num_trg_tokens = get_masks_and_count_tokens_trg(trg_token_ids_batch, pad_token_id)
    return src_mask, trg_mask, num_src_tokens, num_trg_tokens


def get_src_and_trg_batches(token_ids_batch):
    src_token_ids_batch, trg_token_ids_batch = token_ids_batch
    trg_token_ids_batch_input = trg_token_ids_batch[:, :-1]
    trg_token_ids_batch_gt = trg_token_ids_batch[:, 1:].reshape(-1, 1)
    return src_token_ids_batch, trg_token_ids_batch_input, trg_token_ids_batch_gt


# Simple Example class to hold tokenized source and target sentences
class Example:
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg


def interleave_keys(a, b):
    return a * 10000 + b


class FastTranslationDataset(torch.utils.data.Dataset):
    """
    Caching mechanism to speed up data loading.
    Reads a cache file where source and target tokenized examples are interleaved.
    """
    @staticmethod
    def sort_key(ex):
        return interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, cache_path, fields, **kwargs):
        with open(cache_path, encoding='utf-8') as f:
            cached_data = [line.strip().split() for line in f]
        cached_data_src = cached_data[0::2]
        cached_data_trg = cached_data[1::2]
        assert len(cached_data_src) == len(cached_data_trg), "Source and target data should be of the same length."
        self.examples = []
        src_dataset_total_number_of_tokens = 0
        trg_dataset_total_number_of_tokens = 0
        for src_tokens, trg_tokens in zip(cached_data_src, cached_data_trg):
            ex = Example(src_tokens, trg_tokens)
            self.examples.append(ex)
            src_dataset_total_number_of_tokens += len(src_tokens)
            trg_dataset_total_number_of_tokens += len(trg_tokens)
        filename_parts = os.path.split(cache_path)[1].split('_')
        src_language, trg_language = ('English', 'German') if filename_parts[0] == 'en' else ('German', 'English')
        dataset_name = 'IWSLT' if filename_parts[2] == 'iwslt' else 'WMT-14'
        dataset_type = 'train' if filename_parts[3] == 'train' else 'val'
        print(f'{dataset_type} dataset ({dataset_name}) has {src_dataset_total_number_of_tokens} tokens in the source language ({src_language}) corpus.')
        print(f'{dataset_type} dataset ({dataset_name}) has {trg_dataset_total_number_of_tokens} tokens in the target language ({trg_language}) corpus.')
        self.fields = fields
        self.kwargs = kwargs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DatasetWrapper(FastTranslationDataset):
    """
    A simple wrapper for FastTranslationDataset to load both train and validation datasets.
    """
    @classmethod
    def get_train_and_val_datasets(cls, train_cache_path, val_cache_path, fields, **kwargs):
        train_dataset = cls(train_cache_path, fields, **kwargs)
        val_dataset = cls(val_cache_path, fields, **kwargs)
        return train_dataset, val_dataset


def save_cache(cache_path, dataset):
    """
    Saves the tokenized examples from dataset.examples into a cache file.
    Source examples are written on even lines, target examples on odd lines.
    """
    with open(cache_path, 'w', encoding='utf-8') as cache_file:
        for ex in dataset.examples:
            cache_file.write(' '.join(ex.src) + '\n')
            cache_file.write(' '.join(ex.trg) + '\n')
