# config.py

# Python native libraries
import math
import copy
import os
import time
import enum
import argparse

# Visualization related imports
import matplotlib.pyplot as plt
import seaborn

# Deep learning related imports
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.hub import download_url_to_file

# Data manipulation related imports
from datasets import load_dataset

import spacy

# BLEU metric for machine translation evaluation
from nltk.translate.bleu_score import corpus_bleu

# Architecture related constants taken from the paper
BASELINE_MODEL_NUMBER_OF_LAYERS = 6
BASELINE_MODEL_DIMENSION = 512
BASELINE_MODEL_NUMBER_OF_HEADS = 8
BASELINE_MODEL_DROPOUT_PROB = 0.1
BASELINE_MODEL_LABEL_SMOOTHING_VALUE = 0.1

# Paths for saving models and data
CHECKPOINTS_PATH = os.path.join(os.getcwd(), 'models', 'checkpoints')  # semi-trained models during training
BINARIES_PATH = os.path.join(os.getcwd(), 'models', 'binaries')          # location where trained models are located
DATA_DIR_PATH = os.path.join(os.getcwd(), 'data')                        # training data storage

os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
os.makedirs(BINARIES_PATH, exist_ok=True)
os.makedirs(DATA_DIR_PATH, exist_ok=True)

# Special token symbols used in the data section
BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = "<pad>"
