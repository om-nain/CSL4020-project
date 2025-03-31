# components.py
from config import *
import torch

class SublayerLogic(nn.Module):
    def __init__(self, model_dimension, dropout_probability):
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)

    # Applies the sublayer module to the normalized representations and adds a residual connection.
    def forward(self, representations_batch, sublayer_module):
        return representations_batch + self.dropout(sublayer_module(self.norm(representations_batch)))

class PositionwiseFeedForwardNet(nn.Module):
    """
    A position-wise feed-forward network applied independently to each token's representation.
    """
    def __init__(self, model_dimension, dropout_probability, width_mult=4):
        super().__init__()
        self.linear1 = nn.Linear(model_dimension, width_mult * model_dimension)
        self.linear2 = nn.Linear(width_mult * model_dimension, model_dimension)
        self.dropout = nn.Dropout(p=dropout_probability)
        self.relu = nn.ReLU()

    def forward(self, representations_batch):
        return self.linear2(self.dropout(self.relu(self.linear1(representations_batch))))

class Embedding(nn.Module):
    def __init__(self, vocab_size, model_dimension):
        super().__init__()
        self.embeddings_table = nn.Embedding(vocab_size, model_dimension)
        self.model_dimension = model_dimension

    def forward(self, token_ids_batch):
        assert token_ids_batch.ndim == 2, f'Expected: (batch size, max token sequence length), got {token_ids_batch.shape}'
        embeddings = self.embeddings_table(token_ids_batch)
        # Scale the embeddings by the square root of the model dimension (as per the paper)
        return embeddings * math.sqrt(self.model_dimension)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout_probability, expected_max_sequence_length=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_probability)
        position_id = torch.arange(0, expected_max_sequence_length).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dimension, 2, dtype=torch.float) / model_dimension)
        positional_encodings_table = torch.zeros(expected_max_sequence_length, model_dimension)
        positional_encodings_table[:, 0::2] = torch.sin(position_id * frequencies)  # sine for even positions
        positional_encodings_table[:, 1::2] = torch.cos(position_id * frequencies)  # cosine for odd positions
        self.register_buffer('positional_encodings_table', positional_encodings_table)

    def forward(self, embeddings_batch):
        assert embeddings_batch.ndim == 3 and embeddings_batch.shape[-1] == self.positional_encodings_table.shape[1], \
            f'Expected (batch size, max token sequence length, model dimension) got {embeddings_batch.shape}'
        # Get only as many positional encodings as needed for the sequence length
        positional_encodings = self.positional_encodings_table[:embeddings_batch.shape[1]]
        return self.dropout(embeddings_batch + positional_encodings)

def get_clones(module, num_of_deep_copies):
    """
    Produce a ModuleList of deep copies of the given module.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num_of_deep_copies)])

class MultiHeadedAttention(nn.Module):
    """
    Multi-headed attention module from scratch.
    """
    def __init__(self, model_dimension, number_of_heads, dropout_probability, log_attention_weights):
        super().__init__()
        assert model_dimension % number_of_heads == 0, f'Model dimension must be divisible by the number of heads.'
        self.head_dimension = model_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Three linear projections for query, key, and value.
        self.qkv_nets = get_clones(nn.Linear(model_dimension, model_dimension), 3)
        self.out_projection_net = nn.Linear(model_dimension, model_dimension)

        self.attention_dropout = nn.Dropout(p=dropout_probability)
        self.softmax = nn.Softmax(dim=-1)
        self.log_attention_weights = log_attention_weights
        self.attention_weights = None  # Cached for visualization if needed

    def attention(self, query, key, value, mask):
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        if mask is not None:
            scores.masked_fill_(mask == torch.tensor(False), float("-inf"))
        attention_weights = self.softmax(scores)
        attention_weights = self.attention_dropout(attention_weights)
        intermediate_token_representations = torch.matmul(attention_weights, value)
        return intermediate_token_representations, attention_weights

    def forward(self, query, key, value, mask):
        batch_size = query.shape[0]
        # Apply linear projections and reshape for multiple heads.
        query, key, value = [
            net(x).view(batch_size, -1, self.number_of_heads, self.head_dimension).transpose(1, 2)
            for net, x in zip(self.qkv_nets, (query, key, value))
        ]
        intermediate_token_representations, attention_weights = self.attention(query, key, value, mask)
        if self.log_attention_weights:
            self.attention_weights = attention_weights
        # Reshape back to (batch_size, sequence_length, model_dimension)
        reshaped = intermediate_token_representations.transpose(1, 2).reshape(batch_size, -1, self.number_of_heads * self.head_dimension)
        token_representations = self.out_projection_net(reshaped)
        return token_representations
