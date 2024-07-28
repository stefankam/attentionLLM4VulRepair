from torch import nn

from model.graph_attention import GraphAttention
import torch.nn.functional as F


class GraphAugmentedEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphAugmentedEncoderLayer, self).__init__()
        self.self_attn = GraphAttention(embed_dim, num_heads, dropout)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, sequence_embeddings, graph_embeddings, mask=None):
        attn_output, attn_weights = self.self_attn(sequence_embeddings, graph_embeddings, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(sequence_embeddings + attn_output)

        ff_output = self.linear2(self.dropout(F.relu(self.linear1(out1))))
        ff_output = self.dropout2(ff_output)
        out2 = self.norm2(out1 + ff_output)
        return out2, attn_weights


class GraphAugmentedEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.1):
        super(GraphAugmentedEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [GraphAugmentedEncoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, sequence_embeddings, graph_embeddings, mask=None):
        output = sequence_embeddings
        attn_weights = None
        for layer in self.layers:
            output, attn_weights = layer(output, graph_embeddings, mask)
        return output, attn_weights
