from torch import nn
import torch

from model.graph_attention import GraphAttention
import torch.nn.functional as F

from model.graph_attention_v2 import GraphAttentionV2


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
    def __init__(self, encoder, graph_model, embedding_model, out_channels):
        super(GraphAugmentedEncoder, self).__init__()
        self.encoder = encoder
        self.graph_model = graph_model
        self.embeddings = embedding_model.embeddings
        self.out_channels = out_channels

    def forward(self, graphs, sequence_embeddings):
        graph_embeddings = torch.stack([self.graph_model(graph)[0] for graph in graphs])
        # print(f"Graph Embeddings Shape: {graph_embeddings.shape}")

        # Step 2: Combine graph and sequence embeddings using GraphAttentionV2
        combined_model = GraphAttentionV2(embed_dim=self.out_channels, num_heads=1)
        combined_embeddings, attn_weights = combined_model(sequence_embeddings, graph_embeddings)
        # print(f"Combined Embeddings Shape: {combined_embeddings.shape}")
        # print(f"Attention Weights Shape: {attn_weights.shape}")

        # Step 3: Apply attention weights to the encoder (GraphAugmentedEncoder)
        attn_weights = attn_weights.mean(dim=-1)  # Average over the num_heads dimension
        # print(f"Reduced Attention Weights Shape: {attn_weights.shape}")  # Should now be [batch_size, seq_len]
        # Reshape attn_weights to match [batch_size, seq_len, 1]
        attn_weights = attn_weights.unsqueeze(-1)  # Add the third dimension for broadcasting
        # print(f"Reshaped Attention Weights Shape: {attn_weights.shape}")  # Should be [batch_size, seq_len, 1]
        # Apply attention weights
        weighted_embeddings = attn_weights * combined_embeddings  # Apply attention weights
        # print(f"Weighted Embeddings Shape: {weighted_embeddings.shape}")

        # Step 4: Forward pass through the CodeT5 encoder
        encoder_outputs = self.encoder(inputs_embeds=weighted_embeddings)
        return encoder_outputs