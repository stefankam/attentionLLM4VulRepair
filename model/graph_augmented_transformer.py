

from torch import nn
import torch
import sys
sys.path.append('/home/skb67/attentionLLM4VulRepair/')
from model.graph_attention import GraphAttention
import torch.nn.functional as F

from model.graph_attention_v2 import GraphAttentionV2

# Check if a GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        # Step 1: Graph embeddings
        graph_embeddings = torch.stack([self.graph_model(graph)[0].to(device) for graph in graphs])

        # Step 2: Combine graph and sequence embeddings using GraphAttentionV2
        combined_model = GraphAttentionV2(embed_dim=self.out_channels, num_heads=1).to(device)
        sequence_embeddings = sequence_embeddings.to(device)
        graph_embeddings = graph_embeddings.to(device)
        combined_embeddings, attn_weights = combined_model(sequence_embeddings, graph_embeddings)

        # Step 3: Apply attention weights to the encoder
        attn_weights = attn_weights.mean(dim=-1).to(device)  # Average over num_heads dimension
        attn_weights = attn_weights.unsqueeze(-1)  # Add third dimension for broadcasting
        weighted_embeddings = attn_weights * combined_embeddings.to(device)  # Apply attention weights

        # Step 4: Forward pass through the CodeT5 encoder
        weighted_embeddings = weighted_embeddings.to(device)
        encoder_outputs = self.encoder(inputs_embeds=weighted_embeddings)
        
        return encoder_outputs

