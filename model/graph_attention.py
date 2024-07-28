import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence_embeddings, graph_embeddings, mask=None):
        batch_size, seq_len, embed_dim = sequence_embeddings.size()

        q = self.q_proj(sequence_embeddings).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(sequence_embeddings).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(sequence_embeddings).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Integrate graph-based attention weights
        graph_attn_weights = torch.matmul(graph_embeddings.unsqueeze(1), k.transpose(-2, -1)).squeeze(1)
        graph_attn_weights = torch.nn.functional.softmax(graph_attn_weights, dim=-1)

        # Combine the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) * graph_attn_weights
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(attn_output)
        return output, attn_weights
