import torch.nn as nn


class GraphAttentionV2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphAttentionV2, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, sequence_embeddings, graph_embeddings):
        # Ensure embeddings are on the same device as the model
        device = next(self.parameters()).device
        sequence_embeddings = sequence_embeddings.to(device)
        graph_embeddings = graph_embeddings.to(device)

        # Perform multihead attention
        attn_output, attn_weights = self.multihead_attn(
           sequence_embeddings,
           graph_embeddings,
           graph_embeddings
        )

        return attn_output, attn_weights
