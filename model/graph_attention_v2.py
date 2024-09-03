import torch.nn as nn


class GraphAttentionV2(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(GraphAttentionV2, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)

    def forward(self, sequence_embeddings, graph_embeddings, mask=None):
        # sequence embeddings: [batch_size, sequence_len, embed_dim]
        # graph embeddings: [batch_size, graph_size, embed_dim]
        attn_output, attn_weights = self.multihead_attn(sequence_embeddings,
                                                        graph_embeddings,
                                                        graph_embeddings,
                                                        key_padding_mask=mask)
        return attn_output, attn_weights
