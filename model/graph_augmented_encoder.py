import torch
from torch import nn
from model.graph_attention_v2 import GraphAttentionV2
from model.GAT_model import GATModel
from transformers import RobertaTokenizer, RobertaModel, T5ForConditionalGeneration, T5Config, BeamSearchScorer


# Initialize graph and sequence models
in_channels = 768
out_channels = 768

# Load CodeT5 decoder with appropriate configuration
codet5_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
encoder = codet5_model.encoder
decoder = codet5_model.decoder

class graph_augmented_encoder(nn.Module):
  def __init__(self, encoder, graph_model, embedding_model):
    super(graph_augmented_encoder, self).__init__()
    self.encoder = encoder
    self.graph_model = graph_model
    self.embeddings = embedding_model.embeddings

  def forward(self, graphs, sequence_embeddings):
        graph_embeddings = torch.stack([self.graph_model(graph)[0] for graph in graphs])
        # print(f"Graph Embeddings Shape: {graph_embeddings.shape}")

        # Step 2: Combine graph and sequence embeddings using GraphAttentionV2
        combined_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)
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
        weighted_embeddings = attn_weights * sequence_embeddings  # Apply attention weights
        # print(f"Weighted Embeddings Shape: {weighted_embeddings.shape}")

        # Step 4: Forward pass through the CodeT5 encoder
        encoder_outputs = encoder(inputs_embeds=weighted_embeddings)
        print(f"CodeT5 Encoder Outputs Shape: {encoder_outputs.last_hidden_state.shape}")
        return encoder_outputs