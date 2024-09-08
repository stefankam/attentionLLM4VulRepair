import torch
from torch import nn
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaModel, T5ForConditionalGeneration, T5Config
from torch.utils.data import DataLoader
from experiment.utils import get_graph_dfg_data
from model.GAT_model import GATModel
from model.graph_attention_v2 import GraphAttentionV2
from model.graph_augmented_transformer import GraphAugmentedEncoder

# Initialize tokenizer and models
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
embedding_model = RobertaModel.from_pretrained("Salesforce/codet5-base")

# Initialize graph and sequence models
in_channels = 768
out_channels = 768

# Initialize GATModel and GraphAttentionV2
graph_model = GATModel(in_channels, out_channels)
combined_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)

# Load CodeT5 decoder with appropriate configuration
config = T5Config.from_pretrained('Salesforce/codet5-base')
decoder = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base', config=config).get_decoder()

# Define final output layer
vocab_size = decoder.config.vocab_size
output_layer = nn.Linear(768, vocab_size)

# Initialize the GraphAugmentedEncoder
encoder = GraphAugmentedEncoder(num_layers=6, embed_dim=768, num_heads=8)

# Define optimizer and loss function
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(output_layer.parameters()), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# Example training loop
num_epochs = 10

for epoch in range(num_epochs):
    for batch in data_loader:
        # Unpack batch data
        codes, fixes, graphs, sequence_embeddings, fix_embeddings = batch

        # Print shapes for debugging
        print(f"Batch size: {len(codes)}")
        print(f"Sequence Embeddings Shape: {sequence_embeddings.shape}")
        print(f"Graph Embeddings Shape (before GAT): {[g.num_nodes for g in graphs]}")
        print(f"Fix Embeddings Shape: {fix_embeddings.shape}")

        # Step 1: Forward pass through GATModel to get graph embeddings
        graph_embeddings = torch.stack([graph_model(graph)[0] for graph in graphs])
        print(f"Graph Embeddings Shape: {graph_embeddings.shape}")

        # Step 2: Combine graph and sequence embeddings using GraphAttentionV2
        combined_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)
        combined_embeddings, attn_weights = combined_model(sequence_embeddings, graph_embeddings)
        print(f"Combined Embeddings Shape: {combined_embeddings.shape}")
        print(f"Attention Weights Shape: {attn_weights.shape}")

        # Step 3: Prepare inputs for the decoder
        batch_size, seq_len, _ = combined_embeddings.size()
        decoder_input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)  # Initialize decoder input IDs
        decoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)  # Binary mask

        # Step 4: Forward pass through the decoder
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=combined_embeddings,  # Output from combined_model
        )

        print(f"Decoder Outputs Shape: {decoder_outputs.last_hidden_state.shape}")

        # Step 5: Compute logits and loss
        logits = decoder_outputs.last_hidden_state
        print(f"Logits Shape: {logits.shape}")

        # fix_embeddings for loss calculation
        # labels = fix_embeddings  # Flatten to [batch_size * seq_len * hidden_dim]
        labels = torch.mean(fix_embeddings, dim=1)  # labels will be [batch_size, hidden_dim]
        print(f"Labels Shape (after removing seq_len): {labels.shape}")

        # Compute the loss
        loss = loss_fn(logits, labels)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")



