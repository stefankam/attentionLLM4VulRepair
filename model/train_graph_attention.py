import torch
from torch import nn
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaModel, T5ForConditionalGeneration, T5Config, BeamSearchScorer
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

# Tokenize the original input code (you can also pass tokenized_codes from earlier)
tokenized_codes = tokenizer(codes, return_tensors='pt', padding=True, truncation=True, max_length=5120)

# Initialize GATModel and GraphAttentionV2
graph_model = GATModel(in_channels, out_channels)
combined_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)

# Load CodeT5 decoder with appropriate configuration
codet5_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
encoder = codet5_model.encoder
decoder = codet5_model.decoder

# Define final output layer
vocab_size = encoder.config.vocab_size
output_layer = nn.Linear(768, vocab_size)


# Define optimizer and loss function
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(output_layer.parameters()), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# Example training loop
num_epochs = 1

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

        # Step 3: Apply attention weights to the encoder (GraphAugmentedEncoder)
        attn_weights = attn_weights.mean(dim=-1)  # Average over the num_heads dimension
        print(f"Reduced Attention Weights Shape: {attn_weights.shape}")  # Should now be [batch_size, seq_len]
        # Reshape attn_weights to match [batch_size, seq_len, 1]
        attn_weights = attn_weights.unsqueeze(-1)  # Add the third dimension for broadcasting
        print(f"Reshaped Attention Weights Shape: {attn_weights.shape}")  # Should be [batch_size, seq_len, 1]
        # Apply attention weights
        weighted_embeddings = attn_weights * combined_embeddings  # Apply attention weights
        print(f"Weighted Embeddings Shape: {weighted_embeddings.shape}")

        # Step 4: Forward pass through the CodeT5 encoder
        encoder_outputs = encoder(inputs_embeds=weighted_embeddings)
        print(f"CodeT5 Encoder Outputs Shape: {encoder_outputs.last_hidden_state.shape}")

        # Convert original token IDs back to text using tokenizer (to see readable input to encoder)
        encoder_input_ids = tokenized_codes["input_ids"]  # Assuming you have tokenized_codes from the collate_fn
        # Decode the original input token IDs to text (before embedding conversion)
        encoder_text = tokenizer.batch_decode(encoder_input_ids, skip_special_tokens=True)
        print(f"Encoder Text Output: {encoder_text}")      

        # Step 5: Prepare inputs for the decoder
        batch_size, seq_len, _ = encoder_outputs.last_hidden_state.size()
        decoder_input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)  # Initialize decoder input IDs
        decoder_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)  # Binary mask

        #print(f"Combined Embeddings (sample): {combined_embeddings[0, :10, :].cpu().detach().numpy()}")
        print(f"Encoder Outputs (sample): {encoder_outputs.last_hidden_state[0, :10, :].cpu().detach().numpy()}")

        # Step 6: Forward pass through the CodeT5 decoder
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs.last_hidden_state,  # Use the encoder's output
        )
        print(f"Decoder Outputs Shape: {decoder_outputs.last_hidden_state}")

        # Extract the last hidden state from decoder_outputs
        decoder_hidden_states = decoder_outputs.last_hidden_state
        # Convert the decoder outputs (hidden states) back to token IDs using argmax
        decoder_token_ids = torch.argmax(decoder_hidden_states, dim=-1)
        # Decode the token IDs to readable text
        decoder_text = tokenizer.batch_decode(decoder_token_ids, skip_special_tokens=True)
        print(f"Decoder Text Output: {decoder_text}")
       
        # Step 7: Compute logits and loss
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
