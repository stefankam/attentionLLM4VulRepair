import torch
from torch import nn
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, T5ForConditionalGeneration, T5Config, BeamSearchScorer
from model.graph_augmented_encoder import graph_augmented_encoder
from model.seq2seq import Seq2Seq, add_args, build_or_load_gen_model, Beam
from model.GAT_model import GATModel
from model.graph_attention_v2 import GraphAttentionV2
from model.data_loader import data_loader
from torch.autograd import Variable

# Initialize tokenizer and models
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
embedding_model = RobertaModel.from_pretrained("Salesforce/codet5-base")

# Initialize graph and sequence models
in_channels = 768
out_channels = 768

# Initialize GATModel
graph_model = GATModel(in_channels, out_channels)

# Load CodeT5 decoder with appropriate configuration
codet5_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
config = RobertaConfig.from_pretrained("Salesforce/codet5-base")
encoder = codet5_model.encoder

decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

# Initialize GraphAugmentedEncoder
graph_encoder = graph_augmented_encoder(
    encoder=encoder,
    graph_model=graph_model,  # Pass the graph model
    embedding_model=embedding_model
)


# Initialize Seq2Seq model
config = T5Config.from_pretrained("Salesforce/codet5-base")
s2s_model = Seq2Seq(encoder=graph_encoder,
                    decoder=decoder,
                    config=config,
                    beam_size=16,
                    max_length=512,
                    sos_id=tokenizer.cls_token_id,
                    eos_id=tokenizer.sep_token_id)
#s2s_model.to(device)

# Define final output layer
vocab_size = encoder.config.vocab_size
output_layer = nn.Linear(768, vocab_size)
classifier = nn.Linear(config.hidden_size, 2)


# Define optimizer and loss function
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()) + list(output_layer.parameters()), lr=1e-4)
loss_fn = torch.nn.MSELoss()

# Example training loop
num_epochs = 1

for epoch in range(num_epochs):
    for batch in data_loader:
        # Unpack batch data
        code_token_ids, fix_token_ids, codes, fixes, graphs, sequence_embeddings, fix_embeddings = batch
        code_token_ids = code_token_ids
        fix_token_ids = fix_token_ids
        sequence_embeddings = sequence_embeddings
        fix_embeddings = fix_embeddings

        # tokenized_codes = tokenizer(codes, return_tensors='pt', padding=True, truncation=True, max_length=5120)
        source_mask = code_token_ids.ne(tokenizer.pad_token_id)
        target_mask = fix_token_ids.ne(tokenizer.pad_token_id)
        # Call Seq2Seq forward with all necessary inputs, including graphs and sequence_embeddings
        loss, _, _ = s2s_model(codet5_model, graphs, sequence_embeddings,
                               source_ids=code_token_ids,
                               source_mask=source_mask,
                               target_ids=fix_token_ids,
                               target_mask=target_mask)

        loss = Variable(loss, requires_grad=True)
        #preds = s2s_model(graphs, sequence_embeddings, source_ids=code_token_ids, source_mask=source_mask)
        #print("final decoder output: {}", preds)

        p = []
        if epoch == num_epochs - 1:
          with torch.no_grad():
            preds = s2s_model(model= codet5_model, graphs=graphs, sequence_embeddings=sequence_embeddings, source_ids = code_token_ids, source_mask = source_mask)
            for pred in preds:
              text = tokenizer.decode(pred[0],clean_up_tokenization_spaces=False)
              print(text)
              p.append(text)
          print("final decoder output: {}", p)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"Epoch {epoch}, Loss: {loss.item()}")
