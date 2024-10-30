import torch
from torch import nn
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, T5ForConditionalGeneration, T5Config

from graph_augmented_transformer import GraphAugmentedEncoder
from model.utils import print_metrics
from seq2seq import Seq2Seq
from GAT_model import GATModel
from model.data_loader import get_dataload

device = 'cuda'
torch.set_default_device(device)
vulnerability = 'command_injection'
# vulnerability = 'open_redirect'
batch_size=1
max_embeddings_position = 2048
max_target_length = 256


train_data_loader = get_dataload(device, vulnerability = vulnerability, batch_size=batch_size, max_length=max_embeddings_position)
model_path = 'model/pretrained_model/s2s/{}'.format(vulnerability)

# Initialize tokenizer and models
config = RobertaConfig.from_pretrained("Salesforce/codet5-base")
config.max_position_embeddings = max_embeddings_position  # Increase max position embeddings
embedding_model = RobertaModel.from_pretrained("Salesforce/codet5-base", config=config).to(device)
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", config=config)

# Initialize graph and sequence models
in_channels = 768
out_channels = 768

# Initialize GATModel
graph_model = GATModel(in_channels, out_channels)

# Load CodeT5 decoder with appropriate configuration
codet5_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").to(device)
config = T5Config.from_pretrained("Salesforce/codet5-base")
encoder = codet5_model.encoder

decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6).to(device)

# Initialize GraphAugmentedEncoder
graph_encoder = GraphAugmentedEncoder(
    encoder=encoder,
    graph_model=graph_model,  # Pass the graph model
    embedding_model=embedding_model,
    out_channels=out_channels
)

beam_size = 4
# Initialize Seq2Seq model
config = T5Config.from_pretrained("Salesforce/codet5-base")
s2s_model = Seq2Seq(encoder=graph_encoder,
                    decoder=decoder,
                    config=config,
                    beam_size=beam_size,
                    max_length=max_target_length,
                    sos_id=tokenizer.bos_token_id,
                    eos_id=tokenizer.sep_token_id,
                    device=device)


# Define optimizer and loss function
optimizer = Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4)

# Example training loop
num_epochs = 200

evaluation_after_training = True
evaluation_with_valid_data = True

for epoch in range(num_epochs):
    for batch in train_data_loader:
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
        loss, _, _ = s2s_model(graphs, sequence_embeddings,
                               source_ids=code_token_ids,
                               source_mask=source_mask,
                               target_ids=fix_token_ids,
                               target_mask=target_mask)
        p = []
        if epoch == num_epochs - 1:
          with torch.no_grad():
            preds = s2s_model(graphs, sequence_embeddings,
                              source_ids = code_token_ids, source_mask = source_mask)
            for pred in preds:
              text = tokenizer.decode(pred[0],clean_up_tokenization_spaces=False)
              p.append(text)
          print("final decoder output: {}", p)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        print(f"Epoch {epoch}, Loss: {loss.item()}")

if evaluation_after_training:
    # Save the model
    test_data_loader = get_dataload(device, vulnerability=vulnerability, loader_type='test',
                                    max_length=max_target_length)
    data_loads = [test_data_loader]
    if evaluation_with_valid_data:
        valid_data_loader = get_dataload(device, vulnerability=vulnerability, loader_type='valid',
                                         max_length=max_target_length)
        data_loads.append(valid_data_loader)
    references = []
    predictions = []
    for data_loader in data_loads:
        for batch in data_loader:
            code_token_ids, fix_token_ids, codes, fixes, graphs, sequence_embeddings, fix_embeddings = batch
            source_mask = code_token_ids.ne(tokenizer.pad_token_id)
            target_mask = fix_token_ids.ne(tokenizer.pad_token_id)
            with torch.no_grad():
                preds = s2s_model(graphs, sequence_embeddings,
                                  source_ids=code_token_ids,
                                  source_mask=source_mask)
                for pred in preds:
                    text = tokenizer.decode(pred[0], clean_up_tokenization_spaces=False)
                    predictions.append(text)
                for fix in fixes:
                    references.append(fix)
        print_metrics(references, predictions, lang='python')

torch.save(s2s_model, model_path)
