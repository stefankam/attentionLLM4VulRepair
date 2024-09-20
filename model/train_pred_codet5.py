import torch
from torch import nn
from transformers import T5ForConditionalGeneration, T5Config, T5Tokenizer
from model.data_loader import data_loader
from model.GAT_model import GATModel
from model.graph_attention_v2 import GraphAttentionV2
from model.seq2seq import Seq2Seq, add_args, build_or_load_gen_model, Beam
import argparse

# Initialize graph and sequence models
in_channels = 768
out_channels = 768


# Parser
parser = argparse.ArgumentParser()
args = add_args(parser)
config, model, tokenizer = build_or_load_gen_model(args)

encoder = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base").encoder

class modifiedSeq2Seq():
    def __init__(self, encoder_output, decoder, labels, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.labels = labels
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.encoder_output = encoder_output

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        if target_ids is not None:
            attn_mask = -1e4 * (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])

            outputs = loss, loss * active_loss.sum(), active_loss.sum()
            return loss

        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = self.encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * (1 - self.bias[:input_ids.shape[1], :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute([1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p
                        in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            print("preds: ", preds)

def main():
    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch in data_loader:
            codes, fixes, graphs, sequence_embeddings, fix_embeddings = batch

            # Step 1: Forward pass through GATModel to get graph embeddings
            graph_model = GATModel(in_channels, out_channels)
            graph_embeddings = torch.stack([graph_model(graph)[0] for graph in graphs])

            # Step 2: Combine graph and sequence embeddings using GraphAttentionV2
            combined_model = GraphAttentionV2(embed_dim=out_channels, num_heads=1)
            combined_embeddings, attn_weights = combined_model(sequence_embeddings, graph_embeddings)

            # Step 3: Apply attention weights to the encoder
            attn_weights = attn_weights.mean(dim=-1).unsqueeze(-1)
            weighted_embeddings = attn_weights * sequence_embeddings  # Your weighted embeddings

            # Prepare source_ids and source_mask for the source code (sequence_embeddings here represents the source)
            source_inputs = tokenizer(codes, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            source_ids = source_inputs['input_ids']  # Tokenized source code
            source_mask = source_inputs['attention_mask']  # Attention mask for the source code

            # Tokenize the target fix sequences (fixes)
            target_inputs = tokenizer(fixes, padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            target_ids = target_inputs['input_ids']  # Tokenized target fixes
            target_mask = target_inputs['attention_mask']  # Attention mask for the target fixes

            # Position Index (for sequential or graph data)
            position_idx = torch.arange(0, source_ids.size(1), dtype=torch.long).unsqueeze(
                0)  # Sequential positional encoding

            # Attention Mask for the seq2seq transformer
            attn_mask = source_mask  # Can extend to include graph-related attention logic if needed

            encoder_output = encoder(inputs_embeds=weighted_embeddings)
            decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
            decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
            labels = torch.mean(fix_embeddings, dim=1)  # Averaging fix embeddings as the label
            loss = modifiedSeq2Seq(encoder_output=encoder_output, decoder=decoder, labels=labels, config=config,
                            beam_size=args.beam_size, max_length=args.max_target_length,
                            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

            print(f"Epoch {epoch}, Loss: {loss.item()}")




