
import os
import torch
from torch import nn
from torch.optim import Adam
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, T5ForConditionalGeneration, T5Config
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy

from graph_augmented_transformer import GraphAugmentedEncoder
from seq2seq import Seq2Seq
from GAT_model import GATModel
from model.data_loader import get_dataload

# Hyperparameters
vulnerability = 'command_injection'
batch_size = 1
max_embeddings_position = 20000
max_target_length = 256
learning_rate = 1e-4
num_epochs = 100
beam_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Loading
train_data_loader = get_dataload(device,vulnerability=vulnerability, loader_type='train', batch_size=batch_size, max_length=max_embeddings_position)

class VulnerabilityFixer(LightningModule):
    def __init__(self, train_data_loader):
        super().__init__()
        self.train_data_loader = train_data_loader

        # Tokenizer and Embedding Model
        config = RobertaConfig.from_pretrained("Salesforce/codet5-base")
        config.max_position_embeddings = max_embeddings_position
        embedding_model = RobertaModel.from_pretrained("Salesforce/codet5-base", config=config).to(device)
        self.tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", config=config)

        # GAT Model
        in_channels = 768
        out_channels = 768
        graph_model = GATModel(in_channels, out_channels)

        # CodeT5 Encoder and Decoder
        codet5_model = T5ForConditionalGeneration.from_pretrained("Salesforce/codet5-base")
        config = T5Config.from_pretrained("Salesforce/codet5-base")
        encoder = codet5_model.encoder

        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        # Graph-Augmented Encoder
        graph_encoder = GraphAugmentedEncoder(
            encoder=encoder,
            graph_model=graph_model,
            embedding_model=embedding_model,
            out_channels=out_channels
        ).to(device)

        # Seq2Seq Model
        self.s2s_model = Seq2Seq(
            encoder=graph_encoder,
            decoder=decoder,
            config=config,
            beam_size=beam_size,
            max_length=max_target_length,
            sos_id=self.tokenizer.bos_token_id,
            eos_id=self.tokenizer.sep_token_id,
            device=device
        )

        # Optimizer
        self.optimizer = None

    def forward(self, graphs, sequence_embeddings, source_ids, source_mask, target_ids=None, target_mask=None):
        return self.s2s_model(
            graphs,
            sequence_embeddings,
            source_ids=source_ids,
            source_mask=source_mask,
            target_ids=target_ids,
            target_mask=target_mask
        )

    def training_step(self, batch, batch_idx):
        # Unpack batch
        code_token_ids, fix_token_ids, _, _, graphs, sequence_embeddings, _ = batch
        source_mask = code_token_ids.ne(self.tokenizer.pad_token_id)
        target_mask = fix_token_ids.ne(self.tokenizer.pad_token_id)

        # Forward pass
        loss, _, _ = self.forward(
            graphs,
            sequence_embeddings,
            source_ids=code_token_ids,
            source_mask=source_mask,
            target_ids=fix_token_ids,
            target_mask=target_mask
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        return self.optimizer

    def predict_step(self, batch, batch_idx):
        code_token_ids, _, _, _, graphs, sequence_embeddings, _ = batch
        source_mask = code_token_ids.ne(self.tokenizer.pad_token_id)

        # Generate predictions
        preds = self.s2s_model(
            graphs,
            sequence_embeddings,
            source_ids=code_token_ids,
            source_mask=source_mask
        )

        # Decode predictions
        decoded_preds = [
            self.tokenizer.decode(pred[0], clean_up_tokenization_spaces=False) for pred in preds
        ]
        return decoded_preds

    def train_dataloader(self):
        return self.train_data_loader

# Callbacks and Trainer
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    save_top_k=1,
    mode="min",
    dirpath="checkpoints/",
    filename="best-checkpoint"
)

lr_monitor = LearningRateMonitor(logging_interval="step")

trainer = Trainer(
    accelerator="gpu",  # Use "gpu" for GPUs, or "cpu" for CPU
    devices=1,  # Specify the number of GPUs (use "auto" to auto-detect available GPUs)
    strategy="ddp_find_unused_parameters_true",  # Use DDP for multi-GPU distributed training
    precision=16,  # Mixed precision for faster training and reduced memory usage
    max_epochs=100,  # Maximum number of epochs
    callbacks=[checkpoint_callback, lr_monitor]
)

# Model Training
model = VulnerabilityFixer(train_data_loader)
trainer.fit(model)

torch.save(s2s_model, model_path + "model")
