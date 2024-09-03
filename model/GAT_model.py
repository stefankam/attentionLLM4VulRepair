import torch
from torch_geometric.nn import GATConv


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.6):
        super(GATModel, self).__init__()
        out_channels = out_channels // num_heads
        self.gat_conv1 = GATConv(in_channels, out_channels, heads=num_heads, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x, attention = self.gat_conv1(x, edge_index, return_attention_weights=True)
        return x, attention
