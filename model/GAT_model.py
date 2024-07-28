import torch
from torch_geometric.nn import GATConv


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        self.gat_conv1 = GATConv(in_channels, out_channels, heads=8, dropout=0.6)
        self.gat_conv2 = GATConv(out_channels * 8, out_channels, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat_conv1(x, edge_index)
        x = torch.nn.functional.elu(x)
        x = self.gat_conv2(x, edge_index)
        return x
