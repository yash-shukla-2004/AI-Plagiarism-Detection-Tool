from torch_geometric.nn import GATConv,GINConv,global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    def __init__(self, in_channels=32, out_channels=128):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=4, dropout=0.6)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(out_channels * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ))
        self.fc = nn.Linear(64, 64)  # Outputting 64-dimensional embedding

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)  # Final embedding for the code snippet
        return x