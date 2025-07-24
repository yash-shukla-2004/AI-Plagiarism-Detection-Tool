import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from old_app import train_loader

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

def moving_average(values, window_size=5):
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

class GNNModel(nn.Module):
    def __init__(self, in_channels=32, out_channels=128):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=4, dropout=0.6)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(out_channels * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ))
        self.fc = nn.Linear(64, 64)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

def smooth_triplet_loss(anchor, positive, negative, margin=1.0, smoothing=0.1):
    pos_dist = F.pairwise_distance(anchor, positive)
    neg_dist = F.pairwise_distance(anchor, negative)
    smooth_margin = margin * (1 - smoothing)
    loss = F.relu(pos_dist - neg_dist + smooth_margin)
    return loss.mean()

model = GNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

def train_triplet_loss(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for anchor, positive, negative in data_loader:
        optimizer.zero_grad()
        
        anchor_embed = model(anchor.x, anchor.edge_index, anchor.batch)
        positive_embed = model(positive.x, positive.edge_index, positive.batch)
        negative_embed = model(negative.x, negative.edge_index, negative.batch)
        
        loss = smooth_triplet_loss(anchor_embed, positive_embed, negative_embed)
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
        neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
        correct += (pos_dist < neg_dist).sum().item()
        total += anchor.x.size(0)
    
    scheduler.step()
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy

num_epochs = 50
losses = []
accuracies = []

for epoch in range(num_epochs):
    loss, acc = train_triplet_loss(model, train_loader, optimizer, scheduler)
    losses.append(loss)
    accuracies.append(acc)
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")

# Plot smoothed loss and accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(moving_average(losses), label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(moving_average(accuracies), label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.show()
