import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GATConv, GINConv, global_mean_pool
from torch_geometric.data import DataLoader
import torch.nn.functional as F
from old_app import train_loader
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# GAT+GIN Hybrid Model
class GNNModel(nn.Module):
    def __init__(self, in_channels=32, out_channels=128):
        super(GNNModel, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=4, dropout=0.6)
        self.gin = GINConv(nn.Sequential(
            nn.Linear(out_channels * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        ))
        self.fc = nn.Linear(64, 64)

    def forward(self, x, edge_index, batch):
        x = self.gat(x, edge_index)
        x = F.relu(x)
        x = self.gin(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Triplet Margin Loss
triplet_loss = nn.TripletMarginLoss(margin=1.5, p=2)

# Model & Optimizer
model = GNNModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)


# K-means clustering and embedding collection
def kmeans_clustering(embeddings, n_clusters=2):
    # Perform K-means clustering on the embeddings
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(embeddings)
    return kmeans

# Training Loop with Triplet Loss and K-means
def train_triplet_loss(model, data_loader, optimizer, epoch,scheduler, kmeans=None):
    model.train()
    total_loss = 0
    embeddings = []  # Collect embeddings for K-means
    all_labels = []  # To store the true labels (for evaluation)

    for anchor, positive, negative in data_loader:
        optimizer.zero_grad()

        # Generate embeddings
        anchor_embed = model(anchor.x, anchor.edge_index, anchor.batch)
        positive_embed = model(positive.x, positive.edge_index, positive.batch)
        negative_embed = model(negative.x, negative.edge_index, negative.batch)

        # Triplet Loss
        loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Collect embeddings for clustering and evaluation
        embeddings.append(anchor_embed.detach())
        embeddings.append(positive_embed.detach())
        embeddings.append(negative_embed.detach())


        # Assuming you have true labels for anchor samples, e.g., 1 for plagiarized, 0 for non-plagiarized
        # You should have a way to get the true labels for your dataset
        # For now, I'm assuming you have a function to get the true labels
        # all_labels.extend(get_true_labels(anchor))  # Replace with actual labels
    scheduler.step()
    # Concatenate embeddings from the current batch
    all_embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    
    # Apply K-means Clustering after each epoch
    kmeans = kmeans_clustering(all_embeddings)
    # Call after training
    
    evaluate_silhouette(all_embeddings, kmeans)
    # Update K-means for next epoch
    return total_loss / len(data_loader), kmeans


def evaluate(model, data_loader, kmeans, threshold=0.8):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for anchor, positive, negative in data_loader:
            # Get embeddings for anchor, positive, and negative
            anchor_embed = model(anchor.x, anchor.edge_index, anchor.batch)
            positive_embed = model(positive.x, positive.edge_index, positive.batch)
            
            # Compute cosine similarity between anchor and positive (for plagiarism detection)
            cosine_sim = F.cosine_similarity(anchor_embed, positive_embed)
            
            # Predict plagiarism (1 if cosine similarity > threshold, else 0)
            preds = (cosine_sim > threshold).int()

            # Append predictions and true labels
            all_preds.extend(preds.cpu().numpy())

            # Assuming you have true labels for anchor samples, e.g., 1 for plagiarized, 0 for non-plagiarized
            all_labels.extend(anchor.y.cpu().numpy())  # Replace with actual labels

    # Calculate plagiarism percentage (accuracy or other metrics)
    plagiarism_percentage = accuracy_score(all_labels, all_preds) * 100
    return plagiarism_percentage

def evaluate_cosine_similarity(model, data_loader, threshold=0.8):
    model.eval()
    similarity_scores = []

    with torch.no_grad():
        for anchor, positive, _ in data_loader:
            anchor_embed = model(anchor.x, anchor.edge_index, anchor.batch)
            positive_embed = model(positive.x, positive.edge_index, positive.batch)

            sim = F.cosine_similarity(anchor_embed, positive_embed).cpu().numpy()
            similarity_scores.extend(sim)

    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")
    return avg_similarity

from sklearn.metrics import silhouette_score

def evaluate_silhouette(embeddings, kmeans):
    cluster_labels = kmeans.labels_
    score = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette Score: {score:.4f}")
    return score




# Example of usage within training loop:
num_epochs = 50
kmeans = None
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
losses = []
accuracies = []

for epoch in range(num_epochs):
    loss, kmeans = train_triplet_loss(model, train_loader, optimizer, epoch, scheduler, kmeans)
    losses.append(loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
    plagiarism_percentage = evaluate_cosine_similarity(model, train_loader)
    
    accuracies.append(plagiarism_percentage)
    # Evaluate after every epoch
    #plagiarism_percentage = evaluate(model, train_loader, kmeans)
    #print(f"Plagiarism Percentage after epoch {epoch+1}: {plagiarism_percentage:.2f}%")


#evaluate(model,train_loader,kmeans)
# Plot Loss & Accuracy

epochs = np.arange(1, len(losses) + 1)
window_size = 5
smoothed_loss = pd.Series(losses).rolling(window=window_size, min_periods=1).mean()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), losses, marker='o', label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Trend")
plt.legend()

plt.subplot(1,2,2)


plt.plot(epochs, smoothed_loss, 'r-', linewidth=2, label="Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()
plt.grid()
plt.show()

plt.show()


    

    
