import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torch_geometric
from torch_geometric.data import Data
from kmeans import model
from kmeans import kmeans
import os
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch_geometric.nn import global_mean_pool
from old_app import load_data_from_folder,train_loader

import seaborn as sns
import matplotlib.pyplot as plt

# Assume that you have the following functions defined somewhere:
# load_single_file(file_path) to load a single file's graph data.
# The model is already trained.

# New evaluation function to evaluate the plagiarism percentage of a single file
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from scipy.spatial.distance import cdist

import numpy as np
import torch
from scipy.spatial.distance import cdist

import numpy as np
import torch
from scipy.spatial.distance import cdist

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity



class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(ProjectionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Dictionary of projection layers for different embedding sizes
projection_layers = {
    2048: ProjectionLayer(2048, 64),
    512: ProjectionLayer(512, 64),
    64: nn.Identity()  
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for proj in projection_layers.values():
    proj.to(device)

def project_embedding(embedding):
    """Projects embeddings to a fixed size of 64 if needed."""
    embedding_dim = embedding.shape[-1]
    if embedding_dim in projection_layers:
        proj_layer = projection_layers[embedding_dim]
        embedding = proj_layer(torch.tensor(embedding, dtype=torch.float32).to(device))
        return embedding.detach().cpu().numpy().reshape(1, -1)
    return None  

def evaluate_file_with_kmeans(model, file_data, kmeans, train_loader):
    """
    Evaluate the plagiarism percentage of a given file using trained K-means clusters.
    
    Parameters:
    - model: Trained GNN model
    - file_data: Graph data of the file to evaluate
    - kmeans: Trained KMeans model
    - train_loader: DataLoader containing (anchor, positive, negative) tuples
    
    Returns:
    - Plagiarism percentage (0-100)
    """
    model.eval()
    
    # Compute file embedding
    file_embed = model(file_data.x, file_data.edge_index, file_data.batch)
    file_embed = file_embed.detach().cpu().numpy().reshape(1, -1)
    file_embed = project_embedding(file_embed)
    
    if file_embed is None:
        print("Error: Unsupported file embedding size.")
        return 0

    known_embeddings = []
    known_labels = []
    all_sims = []
    for (anchor, pos, neg) in train_loader:
        for sample in [anchor, pos, neg]:
            known_embed = model(sample.x, sample.edge_index, sample.batch)
            known_embed = known_embed.detach().cpu().numpy().reshape(1, -1)
            known_embed = project_embedding(known_embed)

            if known_embed is None:
                continue  

            known_embeddings.append(known_embed)
            all_sims.extend(F.cosine_similarity())
            known_labels.append(kmeans.predict(known_embed)[0])  # Get assigned cluster label

    if not known_embeddings:
        print("No valid known embeddings found. Returning 0% plagiarism.")
        return 0  

    known_embeddings = np.vstack(known_embeddings)  # Ensure correct shape

    # Find closest K-means cluster
    cluster_distances = cdist(file_embed, kmeans.cluster_centers_, metric="euclidean")
    file_cluster = np.argmin(cluster_distances)

    # Extract samples from the same cluster
    cluster_samples = [i for i in range(len(known_labels)) if known_labels[i] == file_cluster]

    if not cluster_samples:
        print("No similar files found in the same cluster. Returning 0% plagiarism.")
        return 0  
    sns.histplot(all_sims, bins=20, kde=True)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Cosine Similarity Distribution")
    plt.show()
    # Compute cosine similarity with clustered embeddings
    cluster_embeds = np.array([known_embeddings[i] for i in cluster_samples])
    similarities = cosine_similarity(file_embed, cluster_embeds).flatten()
    
    plagiarism_percentage = np.mean(similarities) * 100  

    return max(0, min(100, plagiarism_percentage))  # Ensure valid range


# Example usage:

# Assume that you have already trained the K-means model using the embeddings during the training phase
# and loaded the model as well.
# kmeans = KMeans(n_clusters=2, random_state=42)  # Trained K-means model
# model = ...  # Your trained GNN model

# Evaluate a file for plagiarism using the trained model and K-means
file_path = "./testing/test"  # Replace with the actual test file
data = load_data_from_folder(file_path)
plagiarism_percentage = evaluate_file_with_kmeans(model, data[2], kmeans,train_loader)
#plagiarism_percentage = get_plagiarism_percentage(model,train_loader, data[1])
print(f"Plagiarism Percentage: {plagiarism_percentage:.2f}%")
