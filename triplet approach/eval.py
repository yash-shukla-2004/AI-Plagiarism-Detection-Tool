import os
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
from torch_geometric.nn import global_mean_pool
from old_app2 import model
from old_app import load_data_from_folder,train_loader
import torch.nn.functional as F
from utils import save_model

#def load_single_file(file_path):
    #"""
    #Function to load a single code file and convert it to graph data.
    #In a real scenario, replace this with logic to parse the code into graph data (e.g., AST or CFG).
    #"""
    # Dummy graph generation (replace with actual logic for converting code to graph)
    #G = nx.erdos_renyi_graph(n=100, p=0.1)  # Example: random graph (replace with actual logic)
    #node_features = np.random.rand(G.number_of_nodes(), 32)  # Example: 32 features per node
    #x = torch.tensor(node_features, dtype=torch.float)
    #edge_index = torch.tensor(np.array(list(G.edges)).T, dtype=torch.long)
    #batch = torch.zeros(x.size(0), dtype=torch.long)
    
    #data = Data(x=x, edge_index=edge_index, batch=batch)
    #return data



    

def get_plagiarism_percentage(model, anchor_data, suspect_data, margin=1.0):
    """
    Determines plagiarism percentage based on the trained triplet-loss GNN model.
    
    Parameters:
    - model: The trained GNN model.
    - anchor_data: The original code's graph data.
    - suspect_data: The code to be evaluated for plagiarism.
    - margin: The margin used in triplet loss during training.
    
    Returns:
    - Plagiarism percentage (0 to 100%).
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings for anchor and suspect
        anchor_emb = model(anchor_data.x, anchor_data.edge_index, anchor_data.batch)
        suspect_emb = model(suspect_data.x, suspect_data.edge_index, suspect_data.batch)
        
        # Average embeddings if batched
        if anchor_emb.dim() > 1:
            anchor_emb = anchor_emb.mean(dim=0)
        if suspect_emb.dim() > 1:
            suspect_emb = suspect_emb.mean(dim=0)
        
        # Euclidean distance (same as triplet loss distance)
        distance = F.pairwise_distance(anchor_emb.unsqueeze(0), suspect_emb.unsqueeze(0)).item()
        
        # Convert distance to plagiarism percentage
        similarity_score = max(0, (margin - distance) / margin)  # Normalize to [0,1]
        plagiarism_percentage = similarity_score * 100  # Scale to [0,100]
    
    return plagiarism_percentage

# Example usage:
anchor_data,_,_ = next(iter(train_loader))
file_path = './testing/test'  # Path to your code file
data = load_data_from_folder(file_path)

plagiarism_percentage = get_plagiarism_percentage(model,anchor_data, data[0])
print(f"Plagiarism Percentage: {plagiarism_percentage:.2f}%")
save_model(model, 'plagiarism_detector_model_1.pth')
print("Model Saved.")