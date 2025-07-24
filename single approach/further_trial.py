import os
import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.nn import global_mean_pool, GATConv, GINConv
import numpy as np
from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from trial2 import data_list1
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import random
import os
import os
import torch
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import javalang  # Java parser library
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
import random
import os
from preprocessing import preprocessing

# Function to load a single Java file, parse it, and convert it to a graph
import javalang  # Java parser library
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

# Function to load a single Java file, parse it, and convert it to a graph
import javalang  # Java parser library
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Function to load a single Java file, parse it, and convert it to a graph
def load_single_file(file_path):
    #print(f"Loading file: {file_path}")  # Debug: Track the file being processed

    # Read the Java file content
    try:
        with open(file_path, 'r') as file:
            java_code = file.read()
            java_code = preprocessing(java_code)
        #print(f"Successfully read the file: {file_path}")
    except Exception as e:
        #print(f"Error reading {file_path}: {e}")
        return None  # Return None if the file cannot be read
    
    # Parse Java code into AST using javalang
    try:
        tree = javalang.parse.parse(java_code)
        #print(f"Successfully parsed the Java code from: {file_path}")
    except Exception as e:
        #print(f"Error parsing {file_path}: {e}")
        return None  # Return None if there is a parsing error
    
    # Create a random graph structure (to replace with meaningful logic)
    G = nx.Graph()
    
    # Try to generate nodes and edges from the AST
    node_features = []
    edge_list = []

    def add_node(node_name, node_type):
        """Helper function to add a node to the graph."""
        node_id = len(G.nodes)
        G.add_node(node_id, type=node_type, name=node_name)
        node_features.append(np.random.rand(32))  # Random features for now (replace with actual logic)
        #print(f"Added node: {node_name} of type {node_type}")  # Debug: Track added nodes
        return node_id

    def traverse_ast(node, parent_id=None):
        """Recursively traverse the AST and add nodes and edges."""
        if node is None:
            return  # Skip None nodes

        # Skip string literals and other non-node types
        if isinstance(node, str):
            #print(f"Skipping string literal: {node}")  # Debug: Track string literals
            return

        if isinstance(node, javalang.tree.MethodDeclaration):
            # Add method node and create an edge from the parent (if any)
            method_id = add_node(node.name, "Method")
            if parent_id is not None:
                G.add_edge(parent_id, method_id)
            #print(f"Added method node: {node.name}")  # Debug: Track method nodes
            
            # Traverse method body (if it exists)
            if node.body:
                for element in node.body:
                    traverse_ast(element, method_id)
        
        elif isinstance(node, javalang.tree.LocalVariableDeclaration):
            # Handle local variable declaration (this has 'variables' attribute, which is a list of VariableDeclarators)
            for declarator in node.declarators:
                if hasattr(declarator, 'name'):  # Access the 'name' attribute
                    variable_id = add_node(declarator.name, "Variable")
                    if parent_id is not None:
                        G.add_edge(parent_id, variable_id)
                    #print(f"Added variable node: {declarator.name}")


        
        elif isinstance(node, list):  # Check if node is a list of children
            for child in node:
                traverse_ast(child, parent_id)  # Recursively traverse each child in the list
            return  # Exit after processing the list

        else:
            # Handle other types of nodes (expressions, statements, etc.)
            if hasattr(node, 'children'):  # Ensure 'children' attribute exists
                for child in node.children:
                    traverse_ast(child, parent_id)
            #else:
                #print(f"Skipping unhandled node type: {type(node)}")  # Debug: Track unhandled nodes



    # Start traversal from the root of the AST
    #print("Starting AST traversal...")
    traverse_ast(tree)
    #print(f"Finished AST traversal for: {file_path}")

    # If the graph is empty (could happen if the code is invalid), return None
    if len(G.nodes) == 0:
        #print(f"No valid nodes found for {file_path}, skipping...")
        return None

    # Convert node features to a torch tensor
    x = torch.tensor(node_features, dtype=torch.float)

    # Getting edges and converting to torch tensor
    edge_index = torch.tensor(np.array(list(G.edges)).T, dtype=torch.long)

    # Dummy batch assignment (this is a single graph, so all nodes are in one batch)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    # Create the PyG Data object
    data = Data(x=x, edge_index=edge_index, batch=batch)

    #print(f"Successfully created graph for: {file_path}")  # Debug: Graph creation success
    return data

# Function to load data from a folder and label the graphs
def load_data_from_folder(folder_path, is_augmented=False):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".java"):  # Assuming Java files
            file_path = os.path.join(folder_path, filename)
            data = load_single_file(file_path)
            label = 1 if is_augmented else 0  # 1 for augmented, 0 for original
            if data is not None:
                data.y = torch.tensor([label], dtype=torch.float).view(-1, 1)  # Adding the label to the data
                data_list.append(data)
            else:
                print(f"Skipping file with invalid data: {file_path}")
            
    return data_list

# Function to load both original and augmented data
def load_original_and_augmented_data(original_folder, augmented_folder):
    original_data = load_data_from_folder(original_folder, is_augmented=False)
    augmented_data = load_data_from_folder(augmented_folder, is_augmented=True)
    return original_data + augmented_data  # Combine both datasets

# GNN Model for Plagiarism Detection
class GNNModel(nn.Module):
    def __init__(self, in_channels=32, out_channels=128):  # Change in_channels to 32
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, out_channels, heads=4, dropout=0.6)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(out_channels * 4, 128),  # Adjust input channels based on GAT output
            nn.ReLU(),
            nn.Linear(128, 64)
        ))
        self.fc = nn.Linear(64, 1)  # Output one value for plagiarism percentage

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Example usage: Load data from both original and augmented folders
original_folder = "./testing/original"
augmented_folder = "./testing/augmented2"
data_list = load_original_and_augmented_data(original_folder, augmented_folder)
print(f"Loaded {len(data_list)} graphs.")

# DataLoader setup for training
train_loader = DataLoader(data_list, batch_size=32, shuffle=True)

# Calculate class weights based on the class distribution
#num_original = 105
#num_augmented = 315

# Assuming original files are the minority class
#class_weight = torch.tensor([num_augmented / (num_original + num_augmented)], dtype=torch.float)
#print(class_weight)
# Initialize model, criterion, optimizer
model = GNNModel(in_channels=32, out_channels=128)  # Update to match feature size
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits for plagiarism detection
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
import matplotlib.pyplot as plt
import numpy as np
import sys
# Training Loop
losses = []
accuracies = []
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct =0 
    total =0 
    # Loop over batches
    for data in train_loader:
        #print(data.y[:10])  # Print first 10 elements of target

        x, edge_index, batch = data.x, data.edge_index, data.batch
        output = model(x, edge_index, batch)
        #print(f'Output shape: {output.shape}')
        #print(output)
        # The target labels are stored in data.y, which is 0 for original and 1 for augmented
        target = data.y.view(-1, 1).float()
        #print(f'Target shape: {target.shape}')
        # Calculate loss (binary classification)
        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss for each batch (optional)
        print(f"Loss: {loss.item()}")
        predictions = (torch.sigmoid(output) > 0.5).float()
        correct += (predictions == target).sum().item()
        total += target.size(0)
        total_loss += loss.item()

    # Print average loss for this epoch
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    losses.append(avg_loss)
    accuracies.append(accuracy)
    scheduler.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_loader)}')

# Save the model
# torch.save(model.state_dict(), 'model.pth')
# Smooth function using moving average
def smooth_curve(values, weight=0.9):
    smoothed = []
    last = values[0]
    for val in values:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return smoothed

# Plot loss
plt.figure(figsize=(10, 5))

plt.plot(smooth_curve(losses), label='Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Plot accuracy
plt.figure(figsize=(10, 5))

plt.plot(smooth_curve(accuracies), label='Accuracy', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()
plt.show()
sys.exit()
# Evaluate the model for plagiarism percentage on test data with more than one codes
#def evaluate_plagiarism(model, test_loader):
    #model.eval()  # Set the model to evaluation mode
    #total_plagiarism_score = 0
    #total_samples = 0

    #with torch.no_grad():
        #for data in test_loader:
            #x, edge_index, batch = data.x, data.edge_index, data.batch
            #output = model(x, edge_index, batch)  # Get model output (plagiarism score for each graph in batch)
            
            # If the batch size is more than one, we need to calculate the plagiarism score for each sample
            #plagiarism_scores = torch.sigmoid(output).squeeze()  # Apply sigmoid and remove extra dimensions

            # Add the plagiarism scores to the total
            #total_plagiarism_score += plagiarism_scores.sum().item()  # Summing the scores for each graph in the batch
            #total_samples += len(plagiarism_scores)  # Count the number of samples in the batch

    # Calculate average plagiarism score (percentage)
    #avg_plagiarism_percentage = (total_plagiarism_score / total_samples) * 100
    #rint(f"Average Plagiarism Percentage: {avg_plagiarism_percentage:.2f}%")


def evaluate_plagiarism(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    total_plagiarism_score = 0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            output = model(x, edge_index, batch)  # Get model output (plagiarism score for each graph in batch)
            #print(output)
            # Apply sigmoid activation to get probabilities between 0 and 1
            plagiarism_scores = torch.sigmoid(output).squeeze()  # Apply sigmoid and remove extra dimensions
            #print(plagiarism_scores)
            # Handle the case where plagiarism_scores is scalar (single sample in the batch)
            if plagiarism_scores.dim() == 0:
                plagiarism_scores = plagiarism_scores.unsqueeze(0)  # Convert scalar to 1D tensor

            # Add the plagiarism scores to the total
            total_plagiarism_score += plagiarism_scores.sum().item()  # Summing the scores for each graph in the batch
            total_samples += len(plagiarism_scores)  # Count the number of samples in the batch

    # Calculate average plagiarism score (percentage)
    avg_plagiarism_percentage = (total_plagiarism_score / total_samples) * 100
    print(f"Average Plagiarism Percentage: {avg_plagiarism_percentage:.2f}%")


# Example: Evaluate on test data (replace with actual test loader)

print(f"Loaded {len(data_list1)} graphs.")

# DataLoader setup for training
test_loader = DataLoader(data_list1, batch_size=32, shuffle=True)
evaluate_plagiarism(model, test_loader)
