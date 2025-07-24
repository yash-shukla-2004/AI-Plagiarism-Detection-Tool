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



def load_data_from_folder(folder_path):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".java"):  # You can change this if using different file types
            file_path = os.path.join(folder_path, filename)
            data = load_single_file(file_path)
            if data is not None:
                data_list.append(data)
    return data_list

class TripletDataset(Dataset):
    def __init__(self, original_data, augmented_data, negative_data, transform=None):
        """
        Args:
            original_data: List of original code graph data objects
            augmented_data: List of augmented code graph data objects (plagiarized versions)
            negative_data: List of non-plagiarized code graph data objects (negative samples)
            transform: Optional transform to be applied on a sample
        """
        self.original_data = original_data
        self.augmented_data = augmented_data
        self.negative_data = negative_data
        self.transform = transform

    def __len__(self):
        # Return the number of samples (triplets)
        return len(self.original_data)

    def __getitem__(self, idx):
        # Choose anchor (original), positive (augmented), and negative samples
        anchor = self.original_data[idx]
        positive_idx = idx % len(self.augmented_data)  # Ensuring there's a matching augmented sample
        negative_idx = random.randint(0, len(self.negative_data) - 1)  # Random negative sample
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.negative_data) - 1)
        
        positive = self.augmented_data[positive_idx]
        negative = self.negative_data[negative_idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    
from torch_geometric.data import DataLoader

# Example: Assuming you have pre-loaded your original, augmented, and negative data into corresponding lists
original_data = load_data_from_folder("./testing/original")  # Your original code graphs
augmented_data = load_data_from_folder("./testing/augmented")  # Your augmented (plagiarized) code graphs
negative_data = load_data_from_folder("./testing/original")  # Your non-plagiarized code graphs (negative samples)

# Create the triplet dataset
triplet_dataset = TripletDataset(original_data, augmented_data, negative_data)

# DataLoader for batching triplets
train_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True)

# Now, train_loader will provide batches of (anchor, positive, negative) samples
