import javalang
import torch
import networkx as nx
from torch_geometric.data import Data

def parse_java_code_to_ast(java_code):
    """Parse Java code to AST using javalang."""
    return list(javalang.parse.parse(java_code))

def build_graph_from_ast(ast):
    """Build a graph structure from the AST."""
    G = nx.Graph()
    node_features = []
    node_map = {}

    def add_node_to_graph(node, parent_idx=None):
        """Recursively add nodes to graph and establish parent-child relationships."""
        idx = len(node_features)
        node_features.append([len(str(node))])  # Feature based on the node's string length (can be adjusted)
        node_map[idx] = node

        if parent_idx is not None:
            G.add_edge(parent_idx, idx)  # Add edge from parent to current node

        # Print node type and its children for debugging purposes
        print(f"Node {idx}: {node.__class__.__name__}, Parent: {parent_idx}")

        # If node is a tuple, iterate over its fields
        if isinstance(node, tuple):
            for child in node:
                # Recursively add child nodes if present
                if hasattr(child, 'children') and child.children:
                    add_node_to_graph(child, idx)
        
        # If the node has children (i.e., it's an AST node with sub-nodes), add them
        elif hasattr(node, 'children'):
            for child in node.children:
                add_node_to_graph(child, idx)

    # Start traversal from root of the AST
    for node in ast:
        add_node_to_graph(node)

    return G, torch.tensor(node_features, dtype=torch.float)

def create_pyg_data(graph, node_features):
    """Create a PyTorch Geometric Data object."""
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    batch_size = torch.zeros(node_features.size(0), dtype=torch.long)

    return Data(x=node_features, edge_index=edge_index, batch=batch_size)

# Example Java code
java_code = """
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
"""

# Parse the Java code into an AST
ast = parse_java_code_to_ast(java_code)

# Build a graph from the AST
graph, node_features = build_graph_from_ast(ast)

# Convert to PyTorch Geometric Data object
data = create_pyg_data(graph, node_features)

# Display the results
print("Node Features:", data.x)
print("Edge Index:", data.edge_index)
print("Batch Size:", data.batch)
