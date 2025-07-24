import torch
import javalang
from preprocessing import preprocessing

# Function to extract basic features from the AST
def extract_features(java_code):
    try:
        processed_code = preprocessing(java_code)
        tree = processed_code["inbuilt_tokens_ast"]
        #print(f"AST Tree: {tree}")
        
        # Extract features (for simplicity, let's take node type and some other basic features)
        features = []
        edges = create_edges_from_ast(tree, features)
        #print(f"Edge Index: {edges}")
        return features, edges
    except javalang.parser.JavaSyntaxError as e:
        #print(f"Java syntax error during AST parsing: {e}")
        return [(0, 0)], []  # Fallback value
    except Exception as e:
        #print(f"Error during AST parsing: {e}")
        return [(0, 0)], []  # Fallback value

# Function to create edges from the AST, also collects features
def create_edges_from_ast(ast_tree, features):
    edges = []
    node_indices = {i: node for i, node in enumerate(ast_tree.types[0].body)}  # Flatten body nodes

    #print(f"\n--- Edge Creation Start ---")

    for i, node in node_indices.items():
        #print(f"Inspecting node {i}: {type(node).__name__}")
        
        # Simple feature extraction
        feature_vector = []
        if isinstance(node, javalang.tree.MethodDeclaration):
            feature_vector = [1, len(node.name)]  # MethodDeclaration type and method name length
        elif isinstance(node, javalang.tree.LocalVariableDeclaration):
            feature_vector = [2, len(node.type.name)]  # LocalVariableDeclaration type and variable type length
        elif isinstance(node, javalang.tree.MethodInvocation):
            feature_vector = [3, len(node.member)]  # MethodInvocation type and method name length
        elif isinstance(node, (javalang.tree.IfStatement, javalang.tree.WhileStatement)):
            feature_vector = [4, 0]  # Control flow type (If/While)

        features.append(feature_vector)  # Add features for the node

        try:
            if isinstance(node, javalang.tree.MethodDeclaration):
                for subnode in node.body:
                    if isinstance(subnode, javalang.tree.StatementExpression):
                        edges.append((i, len(node_indices) + len(edges)))  # Add edge
                    if isinstance(subnode, javalang.tree.LocalVariableDeclaration):
                        for declarator in subnode.declarators:
                            edges.append((i, len(node_indices) + 1))

            elif isinstance(node, javalang.tree.MethodInvocation):
                method_name = node.member
                for j, other_node in node_indices.items():
                    if isinstance(other_node, javalang.tree.MethodDeclaration) and other_node.name == method_name:
                        edges.append((i, j))

            elif isinstance(node, (javalang.tree.IfStatement, javalang.tree.WhileStatement)):
                edges.append((i, i + 1))

        except Exception as e:
            #print(f"Error processing node {i}: {e}")
            edges.append((i, -1))  # Placeholder edge indicating an error

    #print(f"--- Edge Creation End ---\n")
    return edges

# Example usage with multiple Java codes (graphs)
java_codes = [
    "public class Main { public static void main(String[] args) { int a = 10; int b = 20; int sum = a + b; System.out.println(sum); } }"
]

# Initialize lists to store results for multiple graphs
def preprocessing_with_features(java_codes):
    all_features = []
    all_edge_indices = []
    all_batch_indices = []

    for i, java_code in enumerate(java_codes):
        features, edges = extract_features(java_code)

        # Debugging: Check if features and edges are returned correctly
        #print(f"Code {i}: Features = {features}, Edges = {edges}")

        # Ensure features are not empty or malformed
        features = [f if isinstance(f, list) and len(f) > 0 else [0, 0] for f in features]  # Default feature [0, 0] if empty

        if len(features) == 0 or all(len(f) == 0 for f in features):
            print(f"Warning: No valid features for code {i}. Using default feature values.")
            node_features = torch.zeros(1, 2)  # Default feature vector with 2 values (adjust as necessary)
        else:
            node_features = torch.tensor(features, dtype=torch.float).unsqueeze(0)

        # Handle edges similarly, ensuring it's non-empty and valid
        if not edges or len(edges) == 0:
            print(f"Warning: No valid edges for code {i}. Using default edge index values.")
            edge_index = torch.zeros(2, 0, dtype=torch.long)  # Default empty edge index
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Add batch index (for multiple graphs, each graph gets a different batch index)
        batch_index = torch.tensor([i] * node_features.size(1), dtype=torch.long)  # One batch index per node

        all_features.append(node_features)
        all_edge_indices.append(edge_index)
        all_batch_indices.append(batch_index)

    # Combine all graphs into one batch (if needed)
    x = torch.cat(all_features, dim=1)  # Concatenate all feature tensors along the node dimension
    edge_index = torch.cat(all_edge_indices, dim=1)  # Concatenate all edge_index tensors along the edge dimension
    batch_index = torch.cat(all_batch_indices, dim=0)  # Concatenate batch indices

    return x, edge_index, batch_index


preprocessing_with_features(java_codes)