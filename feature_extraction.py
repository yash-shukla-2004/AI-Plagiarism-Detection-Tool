from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from preprocessing import preprocessing,java_code, inbuilt_ast
import re
import javalang
from collections import defaultdict


def lexical_features(tokens):
    """
    Extracts comprehensive lexical features from the given tokens.
    """
    # Token abstraction
    abstracted_tokens = []
    identifier_map = defaultdict(lambda: f"var_{len(identifier_map) + 1}")
    literal_map = defaultdict(lambda: f"literal_{len(literal_map) + 1}")
    string_literals = []

    for token in tokens:
        # Abstract identifiers and literals
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", token):  # Identifier pattern
            abstracted_tokens.append(identifier_map[token])
        elif re.match(r"^\".*\"$", token) or re.match(r"^\d+$", token):  # String or integer literal
            abstracted_tokens.append(literal_map[token])
            if re.match(r"^\".*\"$", token):  # String literal
                string_literals.append(token)
        else:
            abstracted_tokens.append(token)

    # Token frequency (TF-IDF)
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Adjust n-grams as needed
    token_str = " ".join(tokens)
    tfidf_matrix = tfidf_vectorizer.fit_transform([token_str]).toarray()[0]
    token_freq = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_matrix))

    # Length-based features
    num_tokens = len(tokens)
    num_characters = sum(len(token) for token in tokens)

    # Lexical diversity
    unique_tokens = len(set(tokens))
    diversity = unique_tokens / num_tokens if num_tokens > 0 else 0

    # Whitespace patterns
    spaces = len(re.findall(r' {4}', token_str))  # Count 4-space indentation
    tabs = len(re.findall(r'\t', token_str))  # Count tab-based indentation

    # Consolidate features
    return {
        "tokens": tokens,
        "abstracted_tokens": abstracted_tokens,
        "token_freq": token_freq,
        "length_features": {
            "num_tokens": num_tokens,
            "num_characters": num_characters
        },
        "diversity": diversity,
        "whitespace_patterns": {
            "spaces": spaces,
            "tabs": tabs
        },
        "string_literals": string_literals
    }


def ast_features(tree):
    """
    Extracts AST-based features such as node types, structural relationships, paths, and subtree patterns.

    Args:
        tree (javalang.tree.CompilationUnit): The parsed AST tree.

    Returns:
        dict: Extracted AST features including node types, structural relationships, paths, and others.
    """
    if not tree:
        return {
            "node_types": {},
            "parent_child_relationships": [],
            "subtrees": [],
            "paths": [],
            "leaf_nodes": [],
            "max_depth": 0,
            "tree_width": 0
        }

    # Helper function to traverse the AST and extract features
    def traverse_ast(node, depth=0, parent=None):
        nonlocal max_depth, parent_child_relationships, nodes, subtrees, paths, leaf_nodes
        if not isinstance(node, javalang.ast.Node):
            return
        
        # Update max depth
        max_depth = max(max_depth, depth)
        
        # Record node type
        node_type = type(node).__name__
        nodes.append(node_type)
        
        # Record leaf nodes (nodes with no children)
        if not node.children or len(node.children) == 0:
            leaf_nodes.append(node)
        
        # Record parent-child relationships
        if parent:
            parent_child_relationships.append((type(parent).__name__, node_type))
        
        # Ensure node.children is a valid iterable
        if node.children and isinstance(node.children, (list, tuple)):
            for child in node.children:
                if isinstance(child, list):
                    # If child is a list, recurse on each sub-child
                    for sub_child in child:
                        traverse_ast(sub_child, depth + 1, node)
                elif isinstance(child, javalang.ast.Node):
                    # Recurse on individual child node
                    traverse_ast(child, depth + 1, node)


    # Initialize variables for feature extraction
    nodes = []
    parent_child_relationships = []
    max_depth = 0
    leaf_nodes = []
    subtrees = []
    paths = []

    # Traverse the AST
    traverse_ast(tree)
    
    # Compute tree width (maximum number of children at any level)
    def calculate_width(node):
        if not isinstance(node, javalang.ast.Node) or not hasattr(node, "children"):
            return 0
        try:
            return max(
                len([child for child in node.children if isinstance(child, javalang.ast.Node)]),
                *(calculate_width(child) for child in node.children if isinstance(child, javalang.ast.Node))
            )
        except TypeError:
        # Handle case where node.children is None or not iterable
            return 0


    tree_width = calculate_width(tree)

    # Compute subtree patterns
    subtree_patterns = {}
    for path, node in tree.filter(javalang.ast.Node):
        subtree = []
        for subpath, subnode in node.filter(javalang.ast.Node):
            subtree.append(type(subnode).__name__)
        subtree = tuple(subtree)
        if subtree in subtree_patterns:
            subtree_patterns[subtree] += 1
        else:
            subtree_patterns[subtree] = 1

    # Compute paths (placeholder for advanced path-based representation like code2vec)
    # Actual implementation would involve path extraction algorithms for ASTs
    paths = ["path_1_placeholder", "path_2_placeholder"]

    # Return extracted features
    return {
        "node_types": {node: nodes.count(node) for node in set(nodes)},
        "parent_child_relationships": parent_child_relationships,
        "subtrees": subtree_patterns,
        "paths": paths,
        "leaf_nodes": [type(node).__name__ for node in leaf_nodes],
        "max_depth": max_depth,
        "tree_width": tree_width
    }


def semantic_features(tokens, ast):
    """
    Extracts semantic features such as variable usage, data flow, control flow, method calls,
    code patterns, and function-level features based on tokens and the AST.

    Args:
        tokens (list): List of tokens extracted from the code.
        ast (dict): Abstract Syntax Tree (AST) representation of the code.

    Returns:
        dict: Dictionary containing extracted semantic features.
    """
    # Variable Usage
    variable_usage = {
        "declared_variables": set(),
        "used_variables": set(),
        "dependency_graph": {}  # Mapping of variable usage relationships
    }
    for node in ast.get("VariableDeclarator", []):
        variable_usage["declared_variables"].add(node["name"])
    for node in ast.get("MemberReference", []):
        variable_usage["used_variables"].add(node["name"])
    variable_usage["dependency_graph"] = {
        var: [
            ref["line"] for ref in ast.get("MemberReference", []) if ref["name"] == var
        ]
        for var in variable_usage["declared_variables"]
    }

    # Data Flow
    data_flow = {
        "assignments": [
            (assign["left"], assign["right"]) for assign in ast.get("Assignment", [])
        ],
        "propagation": [
            {"source": flow["source"], "target": flow["target"]}
            for flow in ast.get("DataFlow", [])
        ]
    }

    # Control Flow
    control_flow = {
        "conditions": [node for node in ast.get("IfStatement", [])],
        "branches": len(ast.get("IfStatement", [])),
        "loops": len(ast.get("ForStatement", [])) + len(ast.get("WhileStatement", []))
    }

    # Method Calls
    method_calls = [
        {"method_name": call["name"], "args": call["args"]}
        for call in ast.get("MethodInvocation", [])
    ]

    # Code Patterns
    code_patterns = {
        "recursion": any(
            call["name"] in ast.get("MethodDeclaration", {}).get("name", "")
            for call in ast.get("MethodInvocation", [])
        ),
        "sorting_algorithms": any(
            pattern in tokens
            for pattern in ["sort", "sorted", "quicksort", "mergesort"]
        )
    }

    # Function-Level Features
    function_features = [
        {
            "name": func["name"],
            "args": func["args"],
            "return_type": func["returnType"],
            "calls": [
                call["name"]
                for call in ast.get("MethodInvocation", [])
                if call.get("parent") == func["name"]
            ]
        }
        for func in ast.get("MethodDeclaration", [])
    ]

    # Consolidate features
    return {
        "variable_usage": variable_usage,
        "data_flow": data_flow,
        "control_flow": control_flow,
        "method_calls": method_calls,
        "code_patterns": code_patterns,
        "function_features": function_features
    }


def feature_extraction(preprocessed_data):
    """
    Extracts lexical, syntactic, and semantic features from the preprocessed data.
    """
    tokens = preprocessed_data["tokens"]
    ast_tree = preprocessed_data["inbuilt_tokens_ast"]
    ast = inbuilt_ast(java_code)

    # Lexical features
    lexical = lexical_features(tokens)

    # AST-based features
    ast = ast_features(ast)

    # Semantic embeddings
    semantic = semantic_features(tokens,ast)

    return {
        "lexical": lexical,
        "ast": ast,
        "semantic": semantic
    }    




def preprocessing_with_features(code):
    """
    Preprocessing pipeline with feature extraction.
    """
    preprocessed = preprocessing(code)
    features = feature_extraction(preprocessed)

    print("Lexical Features:")
    print(features["lexical"])
    print("\nAST Features:")
    print(features["ast"])
    print("\nSemantic Features (embeddings shape):")
    print(features["semantic"])

    return features


preprocessing_with_features(java_code)