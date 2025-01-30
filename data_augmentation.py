import os
import random
import re
import javalang
from typing import List

def rename_variables(java_code: str) -> str:
    tree = javalang.parse.parse(java_code)
    var_map = {}
    counter = 0
    
    for path, node in tree:
        if isinstance(node, javalang.tree.VariableDeclarator):
            if node.name not in var_map:
                new_name = f'var{counter}'
                var_map[node.name] = new_name
                counter += 1
    
    for old_name, new_name in var_map.items():
        java_code = re.sub(rf'\b{old_name}\b', new_name, java_code)
    
    return java_code

def reorder_statements(java_code: str) -> str:
    statements = java_code.split(";")
    if len(statements) > 2:
        random.shuffle(statements[:-1])  # Avoid shuffling the last empty statement
    return ";".join(statements)

def change_whitespace(java_code: str) -> str:
    java_code = re.sub(r'\s+', ' ', java_code)  # Reduce multiple spaces to single
    java_code = java_code.replace('{', '{\n').replace('}', '\n}')
    return java_code

def augment_java_code(java_code: str) -> str:
    transformations = [rename_variables, reorder_statements, change_whitespace]
    random.shuffle(transformations)
    for transform in transformations:
        java_code = transform(java_code)
    return java_code

def augment_dataset(input_folder: str, output_folder: str, num_variants: int = 3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    processed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".java"):
            try:
                with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as f:
                    original_code = f.read()
                
                for i in range(num_variants):
                    augmented_code = augment_java_code(original_code)
                    new_filename = f"{os.path.splitext(filename)[0]}_aug{i}.java"
                    with open(os.path.join(output_folder, new_filename), "w", encoding="utf-8") as f:
                        f.write(augmented_code)
                
                processed_count += 1
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                skipped_count += 1
    
    print(f"Processing completed. Files processed: {processed_count}, Files skipped: {skipped_count}")

if __name__ == "__main__":
    input_folder = "./dataset/original"
    output_folder = "./dataset/augmented"
    augment_dataset(input_folder, output_folder)
