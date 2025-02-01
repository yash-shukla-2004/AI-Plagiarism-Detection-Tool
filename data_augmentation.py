import os
import random
import re
import javalang
from typing import List

def rename_variables(java_code: str) -> str:
    java_keywords = {"abstract", "assert", "boolean", "break", "byte", "case", "catch", "char", "class", "const", "continue", "default", "do", "double", "else", "enum", "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof", "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return", "short", "static", "strictfp", "super", "switch", "synchronized", "this", "throw", "throws", "transient", "try", "void", "volatile", "while"}
    standard_methods = {"System", "out", "println", "print", "err"}
    
    try:
        tree = javalang.parse.parse(java_code)
    except javalang.parser.JavaSyntaxError as e:
        print(f"Syntax error in Java code, skipping renaming: {e}")
        return java_code
    
    var_map = {}
    counter = 0
    
    for path, node in tree:
        if isinstance(node, javalang.tree.VariableDeclarator):
            if node.name not in var_map and node.name not in java_keywords and node.name not in standard_methods:
                new_name = f'var{counter}'
                var_map[node.name] = new_name
                counter += 1
    
    for old_name, new_name in var_map.items():
        java_code = re.sub(rf'\b{old_name}\b', new_name, java_code)
    
    return java_code

def reorder_statements(java_code: str) -> str:
    import_section = []
    code_section = []
    method_body = []
    inside_method = False

    lines = java_code.split("\n")
    
    for line in lines:
        stripped = line.strip()

        if stripped.startswith("import "):
            import_section.append(line)
        elif re.match(r"^(public|private|protected|static|\s)*\s*(class|interface|enum|void|\w+\s+\w+)\s*\(", stripped):
            # Preserve method/class headers
            code_section.append(line)
            inside_method = True
        elif inside_method and stripped == "}":
            # Closing method/class - append stored method body
            random.shuffle(method_body)
            code_section.extend(method_body)
            code_section.append(line)
            method_body = []
            inside_method = False
        elif inside_method:
            # Store method body for shuffling
            method_body.append(line)
        else:
            # Outside method, directly add
            code_section.append(line)

    # Join everything back together
    return "\n".join(import_section + code_section)

def change_whitespace(java_code: str) -> str:
    java_code = re.sub(r'\s+', ' ', java_code)  # Reduce multiple spaces to single
    java_code = java_code.replace('{', '{\n').replace('}', '\n}')
    return java_code

def augment_java_code(java_code: str) -> str:
    transformations = [rename_variables, reorder_statements, change_whitespace]
    random.shuffle(transformations)
    for transform in transformations:
        try:
            java_code = transform(java_code)
        except Exception as e:
            print(f"Error applying {transform.__name__}: {e}")
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
                    if augmented_code.strip():  # Ensure the output is valid
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
