import os
import re
import random
import javalang

def extract_identifiers(code: str) -> set:
    tokens = list(javalang.tokenizer.tokenize(code))
    identifiers = set()
    for token in tokens:
        if isinstance(token, javalang.tokenizer.Identifier):
            identifiers.add(token.value)
    return identifiers

def augment_java_code(code: str) -> str:
    identifiers = extract_identifiers(code)

    def randomize_whitespace(match):
        spaces = ' ' * random.randint(1, 4)
        return spaces

    code_lines = code.splitlines()
    augmented_lines = []

    for line in code_lines:
        if not line.strip():
            augmented_lines.append('')
            continue

        line_aug = re.sub(r'\s+', randomize_whitespace, line)

        for identifier in identifiers:
            if identifier not in {'main', 'System', 'out', 'println'}:
                line_aug = re.sub(rf'\b{identifier}\b', f'{identifier}_aug', line_aug)

        augmented_lines.append(line_aug)

    return '\n'.join(augmented_lines)

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
                    output_filename = f"{filename[:-5]}_aug_{i+1}.java"
                    with open(os.path.join(output_folder, output_filename), "w", encoding="utf-8") as out_f:
                        out_f.write(augmented_code)

                processed_count += 1
            except Exception as e:
                skipped_count += 1
                print(f"Skipped {filename} due to error: {e}")

    print(f"Processed {processed_count} files, skipped {skipped_count} files.")

if __name__ == "__main__":
    input_folder = "./testing/original"
    output_folder = "./testing/augmented"
    augment_dataset(input_folder, output_folder)
