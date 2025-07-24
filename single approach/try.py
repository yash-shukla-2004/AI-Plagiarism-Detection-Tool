import os
import shutil

def move_files(src_dir, dest_dir):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Loop through the files in the source directory
    for filename in os.listdir(src_dir):
        if filename.endswith('1.java'):
            # Full path of the file
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            # Move the file to the destination directory
            shutil.move(src_file, dest_file)
            print(f"Moved: {filename}")

# Example usage
src_dir = "./testing/augmented"
dest_dir = "./testing/augmented2"
move_files(src_dir, dest_dir)
