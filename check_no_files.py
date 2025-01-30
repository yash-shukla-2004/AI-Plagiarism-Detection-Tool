import os

def count_files_in_folder(directory):
    for foldername, subfolders, filenames in os.walk(directory):
        print(f'Folder: {foldername}, Files: {len(filenames)}')

# Replace 'your_directory_path' with the path of your folder
count_files_in_folder('./dataset')
