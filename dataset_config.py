from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Data
import os
from gnn_feature_extraction import preprocessing_with_features
from config import batch_size, dataset_testing_path, dataset_testing_path_aug


class JavaCodeTripletDataset(Dataset):
    def __init__(self, original_folder, augmented_folder):
        # Initialize the dataset by listing original and augmented files
        self.original_files = sorted([os.path.join(original_folder, f) for f in os.listdir(original_folder)])
        self.augmented_files = {
            file: [os.path.join(augmented_folder, f"{os.path.basename(file)[:-5]}_aug{i}.java") 
                   for i in range(3)] 
            for file in self.original_files
        }

    def __len__(self):
        return len(self.original_files)

    def __getitem__(self, idx):
        # Load original (anchor)
        with open(self.original_files[idx], 'r') as f:
            anchor_code = f.read()
        anchor_features = preprocessing_with_features(anchor_code)

        # Load one augmented version (positive sample)
        aug_file = self.augmented_files[self.original_files[idx]][torch.randint(0, 3, (1,)).item()]
        with open(aug_file, 'r') as f:
            positive_code = f.read()
        positive_features = preprocessing_with_features(positive_code)

        # Select a random unrelated original (negative sample)
        neg_idx = torch.randint(0, len(self.original_files), (1,)).item()
        while neg_idx == idx:  # Ensure it's not the same as anchor
            neg_idx = torch.randint(0, len(self.original_files), (1,)).item()
        with open(self.original_files[neg_idx], 'r') as f:
            negative_code = f.read()
        negative_features = preprocessing_with_features(negative_code)

        # Convert to tensors
        def to_tensor(features):
            # Concatenate all feature types (lexical, AST, semantic)
            lexical = torch.tensor(features["lexical"]["tokens"], dtype=torch.float)
            ast = torch.tensor(features["ast"]["node_types"], dtype=torch.float)
            semantic = torch.tensor(features["semantic"]["variable_usage"], dtype=torch.float)  # Example of semantic
            
            # You can add all other components of the feature dictionaries in a similar manner
            # For example, if you want to concatenate other semantic features like data_flow, control_flow, etc.
            all_features = torch.cat((lexical, ast, semantic), dim=0)
            
            return all_features
        
        anchor = to_tensor(anchor_features)
        positive = to_tensor(positive_features)
        negative = to_tensor(negative_features)

        # Returning anchor, positive, and negative as a tuple
        return anchor, positive, negative
    

triplet_dataset = JavaCodeTripletDataset(dataset_testing_path,dataset_testing_path_aug)
triplet_loader = DataLoader(triplet_dataset,batch_size=batch_size,shuffle=True)


print(len(triplet_dataset))

    



