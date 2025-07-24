from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

import torch
import os
from gnn_feature_extraction import preprocessing_with_features
from config import batch_size, dataset_testing_path, dataset_testing_path_aug
from torch_geometric.data import Data

class JavaCodeTripletDataset(Dataset):
    def __init__(self, original_folder, augmented_folder):
        self.original_files = sorted([os.path.join(original_folder, f) for f in os.listdir(original_folder) if f.endswith('.java')])
        self.augmented_files = {
            file: [os.path.join(augmented_folder, f"{os.path.basename(file)[:-5]}_aug_{i}.java") 
                   for i in range(1, 4)] 
            for file in self.original_files
        }

    def __len__(self):
        return len(self.original_files)

    def __getitem__(self, idx):
        # Load original (anchor)
        with open(self.original_files[idx], 'r') as f:
            anchor_code = f.read()
        anchor_x, anchor_edge_index, anchor_batch_index = preprocessing_with_features([anchor_code])

        # Load all three augmented versions (positive samples)
        positives = []
        for aug_file in self.augmented_files[self.original_files[idx]]:
            if not os.path.exists(aug_file):  # Check if augmented file exists
                print(f"Warning: Augmented file {aug_file} not found. Using default values.")
                # Default values in case the file is missing
                pos_x = torch.zeros(1, 2)  # Default feature vector with 2 values (adjust as needed)
                pos_edge_index = torch.zeros(2, 0, dtype=torch.long)  # Default empty edge index
                pos_batch_index = torch.tensor([0], dtype=torch.long)  # Default batch index
            else:
                with open(aug_file, 'r') as f:
                    positive_code = f.read()
                pos_x, pos_edge_index, pos_batch_index = preprocessing_with_features([positive_code])

            # Wrap the positive sample as a Data object
            positive_data = Data(x=pos_x, edge_index=pos_edge_index, batch=pos_batch_index)
            positives.append(positive_data)

        # Select a random unrelated original (negative sample)
        neg_idx = torch.randint(0, len(self.original_files), (1,)).item()
        while neg_idx == idx:
            neg_idx = torch.randint(0, len(self.original_files), (1,)).item()
        with open(self.original_files[neg_idx], 'r') as f:
            negative_code = f.read()
        neg_x, neg_edge_index, neg_batch_index = preprocessing_with_features([negative_code])

        # Wrap the negative sample as a Data object
        negative_data = Data(x=neg_x, edge_index=neg_edge_index, batch=neg_batch_index)

        # Wrap the anchor as a Data object
        anchor_data = Data(x=anchor_x, edge_index=anchor_edge_index, batch=anchor_batch_index)

        return anchor_data, positives, negative_data


#def collate_fn(batch):
    # Extract anchors, positives, and negatives from the batch
    #anchors, positives_list, negatives = zip(*batch)

    # Create lists for each element (anchor, positives, negatives) and wrap in Data objects
    #anchor_batch = [anchor for anchor in anchors]
    #positive_batch = [positive for positive_list in positives_list for positive in positive_list]
    #negative_batch = [negative for negative in negatives]

    # Return the batches as lists of Data objects
    #return anchor_batch, positive_batch, negative_batch



def collate_fn(batch):
    # batch is a list of tuples (anchor_data, positives, negative_data)
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)


triplet_dataset = JavaCodeTripletDataset(dataset_testing_path, dataset_testing_path_aug)
#triplet_loader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

triplet_loader = DataLoader(triplet_dataset, batch_size=32, shuffle=True,collate_fn = collate_fn)

print(len(triplet_dataset))

#sample_idx = 0  # Change index to check different samples
#anchor, positives, negative = triplet_dataset[sample_idx]

#print("Anchor sample:")
#print(anchor)

#print("\nPositive samples:")
#for pos in positives:
    #print(pos)

#print("\nNegative sample:")
#print(negative)
