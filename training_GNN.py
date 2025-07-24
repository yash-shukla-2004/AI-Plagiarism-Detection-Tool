import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GINConv,global_mean_pool
from dataset_config import triplet_loader #dataset
from config import in_channels,out_channels,hidden_channels,lr,num_epochs,margin

#define hyper-parameters
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GNNModel(nn.Module):

    def __init__(self,in_channels, hidden_channels, out_channels):
        super(GNNModel,self).__init__()

        self.gat = GATConv(in_channels,hidden_channels,heads=8,concat=True)

        gin_nn = nn.Sequential(
            nn.Linear(hidden_channels*4,hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels,hidden_channels))
        self.gin = GINConv(gin_nn)

        self.fc = nn.Linear(hidden_channels*4,out_channels)

    def forward(self,x,edge_index,batch):
        out = F.relu(self.gat(x,edge_index))
        out = F.relu(self.gin(out,edge_index))
        out = global_mean_pool(out,batch)
        embeddings = self.fc(out)

        return embeddings
    


#optimiser,model and criterion
model = GNNModel(in_channels,hidden_channels,out_channels)
criterion = nn.TripletMarginLoss(margin= margin, p=2)
optimizer = optim.Adam(model.parameters(), lr=lr)

#training function
import torch
import torch.optim as optim
from torch_geometric.data import Data
import torch.nn.functional as F

from torch_geometric.data import DataLoader

def train(model, triplet_loader, optimizer, criterion, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, data in enumerate(triplet_loader):
            # If data is a tuple, unpack it correctly
            if isinstance(data, tuple):
                anchor_data_list, positive_data_list, negative_data_list = data
            else:
                anchor_data_list = positive_data_list = negative_data_list = [data]

            # Process the first Data object in the list for positive and negative
            anchor_data = anchor_data_list[0]
            positive_data = positive_data_list[0]
            negative_data = negative_data_list[0]

            # Directly use the data without padding
            anchor_x = anchor_data.x
            positive_x = positive_data[0].x
            negative_x = negative_data.x

            print(f"Anchor Data - x shape: {anchor_x.shape}, edge_index shape: {anchor_data.edge_index.shape}, batch shape: {anchor_data.batch.shape}")
            print(f"Positive Data - x shape: {positive_x.shape}, edge_index shape: {positive_data[0].edge_index.shape}, batch shape: {positive_data[0].batch.shape}")
            print(f"Negative Data - x shape: {negative_x.shape}, edge_index shape: {negative_data.edge_index.shape}, batch shape: {negative_data.batch.shape}")

            # Squeeze the x tensors for anchor, positive, and negative data
            anchor_x_squeezed = anchor_data.x.squeeze(0)  # Removing the batch dimension
            positive_x_squeezed = positive_data[0].x.squeeze(0)  # For positive data
            negative_x_squeezed = negative_data.x.squeeze(0)  # For negative data

            # Verify the shapes after squeezing
            print(f"Anchor Data x shape after squeezing: {anchor_x_squeezed.shape}")
            print(f"Positive Data x shape after squeezing: {positive_x_squeezed.shape}")
            print(f"Negative Data x shape after squeezing: {negative_x_squeezed.shape}")


            # Pass the data into the model
            anchor_output = model(anchor_x_squeezed, anchor_data.edge_index, anchor_data.batch)
            positive_output = model(positive_x_squeezed, positive_data[0].edge_index, positive_data[0].batch)
            negative_output = model(negative_x_squeezed, negative_data.edge_index, negative_data.batch)

            # Compute loss
            loss = criterion(anchor_output, positive_output, negative_output)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}')


# Assuming model, triplet_loader, optimizer, and criterion are already defined
num_epochs = 10
train(model, triplet_loader, optimizer, criterion, num_epochs)








