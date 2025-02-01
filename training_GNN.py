import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GINConv,global_mean_pool
from dataset_config import triplet_loader #dataset
from config import in_channels,out_channels,hidden_channels,lr,num_epochs,margin
#define hyper-parameters


class GNNModel(nn.Module):

    def __init__(self,in_channels, hidden_channels, out_channels):
        super(GNNModel,self).__init__()

        self.gat = GATConv(in_channels,hidden_channels,heads=4,concat=True)

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
def train_model(model,data_loader,optimizer,criterion,epochs=50):
    model.train()
    for epoch in range(epochs):
        for anchor,positive,negative in data_loader:
            optimizer.zero_grad()

            anchor_emb = model(anchor.x, anchor.edge_index, anchor.batch)
            positive_emb = model(positive.x, positive.edge_index, positive.batch)
            negative_emb = model(negative.x, negative.edge_index, negative.batch)

            loss = criterion(anchor_emb,positive_emb,negative_emb)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss = {avg_loss:.4f}')

            


