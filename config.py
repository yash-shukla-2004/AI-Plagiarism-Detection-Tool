#paths , hyperparameters and preprocessing settings

#model:
in_channels = 128
hidden_channels = 64
out_channels = 32

weight_decay = 1e-4#weight decay
lr = 0.001#optimiser
num_epochs = 100#training
batch_size = 32#dataloading
num_clusters = 5#kmeans
margin = 1.0 #triplet loss 

#paths
dataset_testing_path = "./testing/original"
dataset_testing_path_aug = "./testing/augmented"