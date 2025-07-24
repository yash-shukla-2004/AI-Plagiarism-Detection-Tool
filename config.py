#paths , hyperparameters and preprocessing settings

#model:
in_channels = 2
hidden_channels = 128
out_channels = 32

weight_decay = 1e-4#weight decay
lr = 0.001#optimiser
num_epochs = 1#training : 100 for running 1 for testing
batch_size = 32#dataloading
num_clusters = 5#kmeans
margin = 1.0 #triplet loss 

#paths
dataset_testing_path = "./testing/original"
dataset_testing_path_aug = "./testing/augmented"