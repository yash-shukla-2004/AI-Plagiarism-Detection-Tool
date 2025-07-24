from sklearn.metrics import silhouette_score
import numpy as np
import torch.nn.functional as F
import torch

def eval_plagiarism(embeddings, cluster_labels, kmeans):
    plagiarism_scores = []
    for i, emb in enumerate(embeddings):
        distance = np.linalg.norm(emb - kmeans.cluster_centers_[cluster_labels[i]])
        max_distance = np.max([np.linalg.norm(e - c) for e in embeddings for c in kmeans.cluster_centers_])
        plagiarism_percent = 100 - (distance / max_distance) * 100
        plagiarism_scores.append(plagiarism_percent)

    return plagiarism_scores

def evaluate_model(model, data_loader):
    model.eval()  # Set model to evaluation mode
    similarities = []

    with torch.no_grad():
        for anchor, positives, negative in data_loader:
            anchor_emb = model(anchor['x'], anchor['edge_index'], anchor['batch_index'])

            pos_sims = []
            for positive in positives:
                positive_emb = model(positive['x'], positive['edge_index'], positive['batch_index'])
                pos_sim = F.cosine_similarity(anchor_emb, positive_emb)
                pos_sims.append(pos_sim.mean().item())

            negative_emb = model(negative['x'], negative['edge_index'], negative['batch_index'])
            neg_sim = F.cosine_similarity(anchor_emb, negative_emb)

            avg_pos_sim = sum(pos_sims) / len(pos_sims)
            similarities.append((avg_pos_sim, neg_sim.mean().item()))

    avg_pos_sim = sum(pos for pos, _ in similarities) / len(similarities)
    avg_neg_sim = sum(neg for _, neg in similarities) / len(similarities)

    print(f'Average Positive Similarity: {avg_pos_sim:.4f}')
    print(f'Average Negative Similarity: {avg_neg_sim:.4f}')

def eval_cluster(embeddings, cluster_labels):
    score = silhouette_score(embeddings, cluster_labels)
    print(f"Silhouette Score: {score}")