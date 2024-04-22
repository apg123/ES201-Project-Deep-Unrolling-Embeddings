import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import SparseDeepUnrolling



def loss(xhat, x, zhat, lam):
    loss = F.mse_loss(xhat, x)
    l1 = torch.norm(zhat, p=1)
    return loss + lam * l1


def train(model, training_embeddings, 
          device='cuda', num_epochs=100, batch_size=1000, lr=0.01, lam=0.12, step=0.1, 
          save_path='data/models/sparse_deep_unrolling.pth'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pass

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    training_embeddings = torch.load('data/embeddings/train.npy')

    initial_weight = None

    params = {
        'embedding_dim': 512,
        'dict_size': 1000,
        'num_layers': 10,
        'lam': 0.12,
        'step': 0.1,
        'device': device
    }

    model = SparseDeepUnrolling(W=initial_weight, **params).to(device)

    train_params = {
        'device': device,
        'num_epochs': 100,
        'batch_size': 1000,
        'lr': 0.01,
        'lam': 0.12,
        'step': 0.1,
        'save_path': 'data/models/sparse_deep_unrolling.pth'
    }

    train(model, training_embeddings, **train_params)
    pass