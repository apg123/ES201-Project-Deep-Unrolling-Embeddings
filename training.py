import torch
import torch.nn as nn
import torch.nn.functional as F
from architecture import SparseDeepUnrolling
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt




def loss_lasso(xhat, x, zhat, lam):
    loss = 0.5 * (x - xhat).pow(2).sum(dim=(1)).mean()
    l1 = torch.norm(zhat, p=1, dim=1).mean()
    return loss + lam * l1

def loss_ls(xhat, x, zhat, lam):
    loss = 0.5 * (x - xhat).pow(2).sum(dim=(1)).mean()
    return loss


def train(model, dataset, 
          device='cuda', num_epochs=100, batch_size=1000, lr=0.001, eps =.001, lam=0.12, step=0.1, loss=loss_ls, 
          normalize = True, save_path='data/models/sparse_deep_unrolling', verbose=False):
    
    print(f"Batch size: {batch_size}, Num epochs: {num_epochs}, Learning rate: {lr}, Epsilon: {eps}, Lambda: {lam}, Step: {step}, Loss: {loss}, Normalize: {normalize}, Save path: {save_path}")
    print(f"Dim: {model.D}, Dict size: {model.P}, Num layers: {model.T}, lambda: {model.lam}, step: {model.step}, device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    losses = []

    losses_ls = []
    losses_lasso = []
    znorms = []
    xhatnorms = []
    largest_svd = []

    epoc_prog = tqdm(range(num_epochs))
    for _epoch in epoc_prog:
        
        model.train()

        for batch in tqdm(dl, disable=False):
            optimizer.zero_grad()
            x = batch[0].to(device).reshape(-1, model.D, 1)
            xhat, zhat = model(x)
            l = loss(xhat, x, zhat, lam)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            if normalize:
                model.normalize()
            
            losses.append(l.item())
            znorms.append(torch.norm(zhat, p=1, dim=1).mean().item())
            xhatnorms.append(torch.norm(xhat, p=2, dim=1).mean().item())

        #print(f"Epoch {epoch} Loss: {l.item()}")
        epoc_prog.set_description(f"Loss: {l.item()}")

        sv, _ = torch.lobpcg(torch.matmul(model.W.T, model.W), k=1)
        sv = sv[0].item() ** (1/2)
        largest_svd.append(sv)
        
    torch.save(model.state_dict(), save_path + str(model.P) + "_" + str(model.T) + "_" + str(model.lam) +".pt")
    return model, {"losses": losses, "znorms": znorms, "xhatnorms": xhatnorms, "largest_svd": largest_svd}

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    read = np.load('data/embeddings/train_raw.npy')
    training_embeddings = torch.Tensor(read).to('cpu')
    #training_embeddings = training_embeddings / torch.norm(training_embeddings, p=2, dim=1).reshape(-1, 1)
    print(training_embeddings.shape)
    print(training_embeddings[0].shape)
    print(training_embeddings.norm(p=2, dim=1).mean())
    print(training_embeddings.norm(p=2, dim=1).shape)

    initial_weight = None

    params = {
        'embedding_dim': 512,
        'dict_size': 2000, #1000, 2000, 10000
        'num_layers': 200, #100, 150, 200
        'lam': 1,
        'step': 0.12, #.12 for 1000 dict size
        'device': device
    }

    model = SparseDeepUnrolling(W=initial_weight, **params).to(device)

    train_params = {
        'device': device,
        'num_epochs': 30,
        'batch_size': 1000, #1000, 2000
        'lr': 0.0001, #0.0001, 0.00005, 0.00001
        'eps': 1e-6,
        'lam': 1,
        'step': 0.12,
        'loss': loss_lasso,
        'normalize': True,
        'save_path': 'data/models/sparse_deep_unrolling',
    }

    data = TensorDataset(training_embeddings)

    trained_model, rdict = train(model, data, **train_params)

    print("Training complete")

    plt.plot(rdict['losses'])
    plt.legend(["Loss"])
    plt.title("Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(rdict['znorms'])
    plt.legend(["Norm"])
    plt.title("Norm of Z")
    plt.xlabel("Batch")
    plt.ylabel("Norm")
    plt.show()

    plt.plot(rdict['xhatnorms'])
    plt.legend(["Norm"])
    plt.title("Norm of xhat")
    plt.xlabel("Batch")
    plt.ylabel("Norm")
    plt.show()

    plt.plot(rdict['largest_svd'])
    plt.legend(["SVD"])
    plt.title("Largest SVD of W^T W")
    plt.xlabel("Epoch")
    plt.ylabel("SVD")
    plt.show()