import torch
import numpy as np
import splice 
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

l1_penalty = 0.2
dict_size = 10000
batch_size = 1024 

image_mean = torch.tensor(np.load('data/embeddings/train_center.npy')).float()

dictionary = torch.tensor(np.load('data/dict/mscoco_clip_centered.npy')).float()
model = splice.SPLICE(image_mean=image_mean, dictionary = dictionary, solver='admm', device=device, l1_penalty=l1_penalty)
dense_embeddings = torch.tensor(np.load('data/embeddings/test_centered.npy')).float().to(device)

if True: 
    r = None
    print(len(dense_embeddings))
    for i in tqdm(range(0, len(dense_embeddings), batch_size)):
        if r is None:
            r = model.decompose(dense_embeddings[i:i+batch_size])
        else:
            r = torch.concat((r, model.decompose(dense_embeddings[i:i+batch_size])), dim=0)
        torch.save(r, f'data/embeddings/test_splice_sparse_{l1_penalty}_{dict_size}.pt')

    print(r.shape)

if True:
    r = torch.load(f'data/embeddings/test_splice_sparse_{l1_penalty}_{dict_size}.pt').float()
    xhat = None
    for i in tqdm(range(0, len(r), batch_size)):
        if xhat is None:
            xhat = model.recompose_image(r[i:i+batch_size])
        else:
            xhat = torch.concat((xhat, model.recompose_image(r[i:i+batch_size])), dim=0)
        torch.save(xhat, f'data/embeddings/test_splice_reconstructed_{l1_penalty}_{dict_size}.pt')

    print(xhat.shape)

