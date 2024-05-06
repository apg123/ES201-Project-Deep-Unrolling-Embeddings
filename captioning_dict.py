import torch
from architecture import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import ZeroCLIP as zc


state = torch.load('data/models/sparse_deep_unrolling2000_200_1.pt')

print(state['W'])
D = state['W'].T

forbidden_tokens_file_path = "./zeroshotimage/forbidden_tokens.npy"
zero_clip = zc.CLIPTextGenerator(forbidden_tokens_file_path=forbidden_tokens_file_path,target_seq_length=12)

all_texts = []

for i in tqdm(range(0, 2000)):
    val = D[i]
    print(val.shape)
    texts = zero_clip.run(val, cond_text=" ", beam_size=3)
    all_texts.append(texts)

lines = ['//'.join(text) for text in all_texts]
with open("results/captions.txt", mode="w") as file:
    file.write("\n".join(lines))