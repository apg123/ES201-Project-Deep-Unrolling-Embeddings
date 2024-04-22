import torch as torch
import random
import itertools
import numpy as np
import torch
import clip
import math
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os



def process_images(model, preprocess, image_locations, device = 'cuda', batch_size=200):
    all_features = []
    for i in tqdm(range(0, len(image_locations), batch_size)):
        try:
            images = torch.cat([preprocess(Image.open(img)).unsqueeze(0)for img in image_locations[i:i+batch_size]]).to(device)
        except:
            print("Error with images, preprossing each image individually")
            print("Error with image", img)
            good_images = []
            for img in image_locations[i:i+batch_size]:
                try:
                    good_images.append(preprocess(Image.open(img)).unsqueeze(0))
                except:
                    print("Error with image", img)
                    continue
            images = torch.cat(good_images).to(device)
        with torch.no_grad():
            features = model.encode_image(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)

def process_images(model, preprocess, image_locations, device = 'cuda', batch_size=200):
    all_features = []
    for i in tqdm(range(0, len(image_locations), batch_size)):
        images = torch.cat([preprocess(Image.open(img)).unsqueeze(0)for img in image_locations[i:i+batch_size]]).to(device)
        with torch.no_grad():
            features = model.encode_image(images)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device)

batch_size = 800

image_folder = 'data/coco/train2017/train2017/'
files = os.listdir(image_folder)
paths = [image_folder + file for file in files]
features = process_images(model, preprocess, paths, batch_size=batch_size)
np.save('data/embeddings/train.npy', features)

image_folder = 'data/coco/val2017/val2017/'
files = os.listdir(image_folder)
paths = [image_folder + file for file in files]
features = process_images(model, preprocess, paths, batch_size=batch_size)
np.save('data/embeddings/val.npy', features)

image_folder = 'data/coco/test2017/test2017/'
files = os.listdir(image_folder)
paths = [image_folder + file for file in files]
features = process_images(model, preprocess, paths, batch_size=batch_size)
np.save('data/embeddings/test.npy', features)

