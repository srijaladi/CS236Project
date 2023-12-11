import numpy as np
import pandas as pd
import torch
import matplotlib
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWL
import PIL
from PIL import Image
import random
import os
import shutil
import scipy
from scipy.linalg import sqrtm

def force_create(folderpath):
    if os.path.exists(folderpath):
        shutil.rmtree(folderpath)
    os.mkdir(folderpath)
    
def safe_create(folderpath):
    os.mkdir(folderpath)

def get_image(train: bool, name: str, number: int):
    idx = (4 - len(str(number))) * "0" + str(number)
    path = "train"
    if not(train):
        path = "test"
    path = "CIFAR-10-images/" + path
    path += "/" + name
    path += "/" + str(idx) + ".jpg"
    img = Image.open(path)
    np_data = np.asarray(img)
    fin_data = np.swapaxes(np.swapaxes(np_data, 1, 2), 0, 1)
    return fin_data/256

def load_all_data(train: bool, name: str):
    amt = 1000
    if train:
        amt = 5000
    all_data = np.zeros((amt,3,32,32))
    for i in range(amt):
        all_data[i] = get_image(train, name, i)
    return all_data

    
def np_arr_to_img(data : np.ndarray):
    if data.shape[0] == 3:
        data = np.swapaxes(np.swapaxes(data, 0, 1), 1, 2)
    res = Image.fromarray(data)
    return res

def save_image(img, filepath):
    img.save(filepath)
    
def save_np_image(data : np.ndarray, filepath):
    plt.imshow(np.swapaxes(np.swapaxes(data, 0, 1), 1, 2))
    plt.savefig(filepath)

def save_np_image_batch(data : np.ndarray, folderpath):
    safe_create(folderpath)
    for i,x in enumerate(data):
        filepath = folderpath + "/" + str(i) + ".jpg"
        save_np_image(x, filepath)
    if len(data) >= 16:
        data_use = data[:16]
        assert(data_use.shape[0] == 16 and data_use.shape[1] == 3 and data_use.shape[2] == 32)
        res = np.zeros((3, 128, 128))
        for i,x in enumerate(data_use):
            r, c = i//4, i%4
            res[:,r*32:r*32+32,c*32:c*32+32] = x[:,:,:]
        filepath = folderpath + "/group.jpg"
        save_np_image(res, filepath)
    
def gen_rand_img(shape):
    return np.random.uniform(0,1,size = shape)

def load_sample(ALL_DATA, B_PER, NUM_CLASSES, DEVICE, TOTAL = 5000):
    sz = NUM_CLASSES * B_PER
    idxs = np.random.randint(0,TOTAL,size = sz)
    lbls = np.repeat(np.arange(NUM_CLASSES), B_PER)
    base = lbls * TOTAL
    locs = base + idxs
    DATA = ALL_DATA[locs]
    return torch.from_numpy(DATA).float().to(DEVICE), torch.from_numpy(lbls.astype(int)).long().to(DEVICE)

def gen_sample_labels(B_PER, NUM_CLASSES, DEVICE):
    base = np.repeat(np.arange(NUM_CLASSES), B_PER)
    return torch.from_numpy(base.astype(int)).long().to(DEVICE)

def get_fid_comps(data : torch.tensor):
    mu, cov = torch.mean(data, dim = 0), torch.cov(data.T)
    return mu, cov

def get_fid(M1, M2, C1, C2):
    p1 = torch.sum(torch.square(M1 - M2), dim = -1)
    p2 = C1.trace() + C2.trace()
    p3 = torch.linalg.eigvals(C1 @ C2).sqrt().real.sum(dim=-1)
    return p1 + p2 - 2 * p3




