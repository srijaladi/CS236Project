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
import Helpers
from Helpers import *
from Helpers import force_create, save_np_image_batch, load_all_data, load_sample, get_fid_comps, get_fid
import Models
from Models import Discriminator, Generator, GAN, Classifier, FIDInceptionV3, Interpolate
import argparse
import torchvision
from torchvision.models import resnet50, alexnet, lenet_5, mobilenet_v3_small


parser = argparse.ArgumentParser()

parser.add_argument(
    "--gacc",
    type=int,
    required = True,
    default=20,
    help="percent accuracy to pre-train discriminator layers up to",
)

args = parser.parse_args()

ALL_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
NUM_CLASSES = 10

PRE_PRINT_ITER  = 100
PRE_SAVE_ITER   = 1000
PRE_TRAIN_ITERS = 10000
PRE_PRINT_ITER  = 1
PRE_SAVE_ITER   = 1
PRE_TRAIN_ITERS = 1
GOAL_PERC_ACC   = args.gacc
B_PER = 10

PRINT_ITER  = 100
SAVE_ITER   = 1000
TRAIN_ITERS = 25001

PRINT_ITER  = 10
SAVE_ITER   = 10
TRAIN_ITERS = 25

LATENT_SIZE = 128
EXP_NAME    = "PRE_" + str(int(GOAL_PERC_ACC)) + "_SHIP"
EXP_NAME    = "EXPERIMENTS/" + EXP_NAME
force_create(EXP_NAME)

FID_EVAL_SIZE = 20
FID_BASE_SIZE = 20
EVAL_SIZE   = 20
B = 64


IN_DIM_C, IN_DIM_H, IN_DIM_W = 3, 32, 32
IN_DIMS = (IN_DIM_C, IN_DIM_H, IN_DIM_W)
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


RELU_INPLACE = True
RELU = nn.ReLU(inplace = RELU_INPLACE)
LEAKY_RELU = nn.LeakyReLU(0.1, inplace = RELU_INPLACE)
RN50 = resnet50(weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
UPSAMPLE = Interpolate(224)
print("GOT RESNET50")

layers_C = [nn.Conv2d(3, 32, (4,4), stride = 2, padding = 1), LEAKY_RELU, 
            nn.Conv2d(32, 32, (3,3), padding = 1), LEAKY_RELU,
            nn.Conv2d(32, 64, (4,4), stride = 2, padding = 1), LEAKY_RELU, 
            nn.Conv2d(64, 64, (3,3), padding = 1), LEAKY_RELU,
            nn.Flatten(), 
            nn.Linear(64 * IN_DIM_H // 4 * IN_DIM_W // 4, 32 * 32), LEAKY_RELU, 
            nn.Linear(32 * 32, 10)]

layers_C = [UPSAMPLE, RN50, nn.Linear(2048, 1024), LEAKY_RELU, 
            nn.Linear(1024,100), LEAKY_RELU, nn.Linear(100,10)]

layers_D = [nn.Conv2d(3, 32, (4,4), stride = 2, padding = 1), LEAKY_RELU, 
            nn.Conv2d(32, 32, (3,3), padding = 1), LEAKY_RELU,
            nn.Conv2d(32, 64, (4,4), stride = 2, padding = 1), LEAKY_RELU, 
            nn.Conv2d(64, 64, (3,3), padding = 1), LEAKY_RELU,
            nn.Flatten(), 
            nn.Linear(64 * IN_DIM_H // 4 * IN_DIM_W // 4, 32 * 32), LEAKY_RELU, 
            nn.Linear(32 * 32, 10), LEAKY_RELU, nn.Linear(10,1)]

layers_G = [UPSAMPLE, RN50, nn.Linear(2048, 1024), LEAKY_RELU, 
            nn.Linear(1024,100), LEAKY_RELU, 
            nn.Linear(100,10), LEAKY_RELU, nn.Linear(10,1)]

layers_G = [nn.Linear(LATENT_SIZE,32 * 32), nn.BatchNorm1d(32 * 32), RELU, 
            nn.Linear(32 * 32, 64 * 8 * 8), nn.BatchNorm1d(64 * 8 * 8), RELU,
            nn.Unflatten(1,(64,8,8)), nn.PixelShuffle(2),
            nn.Conv2d(64 // 4, 32, (3,3), padding = 'same'), torch.nn.BatchNorm2d(32), RELU, nn.PixelShuffle(2),
            nn.Conv2d(32 // 4, IN_DIM_C, (3,3), padding = 'same')]

ALL_DATA = load_all_data(True, "ship")

#print("WORKING")
#FID = FIDInceptionV3().to(DEVICE)
#M1, C1 = get_fid_comps(FID(torch.from_numpy(ALL_DATA[0:FID_BASE_SIZE]).float().to(DEVICE)).logits)
#print("GOT INCEPTION BASELINE")

keys = {x : i for i,x in enumerate(ALL_CLASSES)}
ALL_PRE_TRAIN_DATA = np.zeros((10,5000,3,32,32))
for i,x in enumerate(ALL_CLASSES):
    print(x)
    ALL_PRE_TRAIN_DATA[i] = load_all_data(True, x)
ALL_PRE_TRAIN_DATA = ALL_PRE_TRAIN_DATA.reshape(-1,3,32,32)

ALL_PRE_TEST_DATA = np.zeros((10,1000,3,32,32))
for i,x in enumerate(ALL_CLASSES):
    print(x)
    ALL_PRE_TEST_DATA[i] = load_all_data(False, x)
ALL_PRE_TEST_DATA = torch.from_numpy(ALL_PRE_TEST_DATA.reshape(-1,3,32,32)).float().to(DEVICE)
ALL_PRE_TEST_LABELS = torch.zeros(10 * 1000)
for i in range(10):
    ALL_PRE_TEST_LABELS[i*1000:i*1000+1000] = i
ALL_PRE_TEST_LABELS = ALL_PRE_TEST_LABELS.int().to(DEVICE)

print("Loaded data")

pre_losses = []
pre_accs = []

PRE_MODEL = Classifier(layers_C, 1e-3).to(DEVICE)
print("CREATED PRE TRAINING MODEL")

print("Began pre-training")
for itr in range(PRE_TRAIN_ITERS):
    x, y = load_sample(ALL_PRE_TRAIN_DATA, B_PER, NUM_CLASSES, DEVICE)
    loss = PRE_MODEL.train(x,y)
    correct, total, acc = PRE_MODEL.accuracy(ALL_PRE_TEST_DATA, ALL_PRE_TEST_LABELS)

    pre_losses.append(loss.item())
    pre_accs.append(acc.item())

    if acc >= (GOAL_PERC_ACC + 1)/100:
        print("REACHED GOAL ACCURACY")
        print("PRINTING ITR:", itr, acc)
        print("SAVING ITR:", itr)
        np.savetxt(EXP_NAME + "/pre_losses.txt", np.array(pre_losses))
        np.savetxt(EXP_NAME + "/pre_accs.txt", np.array(pre_accs))
        torch.save(PRE_MODEL, EXP_NAME + "/PRE_MODEL.txt")
        break

    if itr%PRE_PRINT_ITER == 0:
        print("PRINTING ITR:", itr, acc)
    if itr%PRE_SAVE_ITER == 0:
        print("SAVING ITR:", itr)
        np.savetxt(EXP_NAME + "/pre_losses.txt", np.array(pre_losses))
        np.savetxt(EXP_NAME + "/pre_accs.txt", np.array(pre_accs))
        torch.save(PRE_MODEL, EXP_NAME + "/PRE_MODEL.txt")


print("FINAL ACCURACY OF PRE-TRAINING:", PRE_MODEL.accuracy(ALL_PRE_TEST_DATA, ALL_PRE_TEST_LABELS)[2])

for i,layer in enumerate(layers_C):
    layers_D[i].load_state_dict(layer.state_dict())

MODEL = GAN(layers_D, layers_G, LATENT_SIZE, 1e-3, 1e-3).to(DEVICE)
print("CREATED GAN")

losses = []
accs = []
full_accs = []
fids = []

for itr in range(TRAIN_ITERS):
    x_real = torch.from_numpy(ALL_DATA[np.random.randint(0,5000,size = B)]).float().to(DEVICE)
    loss_D, x_fake = MODEL.train_D(x_real)
    loss_G, x_fake = MODEL.train_G(B)
    correct, total, acc, correct_real, correct_fake = MODEL.accuracy(x_real, x_fake)

    losses.append([loss_D.item(), loss_G.item()])
    accs.append(acc.item())
    full_accs.append([correct.item(), total, acc.item(), correct_real.item(), correct_fake.item()])

    if itr%PRINT_ITER == 0:
        fake = MODEL.generate(FID_EVAL_SIZE)
        M2, C2 = get_fid_comps(FID(fake).logits)
        fid = get_fid(M1, M2, C1, C2)
        fids.append(fid.item())
        print("PRINT_ITER:", itr, fid)
    if itr%SAVE_ITER == 0:
        print("SAVING ITR:", itr)
        np.savetxt(EXP_NAME + "/losses.txt", np.array(losses))
        np.savetxt(EXP_NAME + "/accs.txt", np.array(accs))
        np.savetxt(EXP_NAME + "/fids.txt", np.array(fids))
        np.savetxt(EXP_NAME + "/full_accs.txt", np.array(full_accs))
        torch.save(MODEL, EXP_NAME + "/GAN.txt")
        fake = MODEL.generate(EVAL_SIZE).cpu().detach().numpy()
        save_np_image_batch(fake, EXP_NAME + "/EVAL_ITR_" + str(itr))





