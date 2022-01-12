import os
import warnings
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvisions import transforms

from iafoule.models import MCNN, DenseScaleNet
from iafoule.utils import normalize_target
from iafoule.loader import CreateLoader, validation
from iafoule.writing_text import write, random_date

warnings.filterwarnings("ignore")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', default='mcnn', choices=['mcnn', 'dsnet'])
    p.add_argument('--data', choice["GCC"], default='GCC')
    return p.parse_args()

def writing(img_array):
    im_data = write(3, 3,  'BELVEDERE-DNUM', img_array)
    im_data = write(60, 3 , random_date(), im_data)
    return im_data


availables_models = {'mcnn' : MCNN,
                    'dsnet': DenseScaleNet}

input_grayscale = {'mcnn': True,
                   'dsnet': False}


args = parse_args()
method = args.method
dataset_name = args.data
grayscale = input_grayscale[method]
path_truth = f'/workspace/data/{dataset_name}/maps_adaptive_kernel/'
save_path = '/workspace/data/models/'


# create train and test sample
split = 0.8

paths = pd.read_csv(f'/workspace/data/{dataset_name}/gcc_mapping.csv') # change mapping name
shuffle_path = paths["path_img"].sample(frac=1).reset_index(drop=True)
train_paths = shuffle_path[:int(len(shuffle_path) * 0.8)]
test_paths = shuffle_path[int(len(shuffle_path) * 0.8):]


# learning params 
n_epochs = 20
lr = 0.00001
rand_seed = 64678  
mean_norm = 0.5
std_norm = 0.2

if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    

net = availables_models[method]
params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: 
                                    p.requires_grad, 
                                    net.parameters()), 
                             lr=lr)
criterion = torch.nn.MSELoss()

# add functions
transform = transforms.Compose([writing,
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean_norm, std=std_norm)
])


# load data

train_loader = CreateLoader(train_paths, 
                             gt_path=path_truth, 
                             ratio=4,
                             batch_size=1,
                             n_samples=None,
                             aug=True,
                             grayscale=grayscale,
                             img_transformer=transform)


test_loader = CreateLoader(test_paths,
                             gt_path=path_truth, 
                             ratio=4,
                             batch_size=1,
                             n_samples=None,
                             aug=False,
                             grayscale=grayscale,
                             img_transformer=transform)


train_loss = 0
if torch.cuda.is_available():
    use_GPU = True

best_mae, _  = validation(net, test_loader, cuda=use_GPU)

for epoch in range(n_epochs):
    train_loss = 0.0
    net.train()
    n = 0
    for img, target, count in train_loader:
        n += 1
        optimizer.zero_grad()
        if torch.cuda.is_available():
            img = img.cuda()
            target = target.cuda()
        output = net(img)
        target = normalize_target(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        ms_data = f'Loss: {loss:>9f} [{n:>5d}/{len(train_loader):>5d}]'
        print(ms_data)

    mae, mse = validation(net, test_loader, cuda=False)

    ms_epoch = f'Epoch {epoch+1}/{n_epochs} Loss: {train_loss/len(train_loader):>9f}, MAE: {mae:>2f}, MSE: {mse:>2f}, Best MAE: {best_mae:>2f}'
    print(ms_epoch)
    if mae < best_mae:
        best_mae = mae
        torch.save(net.state_dict(), os.path.join(save_path, f'{method}_{dataset_name}_{epoch}.pth'))

