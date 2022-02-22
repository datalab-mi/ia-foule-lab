import os
import warnings
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms

from iafoule.models import MCNN, DenseScaleNet, MobileCount
from iafoule.utils import normalize_target, get_logger, compute_lc_loss, Timer
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

# args script
availables_models = {'mcnn' : MCNN,
                    'dsnet': DenseScaleNet,
                    'mb': MobileCount}

input_grayscale = {'mcnn': True,
                   'mb': False,
                   'dsnet': False}

args = parse_args()
method = args.method
dataset_name = args.data
grayscale = input_grayscale[method]
net = availables_models[method]

# create train and test sample with less 500 persons
path_truth = f'/workspace/data/{dataset_name}/density/maps_adaptive_kernel/'
save_path = './models/'

paths = pd.read_csv('./data/GCC/density/gcc_mapping.csv') # change name
selected_paths = paths#[paths['n_persons'] < 500]

# shuffle because images are sorted by scene
shuffle_path = selected_paths["path_img"].sample(frac=1).reset_index(drop=True)
split = 0.8
train_paths = shuffle_path[:int(len(shuffle_path) * 0.8)]
test_paths = shuffle_path[int(len(shuffle_path) * 0.8):]


# learning params 
GPU = torch.cuda.is_available()
n_epochs = 500
# custom loss : ponderation loss l1 (loss = l2 + lambda x l1)
lbda = 1000
custom_loss = True
# frequence logs
ms_time = 500

# adam
lr = 1e-6
weight_decay = 1e-4

# mean and std ImageNet
mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]

rand_seed = 64678  
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

if GPU:
    net.cuda()
params = list(net.parameters())
optimizer = torch.optim.Adam(filter(lambda p: 
                                    p.requires_grad, 
                                    net.parameters()), 
                             lr=lr, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()
logger = get_logger(f'./data/logs/{dataset_name}_{method}.txt')

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
                             img_transformer=transform,
                             num_worker=4)


test_loader = CreateLoader(test_paths,
                             gt_path=path_truth, 
                             ratio=4,
                             batch_size=1,
                             n_samples=None,
                             aug=False,
                             grayscale=grayscale,
                             img_transformer=transform,
                             num_worker=4)

path_saved_model = os.path.join(save_path, f'{method}_{dataset_name}.pth')
train_loss = 0
best_mae, _  = validation(net, test_loader, cuda=GPU)

for epoch in range(first_epoch, n_epochs):
    train_loss = 0.0
    net.train()
    n = 0
    max_loss = 0
    
    # do epoch
    for img, target, count in train_loader:
        n += 1
        optimizer.zero_grad()
        if GPU:
            img = img.cuda()
            target = target.cuda()
        output = net(img)
        if not output.shape == target.shape:
            target = normalize_target(output, target, gpu=GPU)
        loss = criterion(output, target)
        if custom_loss:
            lc_loss = compute_lc_loss(output, target)
            loss = loss + lbda * lc_loss
            
        if loss.item() > max_loss:
            max_loss = loss.item()
            gt = target.sum()
            pr = output.sum()
    
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        if (n % ms_time) == 0 or (n == len(train_loader)):
            ms_data = f'Loss: {train_loss/n:>9f} [{n:>5d}/{len(train_loader):>5d}]'
            logger.info(ms_data)
    
    # each epoch validate
    mae, rmse = validation(net, test_loader, cuda=GPU)
    ms_epoch = (f'Epoch {epoch+1}/{n_epochs} Loss: {train_loss/len(train_loader):>15f}, '
                f'MAE: {mae:>2f}, RMSE: {rmse:>2f}, Best MAE: {best_mae:>2f}, MAX: {max_loss:>9f} pred: {pr:>2f} - gt: {gt:>2f}')
    logger.info(ms_epoch)
    
    # save good models
    if mae < best_mae:
        best_mae = mae
        saved_model = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}
        torch.save(saved_model, path_saved_model)


if saved_model.is_file():
    checkpoint = torch.load(path_saved_model)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    first_epoch = checkpoint['epoch']
    loss = checkpoint['loss']