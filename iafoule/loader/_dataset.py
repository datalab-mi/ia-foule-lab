import random
import os
from pathlib import Path
import glob
import scipy
import h5py
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.sparse import load_npz

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _rescale(img_path, gt_path, factor=1, truth_format='npz', grayscale=False):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) if grayscale else cv2.imread(img_path)
    den = _load_density_map(img_path, gt_path, truth_format='npz')
    ht, wd = int(img.shape[0]), int(img.shape[1])
    
    # resize image
    img_wd, img_ht = int(wd / factor), int(ht / factor) 
    img = cv2.resize(img, (img_wd, img_ht))
    
    # resize gt
    gt_wd, gt_ht = int(wd / factor), int(ht / factor)   
    den = cv2.resize(den, (gt_wd, gt_ht))
    return img, den


def _augmentation(img_path, gt_path, factor=1, p_crop=0.5, p_flip=0.5, truth_format='npz', grayscale=False):
    img = Image.open(img_path).convert('L') if grayscale else Image.open(img_path).convert('RGB')
    den = _load_density_map(img_path, gt_path, truth_format='npz')
    
    if factor > 1:
        # crop
        w, h = (img.size[0]// factor, 
            img.size[1]// factor)

        if random.random() >= p_crop:
            dx = int(random.randint(0, 1) * w)
            dy = int(random.randint(0, 1) * h)
        else:
            dx = int(random.random() * w)
            dy = int(random.random() * h)

        img = img.crop((dx, dy, w + dx, h + dy))
        den = den[dy: h + dy, dx: w + dx]
    
    # flip
    if random.random() >= p_flip:
        den = np.fliplr(den)
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return np.array(img), den


def _load_density_map(img_pathname, gt_path, truth_format='npz'):
    # load density maps saved in 'h5' or 'npz'
    name_file = Path(img_pathname).stem
    if truth_format == 'h5':
        f = h5py.File(os.path.join(gt_path, name_file + '.h5'), 'r')
        den = f['density'][:]
        f.close()
        den  = den.astype(np.float32, copy=False)                    
    elif truth_format == 'npz':
        den = load_sparse(os.path.join(gt_path, name_file + '.npz'))
    else:
        raise NotImplementedError
    return den


def load_sparse(filename):
    return load_npz(filename).toarray()


def _load_data(img_path, gt_path, 
               factor=4, img_transformer=None, 
               aug=True, p_crop=0.5, p_flip=0.5, 
               truth_format='npz', grayscale=False):
    
    
    if aug:
        img, den = _augmentation(img_path, gt_path, 
                                 factor, p_crop=0.5, p_flip=0.5, 
                                 truth_format='npz', grayscale=grayscale)
    else:
        img, den = _rescale(img_path, gt_path, 
                            factor, truth_format='npz', grayscale=grayscale)

    # add func modifier
    if img_transformer is not None:
        img = img_transformer(img)
    
    n_pers = den.sum()
    
    if not torch.is_tensor(img):    
        img = np.expand_dims(img, axis=0) if len(img.shape) == 2 else img.reshape(3, img.shape[0], img.shape[1]).copy()
    else:
        img = img.clone()

    den = np.expand_dims(den, axis=0)
    return img, den.copy(), n_pers


class RawDataset(Dataset):
    def __init__(self, paths_img, 
                 gt_path_root, 
                 transform=None,
                 n_samples=None,
                 img_transformer=None, 
                 ratio=1, 
                 aug=True, 
                 p_crop=0.5, 
                 p_flip=0.5, 
                 grayscale=False,
                 truth_format='npz'):
        
        self.nsamples = len(paths_img) if n_samples is None else n_samples
        self.aug = aug
        self.gt_path_root = gt_path_root
        self.paths_img = np.array(paths_img) if n_samples is None else np.random.choice(paths_img, n_samples)
        self.ratio = ratio
        self.transform = transform
        self.img_transformer = img_transformer
        self.p_crop = p_crop 
        self.p_flip= p_flip
        self.truth_format = truth_format
        self.grayscale = grayscale
        
    def __getitem__(self, index):
        img, target, count = _load_data(img_path=self.paths_img[index], 
                                       gt_path=self.gt_path_root,
                                       img_transformer=self.img_transformer,
                                       factor=self.ratio,
                                       aug=self.aug,
                                       grayscale=self.grayscale,
                                       p_crop=self.p_crop,
                                       p_flip=self.p_flip,
                                       truth_format=self.truth_format)
        if self.transform is not None:
            img = self.transform(img[0])
        return img, target, count
    
    def __len__(self):
        return self.nsamples
    
    
class CreateLoader:
    """
    Create DataLoaders using for training a crowdcounting model
    
    args:
        - train_img_path: list, list of training path images.
        - test_img_path: list, list of testing path images.
        - gt_path: str, path where are saved density maps.
                  NOTE: filename must be the same as image
        - n_samples: int, number of created images by DataLoader. If None,
                        number is the len of passed images, default None.
        - batch_size: int, number of image each batch, default 1.
        - ratio: int, division scale of image, default 1.
        - shuffle: bool, shuffle data loader, default True.
        - aug: bool, activate data augmentation (crop if ratio > 1 and flip)
        - img_transformer: func, apply a function during data loading
        - transform: func, apply a function after data loader (NOTE: duplicate)
        - p_crop: float, prob for using crop during data augmentation (when aug is True)
        - p_flip: float, prob for using flip during data augmentation (when aug is True)
        - grayscale: bool, load images as grayscale (only one channel)
        - gt_format: str, method for loading pre-computed density map ('npz' or 'h5')
    
    return:
        train_dataloader, test_dataloader
    """
    def __new__(self, img_paths: list, 
                 gt_path: str,
                 n_samples=None,
                 ratio=1,
                 batch_size=1,
                 aug=True,
                 img_transformer=None,
                 transform=None, 
                 p_crop=0.5, 
                 p_flip=0.5,
                 grayscale=False,
                 shuffle=True,
                 gt_format='npz'):
        
        self.loader = DataLoader(dataset=RawDataset(
            img_paths, gt_path, 
            transform=transform, 
            n_samples=n_samples,
            img_transformer=img_transformer,
            aug=aug,
            grayscale=grayscale,
            ratio=ratio, 
            p_crop=p_crop, 
            p_flip=p_flip,
            truth_format=gt_format),
                                  shuffle=shuffle, 
                                  batch_size=batch_size)

        return self.loader