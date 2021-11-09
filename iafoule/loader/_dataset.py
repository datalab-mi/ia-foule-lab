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
from pathlib import Path
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ._transformer import RandomImageCrop, RandomGammaCorrection, RandomFlip


def _rescale(img, 
             den, 
             factor=1):
    
    wd, ht = img.size
    
    # resize image
    img_wd, img_ht = wd // factor, ht // factor 
    img = img.resize((img_wd, img_ht))
    
    # resize gt
    gt_wd, gt_ht = wd // factor, ht // factor  
    den = cv2.resize(den, (gt_wd, gt_ht), interpolation=cv2.INTER_AREA)
    # keep the sum of density map
    den = den * ((wd * ht)/ (gt_wd * gt_ht))
    return img, den


def _augmentation(img, 
                  den,
                  factor_crop=4,
                  index=None,
                  crop_with_torch=True,
                  seed=None):
    
    # fix seed for crop are the same each epoch
    if index is not None:
        seed += index * 0.2

    if factor_crop > 1:
        img, den = RandomImageCrop(factor=factor_crop, seed=seed, use_torch=crop_with_torch)(img, den)
    
    img, den = RandomFlip(seed=seed)(img, den)
    img = RandomGammaCorrection(seed=seed)(img)
    return img, den


def _load_density_map(img_pathname, gt_path, truth_format='npz'):
    # load density maps saved in 'h5' or 'npz'
    name_file = Path(img_pathname).stem
    if truth_format == 'h5':
        f = h5py.File(os.path.join(gt_path, name_file + '.h5'), 'r')
        den = f['density'][:]
        den  = den.astype(np.float32, copy=False)                    
    elif truth_format == 'npz':
        den = load_sparse(os.path.join(gt_path, name_file + '.npz'))
    else:
        raise NotImplementedError
    return den


def load_sparse(filename):
    return load_npz(filename).toarray()


def _load_data(img_path,
               gt_path, 
               factor=1,
               factor_crop=4,
               index=None,
               transformer=None,
               crop_with_torch=True,
               aug=True,
               truth_format='npz',
               grayscale=False, 
               seed=None):
    
    img = Image.open(img_path).convert('L') if grayscale else Image.open(img_path).convert('RGB')
    den = _load_density_map(img_path, gt_path, truth_format='npz')
    
    if factor > 1:
        img, den = _rescale(img, den, 
                            factor=factor)
    
    if aug:
        img, den = _augmentation(img, den, 
                                 factor_crop=factor_crop, 
                                 seed=seed,
                                 crop_with_torch=crop_with_torch, 
                                 index=index)
    img = np.array(img)
    
    # add func modifier
    if transformer is not None:
        img = transformer(img)
    
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
                 n_samples=None,
                 factor_crop=4,
                 transformer=None, 
                 ratio=1, 
                 aug=True, 
                 crop_with_torch=True,
                 grayscale=False,
                 seed=None,
                 truth_format='npz'):
        
        self.nsamples = len(paths_img) if n_samples is None else n_samples
        self.aug = aug
        self.gt_path_root = gt_path_root
        self.paths_img = np.array(paths_img) if n_samples is None else np.random.choice(paths_img, n_samples)
        self.ratio = ratio
        self.crop_with_torch = crop_with_torch
        self.transformer = transformer
        self.truth_format = truth_format
        self.grayscale = grayscale
        self.seed = seed
        self.factor_crop = factor_crop
        
    def __getitem__(self, index):
        img, target, count = _load_data(img_path=self.paths_img[index],
                                       gt_path=self.gt_path_root,
                                       index=index,
                                       transformer=self.transformer,
                                       factor=self.ratio,
                                       factor_crop=self.factor_crop,
                                       crop_with_torch=self.crop_with_torch,
                                       aug=self.aug,
                                       grayscale=self.grayscale,
                                       truth_format=self.truth_format,
                                       seed=self.seed)
        return img, target, count
    
    def __len__(self):
        return self.nsamples
    
    
class CreateLoader:
    """
    Create DataLoaders using for training a crowdcounting model
    
    args:
        - img_path: list, list of path images.
        - gt_path: str, path where are saved density maps.
                  NOTE: filename must be the same as image
        - n_samples: int, number of created images by DataLoader. If None,
                        number is the len of passed images, default None.
        - batch_size: int, number of image each batch, default 1.
        - seed: int, the the same seed for each epoch, if None the seed is random.
                default None.
        - aug: bool, activate data augmentation (select a crop, with scale, random contrast and flip).
        - num_workers: int, n other workers for apply transformation. Default 0.
        - ratio: int, division scale of image, default 1.
        - factor_crop: int, division scale of crop, default 4. Applied when aug is True.
        - shuffle: bool, shuffle data loader, default True.
        - transform: func, apply a function in the image
        - grayscale: bool, load images as grayscale (only one channel)
        - gt_format: str, method for loading pre-computed density map ('npz' or 'h5')
    
    return:
        dataloader
    """
    def __new__(self, img_paths: list, 
                 gt_path: str,
                 n_samples=None,
                 ratio=1,
                 factor_crop=4,
                 batch_size=1,
                 aug=True,
                 transformer=None,
                 seed=None,
                 crop_with_torch=True,
                 grayscale=False,
                 shuffle=True,
                 gt_format='npz',
                 num_workers=0):
        
        self.loader = DataLoader(dataset=RawDataset(
            img_paths, gt_path, 
            n_samples=n_samples,
            crop_with_torch=crop_with_torch,
            transformer=transformer,
            aug=aug,
            grayscale=grayscale,
            factor_crop=factor_crop,
            ratio=ratio, 
            seed=seed,
            truth_format=gt_format),
                                  shuffle=shuffle, 
                                  batch_size=batch_size,
                                  num_workers=num_workers)
        return self.loader