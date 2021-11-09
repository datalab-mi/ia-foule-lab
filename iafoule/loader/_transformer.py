import cv2
import torch
import numpy as np
import random
from torchvision.transforms.functional import adjust_gamma
from torchvision.transforms import FiveCrop, RandomHorizontalFlip, RandomCrop, TenCrop



class RandomImageCrop(object):
    """
    Apply a random crop
    
    args:
        - factor: int, factor rescale
        - size: tuple, height and with image
        - seed: int, set a seed, if None, no seed applyed.
                default None.
        - use_torch: bool, if true, use TenCrop function by 
                    pytorch else, use cv2 crop.
    """
    
    def __init__(self, factor=4, size=None, seed=None, use_torch=True):
        self.factor = factor
        self.size = size
        self.seed = seed
        self.use_torch = use_torch

    def __call__(self, image, target):
        assert np.sum(image.size) == np.sum(target.shape)
        
        if self.seed is not None:
            random.seed(self.seed)
        
        w, h = image.size
        if self.size is None:
            w, h = w // self.factor, h // self.factor
            self.size = (h, w)
        
        if self.use_torch:
            fc = TenCrop(size=self.size)
            rdm_idx = random.randint(0, 9)

            crop_img = fc(image)[rdm_idx]
            crop_den = fc(torch.from_numpy(target))[rdm_idx]
            return crop_img, crop_den.numpy()
        
        else:
            if random.random() >= 0.5:
                dx = int(random.randint(0, 1) * w)
                dy = int(random.randint(0, 1) * h)
            else:
                dx = int(random.random() * w)
                dy = int(random.random() * h)

            crop_img = image.crop((dx, dy, w + dx, h + dy))
            crop_den = target[dy: h + dy, dx: w + dx]
            return crop_img, crop_den
        
        
class RandomGammaCorrection(object):
    """"
    Apply Gamma Correction to the images
    """
    def __init__(self, p=0.3, seed=None):
        self.p = p
        self.seed = seed

    def __call__(self, image):
        
        if self.seed is not None:
            random.seed(self.seed)

        if random.random() > self.p:
            gammas = [0.5, 1.5]
            self.gamma = random.choice(gammas)
            return adjust_gamma(image, self.gamma, gain=1)
        else:
            return image
        
        

class RandomFlip(object):
    """
    Apply a random flip
    """
    
    def __init__(self, p=0.5, seed=None):
        self.p = p
        self.seed = seed
        
    def __call__(self, image, target):
        rf = RandomHorizontalFlip(p=1)
        if self.seed is not None:
            random.seed(self.seed)
            
        if random.random() > self.p:
            f_img = rf(image)
            f_den = rf(torch.from_numpy(target))
            return f_img, f_den.numpy()
        return image, target