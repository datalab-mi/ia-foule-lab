import torch
import logging
from tqdm import tqdm
import matplotlib as plt
import numpy as np
import cv2


def plot_counting(im, ds, grayscale=True, alpha=0.5):
    """
    Plot a tensor images with alpha with this tensor density map and counting people
    """
    n_dim = 1 if grayscale else 3
    plt.imshow(ds[0].numpy().reshape(ds.shape[2], ds.shape[3], 1))
    plt.imshow(im[0].numpy().reshape(im.shape[2], im.shape[3], n_dim), alpha=alpha)
    plt.axis('off')
    plt.text(y=20, 
             x=20, 
             s=f'Personnes : {int(ds[0].numpy().reshape(ds.shape[2], ds.shape[3], 1).sum())}',
             c='white')


def normalize_target(output, target, gpu=False):
    """
    Normalize target density map to the shape of output model
    """
    if gpu:
        output = output.cpu()
        target = target.cpu()
    # shape output
    ht, wd = output[0][0].detach().numpy().shape
    # shape target
    gt_ht, gt_wd = target[0][0].detach().numpy().shape
    # reshape target to shape output
    den = np.expand_dims(cv2.resize(target[0][0].numpy(), (wd, ht), interpolation=cv2.INTER_AREA), axis=(0, 1))
    # get same sum density map
    den = den * ((gt_ht * gt_wd) / (wd * ht))
    target = torch.from_numpy(den)
    if gpu:
        target = target.cuda()
    return target


def get_logger(filename):
    logger = logging.getLogger('train_logger')

    while logger.handlers:
        logger.handlers.pop()

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, 'w')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('[%(asctime)s], ## %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def compute_lc_loss(output, target, sizes=(1, 2, 4)):
    criterion_L1 = torch.nn.L1Loss(reduction='sum')
    lc_loss = None
    for s in sizes:
        pool = torch.nn.AdaptiveAvgPool2d(s)
        est = pool(output)
        gt = pool(target)
        if lc_loss:
            lc_loss += criterion_L1(est, gt) / s**2
        else:
            lc_loss = criterion_L1(est, gt) / s**2
    return lc_loss


def get_mean_and_std_by_channel(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for i, data in tqdm(enumerate(loader, 0)):

        img, gt_map = data
        if img is None:
            continue
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(img ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return list(mean.numpy()), list(std.numpy())


"""
def get_mean_and_std_by_channel_2(loader):
    # Compute the mean and sd in an online fashion
    # Var[x] = E[X^2] - E^2[X]
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for i, data in tqdm(enumerate(loader, 0)):

        img, gt_map = data
        if img is None:
            continue
        b, c, h, w = img.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return list(mean.numpy()), list(std.numpy())
"""