import numpy as np
import scipy
import scipy.ndimage as ndimage
import scipy.spatial
import torch
import torchvision
import torch.nn as nn

def validation(model, test_loader, cuda=True):
    model.eval()
    mae = 0.0
    mse = 0.0
    with torch.no_grad():
        for img, target, count in test_loader:
            if cuda:
                img = img.cuda()
            output = model(img)
            est_count = output.sum().item()
            mae += abs(est_count - count)
            mse += (est_count - count) ** 2
    mae /= len(test_loader)
    mse /= len(test_loader)
    rmse = mse ** 0.5
    return float(mae), float(rmse)