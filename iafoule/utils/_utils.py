import torch
import torchvision

from torchvision import transforms
import matplotlib as plt


def plot_counting(im, ds, grayscale=True, alpha=0.5):
    """
    Plot a tensor images with alpha with this tensor density map and couting people
    """
    n_dim = 1 if grayscale else 3
    plt.imshow(ds[0].numpy().reshape(ds.shape[2], ds.shape[3], 1))
    plt.imshow(im[0].numpy().reshape(im.shape[2], im.shape[3], n_dim), alpha=alpha)
    plt.axis('off')
    plt.text(y=20, 
             x=20, 
             s=f'Personnes : {int(ds_t[0].numpy().reshape(ds_t.shape[2], ds_t.shape[3], 1).sum())}', 
             c='white');

def normalize_target(output, target):
    """
    Normalize output values to shape that target value
    """
    ht, wd = output[0][0].detach().numpy().shape
    den = np.expand_dims(cv2.resize(target[0][0].numpy(), (wd, ht)), axis=(0,1))
    return torch.from_numpy(den)


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