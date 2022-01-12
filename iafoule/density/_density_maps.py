import json
import os
import pickle
import sys
from contextlib import closing
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy import spatial
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm as notebook_tqdm


def get_img_pathes(root_path,
                   extensions=['jpg', 'png', 'jpeg'],
                   **kwargs):
    """
    Return all images with extensions from all pathes in 'root_path'
    """
    img_pathes = []
    for ext in extensions:
        for img_path in Path(root_path).rglob(f'*.{ext}'):
            if '.ipynb_checkpoints' in str(img_path):  # Tips
                continue
            dict_path = {}
            path_img = str(img_path.parent)
            file_img = str(img_path.name)
            dict_path["path_img"] = os.path.join(path_img, file_img)
            dict_path["filename"] = img_path.stem

            if kwargs:
                images_folder = 'pngs'
                if kwargs["images_folder"] != '':
                    images_folder = kwargs["images_folder"]
                path_m = str(img_path.parent).replace(images_folder, kwargs["folder"])
                if kwargs["just_ext_added"]:
                    file_m = str(img_path.name) + kwargs["ext"]
                else:
                    file_m = str(img_path.name).replace(str(img_path.suffix), kwargs["ext"])
                dict_path["file_m"] = os.path.join(path_m, file_m)
                try:
                    dict_path["n_persons"] = len(get_gt_dots(dict_path["file_m"], file_type=kwargs["file_type"]))
                except FileNotFoundError:
                    print(f"File {dict_path['file_m']} not found")
                    continue
            img_pathes.append(dict_path)
    return pd.DataFrame(img_pathes)


def get_gt_dots(mat_path, file_type='matlab'):
    if file_type == 'matlab':
        """
        Load Matlab file with ground truth labels and save it to numpy array.
        ** cliping is needed to prevent going out of the array
        """
        mat = loadmat(mat_path)
        gt = mat["image_info"][0][0][0].astype(int)
    elif file_type == 'json':
        json_path = os.path.join(mat_path)
        with open(json_path) as f:
            js = json.load(f)
            points = js['points']
            properties = js['properties']
            height = properties['height']
            width = properties['width']
            gt = []
            for pt in points:
                x_ = round(pt['x'])
                y_ = round(pt['y'])
                if x_ < 0 or x_ > width or y_ < 0 or y_ > height:
                    print("POINT INCOHERENT")
                    print('width:', width, 'height:', height)
                    print('x_:', x_, 'y_:', y_)
                else:
                    gt.append((y_, x_))
            gt = np.array(gt)
    else:
        print('file type is unknown')
        sys.exit()
    return gt


def compute_distances(img_paths,
                      mat_paths,
                      out_folder_path='.',
                      out_dist_path='distances_dict.pkl',
                      n_neighbors=4,
                      save=True,
                      file_type='matlab'):
    # calculate distance for each images and each point between theirs n neighbors
    distances_dict = dict()

    for i_path, m_path in notebook_tqdm(zip(img_paths, mat_paths)):
        # load truth values
        # img = plt.imread(i_path)
        # height, width, dim = img.shape
        points = get_gt_dots(m_path, file_type=file_type)
        if len(points) > 0:
            # build tree distances with truth values and add to object
            tree = spatial.KDTree(points.copy())
            # query kdtree to get distance for each points
            distances, _ = tree.query(points, k=n_neighbors)
        else:
            distances = 0
        distances_dict[i_path] = distances

    if save:
        print(f'Distances computed for {len(distances_dict)} files')
        print(f'Saving them to {out_dist_path}')
        with open(os.path.join(out_folder_path, out_dist_path), 'wb') as f:
            pickle.dump(distances_dict, f)
    return distances_dict


def generate_gaussian_kernels(sigma_min=0,
                              sigma_max=20,
                              sigma_step=0.025,
                              sigma_threshold=4,
                              fixed_shape=None,
                              round_decimals=3,
                              out_kernels_path='gaussian_kernels.pkl',
                              path='.',
                              save=True):
    """
    Computing gaussian filter kernel for sigmas in arange(sigma_min, sigma_max, sigma_step) and saving
    them to dict.    
    """

    # generate multiple gaussien kernel for each sigma (sigma_min to sigma_max) 
    # with a threshold (size of the gaussian kernel)

    kernels_dict = dict()
    eps = 1e-4
    sigma_space = np.arange(sigma_min, sigma_max + eps, sigma_step)
    for sigma in notebook_tqdm(sigma_space):
        sigma = np.round(sigma, decimals=round_decimals)

        if fixed_shape is None:
            kernel_size = np.ceil(sigma * sigma_threshold).astype(int)
            img_shape = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        else:
            img_shape = fixed_shape

        img_center = (img_shape[0] // 2, img_shape[1] // 2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = gaussian_filter(arr, sigma, mode='constant')
        kernel = arr / arr.sum()
        kernels_dict[sigma] = kernel

    if save:
        print(f'Computed {len(sigma_space)} gaussian kernels.')
        print(f'Saving them to {out_kernels_path}')

        with open(os.path.join(path, out_kernels_path), 'wb') as f:
            pickle.dump(kernels_dict, f)

    return kernels_dict


def compute_sigma(gt_count,
                  distance=None,
                  n_neighbors=3,
                  min_sigma=1,
                  beta=0.1,
                  method=1,
                  fixed_sigma=15,
                  metric='mean'):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (sum of distance to 3 nearest neighbors) / 10
    * method = 2 : sigma = distance to nearest neighbor
    * method = 3 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    
    Method 1: adaptative kernel 
        NOTE: more neighbors are nearest, more sigma is lower (when lot of person are on 
              same place, we suggest there are heads)
    Method 2: adaptative kernel 
    Method 3: fixed kernel
    """
    if gt_count > 1 and distance is not None:
        if method == 1:
            # NOTE: paper use beta = 0.3 and neighbors = 3
            sigma = getattr(np, 'mean')(distance[1: 1 + n_neighbors]) * beta
        elif method == 2:
            sigma = distance[1]
        elif method == 3:
            sigma = fixed_sigma
    else:
        sigma = fixed_sigma
    if sigma < min_sigma:
        sigma = min_sigma
    return sigma


def find_closest_key(kernel_dict, num):
    """
    Find closest key and return this kernel
    """
    best_key = min(kernel_dict.keys(), key=lambda k: abs(k - num))
    return kernel_dict.get(num, kernel_dict[best_key])


def gaussian_filter_density(points,
                            map_h,
                            map_w,
                            distances=None,
                            kernels_dict=None,
                            min_sigma=2,
                            metric='mean',
                            n_neighbors=3,
                            method=1,
                            beta=0.1,
                            const_sigma=15):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    # number of person in the image (truth label)
    gt_count = points.shape[0]

    # initialize density map with black back
    density_map = np.zeros((map_h, map_w), dtype=np.float32)
    for i in range(gt_count):

        # get positon of truth person
        point_x, point_y = points[i]

        # set the sigma for one point (labelized person) with this neighbors if method 2 or 1 else use fixed sigma
        # dist_img = distances[i] if distances is not None else None
        if distances is not None:
            try:
                dist_img = distances[i]
            except Exception as e:
                # print("Erreur sur distances (i=", i, ") - " + str(e))
                pass
        else:
            dist_img = None
        sigma = compute_sigma(gt_count,
                              distance=dist_img,
                              min_sigma=min_sigma,
                              beta=beta,
                              metric=metric,
                              n_neighbors=n_neighbors,
                              method=method,
                              fixed_sigma=const_sigma)

        # get the closest sigma for using the precalculated kernel e.g: 4.78 -> 5 -> kernel with sigma = 5 
        kernel = find_closest_key(kernels_dict, sigma)

        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2

        # get the box where is the truth value depending on the kernel size and image borders
        min_img_x = max(0, point_x - kernel_size)
        min_img_y = max(0, point_y - kernel_size)
        max_img_x = min(point_x + kernel_size + 1, map_h - 1)
        max_img_y = min(point_y + kernel_size + 1, map_w - 1)

        kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
        kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        # add kernel computation at localization of truth values for each truth person
        # more person are in the same place, more values is highers
        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max,
                                                                 kernel_y_min:kernel_y_max]
    return density_map


def save_computed_density(density_map,
                          filename,
                          data_root='.',
                          method='sparse',
                          out_folder='maps_adaptive_kernel'):
    """
    Save density map to h5py format
    """
    full_path = os.path.join(data_root, out_folder)

    if not os.path.isdir(full_path):
        print(f'Creating {full_path}')
        os.makedirs(full_path)

    if method == 'h5':
        with closing(h5py.File(os.path.join(full_path, filename + '.h5'), 'w')) as hf:
            hf['density'] = density_map

    if method == 'sparse':
        scipy.sparse.save_npz(os.path.join(full_path, filename),
                              scipy.sparse.bsr_matrix(density_map))


class DensityMap:

    def __init__(self,
                 imgs_paths,
                 mats_paths,
                 data_path='.',
                 kernels=None,
                 **kwargs):

        self.data_path = data_path
        self.imgs_paths = imgs_paths
        self.mats_paths = mats_paths

        # load kernels 
        if isinstance(kernels, dict):
            self.kernels = kernels
        elif isinstance(kernels, str):
            try:
                with open(os.path.join(data_path, kernels), 'rb') as f:
                    self.kernels = pickle.loads(f.read())
            except FileNotFoundError:
                print(f"File {os.path.join(data_path, kernels)} not found")
                self.kernels = generate_gaussian_kernels(**kwargs, save=False)
        else:
            print('No precomputed kernels founds.')
            self.kernels = generate_gaussian_kernels(**kwargs, save=False)

    def generate_density_map(self,
                             method=3,
                             beta=0.1,
                             fixed_sigma=15,
                             min_sigma=0,
                             n_neighbors=3,
                             metric='mean',
                             distance=None,
                             save=False,
                             save_method='sparse',
                             map_out_folder='maps_adaptive_kernel',
                             file_type='matlab'):

        density_map_dict = {}
        # load distances if adaptative kernels
        if method in (1, 2):
            # load distance file
            if isinstance(distance, str):
                with open(os.path.join(self.data_path, distance), 'rb') as f:
                    distances_dict = pickle.loads(f.read())
            # load distance dict
            elif isinstance(distance, dict):
                distances_dict = distance
            # raise error if not exist
            else:
                raise TypeError("Les distances n'ont pas pu être chargées")

        for img_path, mat_path in notebook_tqdm(zip(self.imgs_paths, self.mats_paths)):
            # load img and truths values
            img = Image.open(img_path)
            width, height = img.size
            gt_points = get_gt_dots(mat_path, file_type=file_type)

            if method in (1, 2):
                distance = distances_dict[img_path]

            # compute density map and save it
            density_map = gaussian_filter_density(points=gt_points,
                                                  map_h=height,
                                                  map_w=width,
                                                  n_neighbors=n_neighbors,
                                                  distances=distance,
                                                  metric=metric,
                                                  const_sigma=fixed_sigma,
                                                  kernels_dict=self.kernels,
                                                  min_sigma=min_sigma,
                                                  method=method,
                                                  beta=beta)
            filename = str(Path(img_path).stem)

            if save:
                save_computed_density(density_map,
                                      filename=filename,
                                      method=save_method,
                                      data_root=self.data_path,
                                      out_folder=map_out_folder)

            density_map_dict[filename] = density_map
        return density_map_dict


def show_image_with_density(density_map, df_paths, label=None, show_image=True, save_file_directory=None,
                            save_file_suffix=None):
    if label is None:
        label = np.random.choice(list(density_map.keys()))
    path_img = df_paths[df_paths.filename.astype(str) == label].iloc[0].path_img
    if show_image:
        alpha = 0.7
        plt.imshow(plt.imread(path_img), alpha=alpha)
    plt.imshow(density_map[label], alpha=alpha)
    if save_file_directory is not None:
        save_file_path = 'density_map.png'
        if save_file_suffix is not None:
            save_file_path = 'density_map_' + save_file_suffix + '.png'
        plt.savefig(os.path.join(save_file_directory, save_file_path))


def load_sparse(filename):
    return scipy.sparse.load_npz(filename).toarray()
