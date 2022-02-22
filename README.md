# Comptage de Foule 

Ce package permet l'implémentation de modèle de comptage de foule.

## Création des carte de densités

Pour la création des cartes de densités, la méthode suivante peut être appliquée : 

Lecture du **Dataset GCC** avec les metadatas

```python
from iafoule.density import get_img_pathes

data_path = "/workspace/data/GCC/"
params = {
        'folder': 'jsons',
        'ext' : '.json', 
        'metadata_weather': 'weather',
        'metadata_time_info': 'timeInfo'
         }
df_path = get_img_pathes(root_path=data_path, **params)
```


Sauvegarde des *kernels* gaussiens et des distances précalculés

```python
from iafoule.density import generate_gaussian_kernels, compute_distances

distances = compute_distances(img_paths=df_path.path_img,
                              gt_paths=df_path.file_m,
                              out_dist_path='distance.pkl',
                              n_neighbors=7,
                              save=True)

kernels_dict = generate_gaussian_kernels(sigma_min=0,
                                         sigma_max=20,
                                         sigma_step=0.025,
                                         sigma_threshold=4,
                                         fixed_shape=None,
                                         round_decimals=3,
                                         out_kernels_path='gk.pkl',
                                         save=True)
```

Génération des *density map*

```python

from iafoule.density import DensityMap, show_image_with_density

params_1 = {'method': 1, 
            'distance': 'distance.pkl',
            'min_sigma' : 0,
            'metric': 'mean',
            'beta': 0.3,
            'n_neighbors': 3}

dense_maps = dm.generate_density_map(**params_1, 
                                     save_method='sparse',
                                     save=True,
                                     map_out_folder='maps_adaptive_kernel')
```

Possibilité de faire varier les paramètres et de visualiser les *density maps* créées avec la fonction `show_image_with_density`

## Utilisation du DataLoader



