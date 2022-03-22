import math

import numpy as np


def get_metrics(prediction, ground_truth, metric_grids=None):
    """
    get metrics between prediction and ground truth for following types (int, float, np array)
    metrics are absolute_error (AE), absolute_percentage_error (APE), squared_error (SE)
    and for metrics grid (only for np array) gridNxN_absolute_percentage_error (GAPE) and  gridNxN_cell_absolute_error (GCAE) 
    args:
        - prediction: prediction value (int, float, np array)
        - ground_truth: ground truth value (int, float, np array)
        - metric_grids: List of tuple [(2,2),(4,4)], default None.
    """
    
    metrics = dict()
    if metric_grids is None:
        metric_grids = []
    if (isinstance(prediction, int) and isinstance(prediction, int)) or (
            isinstance(ground_truth, float) and isinstance(ground_truth, float)):
        metrics['error'] = prediction - ground_truth
        nb_total = ground_truth
        metric_grids = []
    else:
        metrics['error'] = prediction.sum() - ground_truth.sum()
        nb_total = ground_truth.sum()
    if nb_total == 0:
        nb_total = 1
    metrics['absolute_error'] = np.abs(metrics['error'])
    metrics['absolute_percentage_error'] = 100. * metrics['absolute_error'] / nb_total
    metrics['squared_error'] = metrics['error'] * metrics['error']
    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        gape, gcae = get_grid_metrics(prediction, ground_truth, metric_grid)
        metrics['grid' + str_metric_grid + '_absolute_percentage_error'] = gape
        metrics['grid' + str_metric_grid + '_cell_absolute_error'] = gcae
    return metrics


def get_metrics_with_points(prediction, ground_truth, metric_grids=None):
    """
    get metrics between prediction (np array) and ground truth (list of points)
    metrics are absolute_error (AE), absolute_percentage_error (APE), squared_error (SE)
    and for metrics grid (only for np array) gridNxN_absolute_percentage_error (GAPE) and  gridNxN_cell_absolute_error (GCAE) 
    args:
        - prediction: prediction value (np array)
        - ground_truth: list of points [(x1,y1), (x2,y2),...]
        - metric_grids: List of tuple [(2,2),(4,4)], default None.
    """
    
    metrics = dict()
    nb_total = len(ground_truth)
    metrics['error'] = prediction.sum() - nb_total
    metrics['absolute_error'] = np.abs(metrics['error'])
    if nb_total == 0:
        nb_total = 1
    metrics['absolute_percentage_error'] = 100. * metrics['absolute_error'] / nb_total
    metrics['squared_error'] = metrics['error'] ** 2
    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        gape, gcae = get_grid_metrics_with_points(prediction, ground_truth, metric_grid)
        metrics['grid' + str_metric_grid + '_absolute_percentage_error'] = gape
        metrics['grid' + str_metric_grid + '_cell_absolute_error'] = gcae
    return metrics


def get_grid_metrics(prediction_map, ground_truth_map, metric_grid):
    """
    get grid metrics between prediction (np array) and ground truth (np array)
    metrics are gridNxN_absolute_percentage_error (GAPE) and  gridNxN_cell_absolute_error (GCAE) 
    args:
        - prediction_map (array) : prediction value
        - ground_truth_map (array) : ground_truth value
        - metric_grids: List of tuple [(2,2),(4,4)], default None.
    """

    matrix_prediction_map = np.zeros(metric_grid)

    pm_width = prediction_map.shape[1]
    pm_height = prediction_map.shape[0]

    n_w = int(math.ceil(pm_width / metric_grid[0]))
    n_h = int(math.ceil(pm_height / metric_grid[1]))
    for iw in range(metric_grid[0]):

        x_start = iw * n_w
        x_stop = (iw + 1) * n_w
        if x_stop > pm_width:
            x_stop = pm_width

        for ih in range(metric_grid[1]):

            y_start = ih * n_h
            y_stop = (ih + 1) * n_h
            if y_stop > pm_height:
                y_stop = pm_height

            sub_prediction_map = prediction_map[y_start:y_stop, x_start:x_stop]
            matrix_prediction_map[iw, ih] = sub_prediction_map.sum()

    matrix_ground_truth_map = np.zeros(metric_grid)

    gtm_width = ground_truth_map.shape[1]
    gtm_height = ground_truth_map.shape[0]

    n_w = int(math.ceil(gtm_width / metric_grid[0]))
    n_h = int(math.ceil(gtm_height / metric_grid[1]))
    for iw in range(metric_grid[0]):

        x_start = iw * n_w
        x_stop = (iw + 1) * n_w
        if x_stop > gtm_width:
            x_stop = gtm_width

        for ih in range(metric_grid[1]):

            y_start = ih * n_h
            y_stop = (ih + 1) * n_h
            if y_stop > gtm_height:
                y_stop = gtm_height

            sub_ground_truth_map = ground_truth_map[y_start:y_stop, x_start:x_stop]
            matrix_ground_truth_map[iw, ih] = sub_ground_truth_map.sum()

    matrix_difference = matrix_prediction_map - matrix_ground_truth_map

    matrix_final = matrix_difference.round()
    matrix_final = np.absolute(matrix_final)

    gt_nb_person = ground_truth_map.sum()
    if gt_nb_person == 0:
        gt_nb_person = 1

    # grid absolute percentage error
    gape = 100. * matrix_final.sum() / gt_nb_person

    # grid cell absolute error
    gcae = (matrix_final.sum() / metric_grid[0] / metric_grid[1]).round()

    return gape, gcae


def get_grid_metrics_with_points(prediction_map, ground_truth_points, metric_grid):
    """
    get grid metrics between prediction (np array) and ground truth (list of points)
    metrics gridNxN_absolute_percentage_error (GAPE) and  gridNxN_cell_absolute_error (GCAE) 
    args:
        - prediction_map (array) : prediction value
        - ground_truth_map (array) : ground_truth value
        - metric_grids: List of tuple [(2,2),(4,4)], default None.
    """

    matrix_ground_truth_points = np.zeros(metric_grid)

    width = prediction_map.shape[1]
    height = prediction_map.shape[0]

    n_w = int(math.ceil(width / metric_grid[0]))
    n_h = int(math.ceil(height / metric_grid[1]))
    for iw in range(metric_grid[0]):

        x_start = iw * n_w
        x_stop = (iw + 1) * n_w
        if x_stop > width:
            x_stop = width

        for ih in range(metric_grid[1]):

            y_start = ih * n_h
            y_stop = (ih + 1) * n_h
            if y_stop > height:
                y_stop = height

            nb_points = 0
            for (xx, yy) in ground_truth_points:
                if xx < x_start or xx >= x_stop or yy < y_start or yy >= y_stop:
                    continue
                nb_points += 1
            matrix_ground_truth_points[iw, ih] = nb_points

    matrix_prediction_map = np.zeros(metric_grid)

    pm_width = prediction_map.shape[1]
    pm_height = prediction_map.shape[0]

    n_w = int(math.ceil(pm_width / metric_grid[0]))
    n_h = int(math.ceil(pm_height / metric_grid[1]))
    for iw in range(metric_grid[0]):

        x_start = iw * n_w
        x_stop = (iw + 1) * n_w
        if x_stop > pm_width:
            x_stop = pm_width

        for ih in range(metric_grid[1]):

            y_start = ih * n_h
            y_stop = (ih + 1) * n_h
            if y_stop > pm_height:
                y_stop = pm_height

            sub_prediction_map = prediction_map[y_start:y_stop, x_start:x_stop]
            matrix_prediction_map[iw, ih] = sub_prediction_map.sum()


    matrix_difference = matrix_prediction_map - matrix_ground_truth_points

    matrix_final = matrix_difference.round()
    matrix_final = np.absolute(matrix_final)

    gt_nb_person = len(ground_truth_points)
    if gt_nb_person == 0:
        gt_nb_person = 1

    # grid absolute percentage error
    gape = 100. * matrix_final.sum() / gt_nb_person

    # grid cell absolute error
    gcae = (matrix_final.sum() / metric_grid[0] / metric_grid[1]).round()

    return gape, gcae


def get_benchmark_metrics(data_iterator, metric_grids=None):
    """
    get metrics for a bench of values given by an iterator l
    
    metrics are :
    mean_absolute_error (MAE)
    mean_absolute_percentage_error (MAPE)
    root_mean_squared_error (RMSE) 
    and for grid metrics :
    mean gridNxN_absolute_percentage_error (MGAPE) 
    gridNxN_cell_absolute_error (MGCAE) 
    args:
        - data_iterator (iterator) : The iterator must flow list of predictions, ground truths and precise the type of ground truth
            
            def data_iterator(predictions, ground_truths, ground_truth_type):
                for i, value in enumerate(predictions):
                    prediction = value
                    ground_truth = ground_truths[i]
                    yield {'ground_truth_type':ground_truth_type,
                           'prediction':prediction,
                           'ground_truth':ground_truth}
        
        prediction type     ground truth                                 ground_truth_type             metric_grids
         np array            np arrray                                      density_maps                 List of tuple [(2,2),(4,4)], default None.
         np array            list of points [(x1,y1), (x2,y2),...]          points                       List of tuple [(2,2),(4,4)], default None.
         float               float                                          values                       None
         int                 int                                            values                       None
        - ground_truth_map (array) : ground_truth value
        - metric_grids: List of tuple [(2,2),(4,4)], default None.
    """
    grid_metrics_initialized = False
    metrics = {'mean_absolute_error': [], 'mean_absolute_percentage_error': [], 'root_mean_squared_error': []}
    for data in data_iterator:
        prediction = data['prediction']
        ground_truth = data['ground_truth']
        ground_truth_type = 'values'
        if 'ground_truth_type' in data:
            ground_truth_type = data['ground_truth_type']
        if ground_truth_type == 'values' or metric_grids is None:
            metric_grids = []
        elif not grid_metrics_initialized:
            for metric_grid in metric_grids:
                str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
                m1 = 'grid' + str_metric_grid + '_absolute_percentage_error'
                m2 = 'grid' + str_metric_grid + '_cell_absolute_error'
                metrics['mean_' + m1] = []
                metrics['mean_' + m2] = []
            grid_metrics_initialized = True
        if ground_truth_type == 'points':
            internal_metrics = get_metrics_with_points(prediction,
                                                       ground_truth,
                                                       metric_grids=metric_grids)
        elif ground_truth_type in ['density_maps', 'values']:  # density_maps or int or float
            internal_metrics = get_metrics(prediction,
                                           ground_truth,
                                           metric_grids=metric_grids)
        metrics['mean_absolute_error'].append(internal_metrics['absolute_error'])
        metrics['mean_absolute_percentage_error'].append(internal_metrics['absolute_percentage_error'])
        metrics['root_mean_squared_error'].append(internal_metrics['squared_error'])
        for metric_grid in metric_grids:
            str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
            m1 = 'grid' + str_metric_grid + '_absolute_percentage_error'
            m2 = 'grid' + str_metric_grid + '_cell_absolute_error'
            metrics['mean_' + m1].append(internal_metrics[m1])
            metrics['mean_' + m2].append(internal_metrics[m2])

    metrics['mean_absolute_error'] = np.mean(metrics['mean_absolute_error'])
    metrics['mean_absolute_percentage_error'] = np.mean(metrics['mean_absolute_percentage_error'])

    metrics['root_mean_squared_error'] = np.mean(metrics['root_mean_squared_error'])
    metrics['root_mean_squared_error'] = math.sqrt(metrics['root_mean_squared_error'])

    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        m1 = 'mean_grid' + str_metric_grid + '_absolute_percentage_error'
        m2 = 'mean_grid' + str_metric_grid + '_cell_absolute_error'
        metrics[m1] = np.mean(metrics[m1])
        metrics[m2] = np.mean(metrics[m2])

    return metrics
