import math

import numpy as np


def get_benchmark_metrics(data_iterator, metrics_list=None, metric_grids=None, debug=False):
    # default metric : mean_absolute_error
    # you must ask for root_mean_squared_error and by metric_grids pameters all the grid-metrics
    if metrics_list is None:
        metrics_list = ['mean_absolute_error']
    if metric_grids is None:
        metric_grids = []
    metrics = {'mean_absolute_error': []}
    internal_metrics_list = ['absolute_error']
    if 'root_mean_squared_error' in metrics_list:
        metrics['root_mean_squared_error'] = []
        internal_metrics_list.append('squared_error')
    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        metrics['mean_grid' + str_metric_grid + '_absolute_percentage_error'] = []
        metrics['mean_grid' + str_metric_grid + '_cell_percentage_error'] = []
    for data in data_iterator:
        data_type = 'densities'
        if 'type' in data and data['type'] in ['densities', 'density_points']:
            data_type = data['type']
        prediction_map = data['prediction_map']
        ground_truth = data['ground_truth']
        if data_type == 'densities':
            internal_metrics = get_metrics(prediction_map,
                                           ground_truth,
                                           metrics_list=internal_metrics_list,
                                           metric_grids=[],
                                           debug=debug)
        else:  # density_points
            internal_metrics = get_metrics_with_points(prediction_map,
                                                       ground_truth,
                                                       metrics_list=internal_metrics_list,
                                                       metric_grids=[],
                                                       debug=debug)
        metrics['mean_absolute_error'].append(internal_metrics['absolute_error'])
        metrics['root_mean_squared_error'].append(internal_metrics['squared_error'])
        for metric_grid in metric_grids:
            str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
            m1 = 'grid' + str_metric_grid + '_absolute_percentage_error'
            m2 = 'grid' + str_metric_grid + '_cell_percentage_error'
            metrics['mean_' + m1].append(internal_metrics[m1])
            metrics['mean_' + m2].append(internal_metrics[m2])

    metrics['mean_absolute_error'] = np.mean(metrics['mean_absolute_error'])

    metrics['root_mean_squared_error'] = np.mean(metrics['root_mean_squared_error'])
    metrics['root_mean_squared_error'] = math.sqrt(metrics['root_mean_squared_error'])

    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        m1 = 'mean_grid' + str_metric_grid + '_absolute_percentage_error'
        m2 = 'mean_grid' + str_metric_grid + '_cell_percentage_error'
        metrics[m1] = np.mean(metrics[m1])
        metrics[m2] = np.mean(metrics[m2])

    return metrics


def get_metrics(prediction_map, ground_truth_map, metrics_list=None, metric_grids=None, debug=False):
    if metrics_list is None:
        metrics_list = ['error']
    if metric_grids is None:
        metric_grids = []
    metrics = dict()
    metrics['error'] = ground_truth_map.sum() - prediction_map.sum()
    if 'absolute_error' in metrics_list:
        metrics['absolute_error'] = np.abs(metrics['error'])
    if 'squared_error' in metrics_list:
        metrics['squared_error'] = metrics['error'] * metrics['error']
    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        gape, gcae = get_grid_metrics(prediction_map, ground_truth_map, metric_grid, debug=debug)
        metrics['grid' + str_metric_grid + '_absolute_percentage_error'] = gape
        metrics['grid' + str_metric_grid + '_cell_percentage_error'] = gape
    return metrics


def get_grid_metrics(prediction_map, ground_truth_map, metric_grid, debug=False):
    if debug:
        print('metric_grid:', metric_grid)
        print("prediction_map(sum):", prediction_map.sum())
        print('prediction_map.shape', prediction_map.shape)
        print('prediction_map', prediction_map)
        print("ground_truth_map(sum):", ground_truth_map.sum())
        print('ground_truth_map.shape:', ground_truth_map.shape)
        print('ground_truth_map:', ground_truth_map)

    matrix_prediction_map = np.zeros(metric_grid)

    pm_width = prediction_map.shape[1]
    pm_height = prediction_map.shape[0]
    if debug:
        print('pm_width:', pm_width)
        print('pm_height:', pm_height)

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
            if debug:
                print('iw:', iw, 'x_start:', x_start, 'x_stop:', x_stop, 'ih:', ih, 'y_start:', y_start, 'y_stop:',
                      y_stop)
                print("sub_prediction_map(sum):", sub_prediction_map.sum())

    matrix_ground_truth_map = np.zeros(metric_grid)

    gtm_width = ground_truth_map.shape[1]
    gtm_height = ground_truth_map.shape[0]
    if debug:
        print('gtm_width:', gtm_width)
        print('gtm_height:', gtm_height)

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
            if debug:
                print('iw:', iw, 'x_start:', x_start, 'x_stop:', x_stop, 'ih:', ih, 'y_start:', y_start, 'y_stop:',
                      y_stop)
                print("sub_ground_truth_map(sum):", sub_ground_truth_map.sum())

    if debug:
        print('matrix_ground_truth_map:', matrix_ground_truth_map)
        print('matrix_prediction_map:', matrix_prediction_map)

    matrix_difference = matrix_ground_truth_map - matrix_prediction_map

    if debug:
        print('matrix_difference:', matrix_difference)

    matrix_final = matrix_difference.round()
    matrix_final = np.absolute(matrix_final)

    if debug:
        print('matrix_final:', matrix_final)

    gt_nb_person = ground_truth_map.sum()
    if gt_nb_person == 0:
        gt_nb_person = 1
    if debug:
        print('gt_nb_person:', gt_nb_person)

    # grid absolute percentage error
    gape = matrix_final.sum() / gt_nb_person
    if debug:
        print('gape:', gape)

    # grid cell absolute error
    gcae = (matrix_final.sum() / metric_grid[0] / metric_grid[1]).round()
    if debug:
        print('gcae:', gcae)

    return gape, gcae


def get_metrics_with_points(prediction_map, ground_truth_points, metrics_list=None, metric_grids=None, debug=False):
    if metrics_list is None:
        metrics_list = ['error']
    if metric_grids is None:
        metric_grids = []
    metrics = dict()
    metrics['error'] = len(ground_truth_points) - prediction_map.sum()
    if 'absolute_error' in metrics_list:
        metrics['absolute_error'] = np.abs(metrics['error'])
    if 'squared_error' in metrics_list:
        metrics['squared_error'] = metrics['error'] ** 2
    for metric_grid in metric_grids:
        str_metric_grid = str(metric_grid[0]) + 'x' + str(metric_grid[1])
        gape, gcae = get_grid_metrics_with_points(prediction_map, ground_truth_points, metric_grid, debug=debug)
        metrics['grid' + str_metric_grid + '_absolute_percentage_error'] = gape
        metrics['grid' + str_metric_grid + '_cell_percentage_error'] = gape
    return metrics


def get_grid_metrics_with_points(prediction_map, ground_truth_points, metric_grid, debug=False):
    # Attention si prediction_map plus petite que l'image initiale, ca va foirer

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
                if debug:
                    print("point:", xx, yy)
                nb_points += 1
            matrix_ground_truth_points[iw, ih] = nb_points
            if debug:
                print('iw:', iw, 'x_start:', x_start, 'x_stop:', x_stop, 'ih:', ih, 'y_start:', y_start, 'y_stop:',
                      y_stop)
                print("nb_points:", nb_points)

    matrix_prediction_map = np.zeros(metric_grid)

    pm_width = prediction_map.shape[1]
    pm_height = prediction_map.shape[0]
    if debug:
        print('pm_width:', pm_width)
        print('pm_height:', pm_height)

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
            if debug:
                print('iw:', iw, 'x_start:', x_start, 'x_stop:', x_stop, 'ih:', ih, 'y_start:', y_start, 'y_stop:',
                      y_stop)
                print("sub_prediction_map(sum):", sub_prediction_map.sum())

    if debug:
        print('matrix_ground_truth_points:', matrix_ground_truth_points)
        print('matrix_prediction_map:', matrix_prediction_map)

    matrix_difference = matrix_ground_truth_points - matrix_prediction_map

    if debug:
        print('matrix_difference:', matrix_difference)

    matrix_final = matrix_difference.round()
    matrix_final = np.absolute(matrix_final)

    if debug:
        print('matrix_final:', matrix_final)

    gt_nb_person = len(ground_truth_points)
    if gt_nb_person == 0:
        gt_nb_person = 1
    if debug:
        print('gt_nb_person:', gt_nb_person)

    # grid absolute percentage error
    gape = matrix_final.sum() / gt_nb_person
    if debug:
        print('gape:', gape)

    # grid cell absolute error
    gcae = (matrix_final.sum() / metric_grid[0] / metric_grid[1]).round()
    if debug:
        print('gcae:', gcae)

    return gape, gcae
