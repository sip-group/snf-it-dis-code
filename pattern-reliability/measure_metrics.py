import os
import argparse

from lib.Dataset_cdp import DatasetCDP
from lib.predictor_functions import pool_predict, apply_otsu_threshold
from lib.cdp_metrics import batch_metric

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('-n','--nbsamples', help='An integer giving the number of training samples',
                    type=int,
                    choices=range(1,1441),
                    required=True)

parser.add_argument('-t','--train_data', help='Either "iphone" or "samsung"',
                    choices=['iphone', 'samsung'],
                    required=True)

parser.add_argument('-d','--dataset', help='Either "iphone" or "samsung"',
                    choices=['iphone', 'samsung'],
                    required=True)

parser.add_argument('-r','--run', help='An integer between 1 and 6 giving the print-scan session of the dataset.',
                    type=int,
                    choices=range(1,7),
                    required=True)

parser.add_argument('-p','--processing', help='A string describing which processing to apply to the samples',
                    choices=['no_processing', 'stretch', 'normalize'],
                    default='no_processing')

parser.add_argument('-c','--cores', help='An integer fixing the number of cores used for multiprocessing.',
                    type=int,
                    default=1)

parser.add_argument('-s','--seed', help='An integer fixing the seed for the random selection of the training set.',
                    type=int,
                    required=True)

parser.add_argument('-v','--verbose', help='A parameter asking the program to print its progress in the std output.',
                    action='store_true')

args = vars(parser.parse_args())


if __name__ == '__main__':

    nb_samples = args['nbsamples']
    run = args['run']

    train_dataset = args['train_data']
    dataset = args['dataset']
    processing = args['processing']

    metrics = ['lls', 'dhamm', 'mse', 'l1', 'pcor']
    thresholds = np.concatenate([np.round(k * np.arange(1,10), 8) for k in [1e-4, 1e-3, 1e-2, 1e-1]])
    thresholds = np.append(thresholds, [1.])

    cores = args['cores']
    seed = args['seed']
    verbose = args['verbose']

    save_path = os.path.join('results', 'metrics')

    ######################################################
    ################## CHECK CODEBOOK ####################
    ######################################################

    if verbose:
        print('Load trained codebook')

    codebook_name = f'run_{run}_samples_{nb_samples}_otsu_seed_{seed}.json'
    codebook_path = os.path.join('results', 'codebooks', 'codebook_measures', train_dataset, 'original', codebook_name)

    assert os.path.exists(codebook_path), print(f'No trained codebook {codebook_name}.'
                                                f'Please first run measure_codebook.py with corresponding parameters.')

    with open(codebook_path, "r") as fp:
        codebook_train = np.asarray(json.load(fp))

    ######################################################
    ################## LOADING DATASET ###################
    ######################################################

    if verbose:
        print(f'Loading dataset {dataset} run {run}')

    dset = DatasetCDP(dataset_name=dataset, run=run, nb_samples=1440, hist_match=True)

    loader = DataLoader(dset, batch_size=len(dset), shuffle=False)

    for data in loader:
        for key in data.keys():
            data[key] = np.asarray(data[key])

    possible_choices = list(np.arange(len(dset)))

    np.random.seed(seed)
    random_perm = np.random.permutation(possible_choices)

    if verbose:
        print(f'Seed choice : {random_perm[:5]}')

    for key in data.keys():
        data[key] = data[key][random_perm[nb_samples:nb_samples+500]]

    ######################################################
    ################# MEASURING THE MASK #################
    ######################################################

    if verbose:
        print('Applying the codebook to the dataset')

    data['t_x'] = apply_otsu_threshold(data['x'], block_size=1, verbose=False)
    data['t_f'] = apply_otsu_threshold(data['f'], block_size=1, verbose=False)

    data['biterror_pred_x'] = pool_predict(data['t'] > .5, codebook_train, nb_cores=cores)

    ######################################################
    ################# MEASURING METRICS ##################
    ######################################################

    if verbose:
        print(f'Measuring the masked metrics')

    measures = {metric: {} for metric in metrics}

    for metric in metrics:
        for mu in thresholds:
            measures[metric][mu] = {'x': [], 'f': []}

    for metric in metrics:
        for key in ['x', 'f']:

            if verbose:
                print(f'Computing metric {metric} {key}')

            for mu in thresholds:

                if metric == 'lls':
                    eps = data['biterror_pred_x']
                    batch_t = (1-eps) * data['t'] + eps * (1 - data['t'])
                    batch_y = data[f't_{key}']

                elif metric == 'dhamm':
                    batch_t = (data['t'] > .5)
                    batch_y = data[f't_{key}']

                else:
                    batch_t = data['t']*1
                    batch_y = data[key]

                measures[metric][mu][key] = batch_metric(
                    batch_t,
                    batch_y,
                    metric,
                    k=1,
                    w=np.logical_or(data['biterror_pred_x'] <= mu, False),
                    mode=processing,
                    nb_cores=cores)


    ######################################################
    ################### COMPUTING ROCS ###################
    ######################################################

    if verbose:
        print('Computing ROC curves')

    roc_curves = {metric: {} for metric in metrics}
    roc_auc_scores = {metric: {} for metric in metrics}

    for metric in metrics:
        for mu in thresholds:
            roc_curves[metric][mu] = []

    for metric in metrics:
        y_true = np.concatenate([np.ones(data['x'].shape[0]), np.zeros(data['f'].shape[0])])

        for mu in thresholds:
            score = np.concatenate([measures[metric][mu]['x'], measures[metric][mu]['f']])
            fpr, tpr, gamma = roc_curve(y_true, score)
            if metric != 'pcor':
                roc_curves[metric][mu] = (1-fpr, 1-tpr, gamma)
            else:
                roc_curves[metric][mu] = (fpr, tpr, gamma)

            roc_auc_scores[metric][mu] = roc_auc_score(y_true, score)


    ######################################################
    ################## SAVING RESULTS ####################
    ######################################################

    if verbose:
        print(f'Saving measures in {save_path}')

    measures_json = {metric: {} for metric in metrics}
    roc_curves_json = {metric: {} for metric in metrics}

    for metric in metrics:
        for mu in thresholds:
            measures_json[metric][mu] = {'x': [], 'f': []}
            roc_curves_json[metric][mu]= []

    for metric in metrics:
        for mu in measures[metric].keys():
            measures_json[metric][mu] = {'x': [], 'f': []}
            roc_curves_json[metric][mu] = []

            measures_json[metric][mu]['x'] = measures[metric][mu]['x'].tolist()
            measures_json[metric][mu]['f'] = measures[metric][mu]['f'].tolist()

            fpr, tpr, gamma = roc_curves[metric][mu]
            roc_curves_json[metric][mu] = (fpr.tolist(), tpr.tolist(), gamma.tolist())

    fn = f'train_{train_dataset}_{nb_samples}_run_{run}_otsu_{processing}_seed_{seed}.json'

    curr_path = os.path.join(save_path, 'measures_data', dataset)
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    with open(os.path.join(curr_path, fn), "w") as fp:
        json.dump(measures_json, fp)

    curr_path = os.path.join(save_path, 'roc_curves', dataset)
    if not os.path.exists(curr_path):
        os.makedirs(curr_path)

    with open(os.path.join(curr_path, fn), "w") as fp:
        json.dump(roc_curves_json, fp)

    with open(os.path.join(curr_path, 'AUC_' + fn), "w") as fp:
        json.dump(roc_auc_scores, fp)
