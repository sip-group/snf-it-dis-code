import os
import argparse

import numpy as np
from lib.Dataset_cdp import DatasetCDP
from lib.predictor_functions import batch_train_codebook, train_codebook
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import json


parser = argparse.ArgumentParser(description='Description of your program')

parser.add_argument('-n','--nbsamples', help='An integer giving the number of training samples',
                    type=int,
                    choices=range(1,1441),
                    required=True)

parser.add_argument('-d','--dataset', help='Either "iphone" or "samsung"',
                    choices=['iphone', 'samsung'],
                    required=True)

parser.add_argument('-r','--run', help='An integer between 1 and 6 giving the print-scan session of the dataset.',
                    type=int,
                    choices=range(1,7),
                    required=True)

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
    dataset = args['dataset']
    cores = args['cores']
    seed = args['seed']
    verbose = args['verbose']
    save_path = os.path.join('results', 'codebooks', 'codebook_measures', dataset)

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
        data[key] = data[key][random_perm[:nb_samples]]

    ######################################################
    ################# TRAINING CODEBOOK ##################
    ######################################################

    if verbose:
        print('Training the codebook on originals')

    local_prob_x, _ = batch_train_codebook((data['t'] > 0.5), data['x'], nb_cores=cores, estimator='otsu', block_size=1)

    if verbose:
        print('Training the codebook on fakes')

    local_prob_f, _ = batch_train_codebook((data['t'] > 0.5), data['f'], nb_cores=cores, estimator='otsu', block_size=1)

    hist = {'original': [], 'fake': []}

    for i in range(512):
        code = bin(i)[2:].zfill(9)
        hist['original'] += [local_prob_x['error'][code]]
        hist['fake'] += [local_prob_f['error'][code]]

    ######################################################
    ################## SAVING RESULTS ####################
    ######################################################

    if verbose:
        print(f'Saving codebooks in {save_path}')

    filename = f'run_{run}_samples_{nb_samples}_otsu_seed_{seed}.json'

    for y in ['original', 'fake']:

        curr_path = os.path.join(save_path, y)
        if not os.path.exists(curr_path):
            os.makedirs(curr_path)

        with open(os.path.join(curr_path, filename), "w") as fp:
            json.dump(hist[y], fp)

