import os

import numpy as np
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser

from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, roc_auc_score

import torch
from torch.nn import Sigmoid

import matplotlib.pyplot as plt


def is_experiment_folder(path):
    """Checks that the passed directory is an experiment directory by checking that at least one 'run' folder exists"""
    return np.any(['run' in fold and os.path.isdir(os.path.join(path, fold)) for fold in os.listdir(path)])


def get_subfolders(path, runs_only=False):
    """Return (absolute path) sub-folders of the given path"""
    result = []
    for fn in os.listdir(path):
        p = os.path.join(path, fn)
        if os.path.isdir(p):
            if not runs_only or 'run' in p:
                result.append(p)
    return result


def get_all_lvs(run_dir, train=False):
    """Loads latent vectors"""
    # Getting clusters
    lv_path = os.path.join(run_dir, 'lv')
    file_names = sorted(os.listdir(lv_path), reverse=True)
    file_names = [fn for fn in file_names if (train and 'train' in fn) or (not train and 'test' in fn)]
    clusters = [fn.split(".pt")[0] for fn in file_names if '.pt' in fn]

    # Collecting the latent vectors
    all_lvs = []
    for file_name in file_names:
        if '.pt' in file_name:
            all_lvs.append(torch.load(os.path.join(lv_path, file_name)))

    return all_lvs, clusters


def plot_histograms(distances, cluster_names, title):
    """Given distances, plots histogram of distributions"""
    values = []
    classes = []

    for tensor, cluster_name in zip(distances, cluster_names):
        t = tensor.flatten().numpy()
        values.extend(t)
        classes.extend([cluster_name] * len(t))

    data = pd.DataFrame({
        'values': values,
        'class': classes
    })

    sns.set_style()
    sns.histplot(x='values', hue='class', data=data, element='step', stat='probability', kde=True)
    plt.xlabel("L2 Distance from templates")
    plt.title(title)
    plt.show()


def scatter_plot(all_lvs, clusters, title, dimensions=2, discard_templates=True):
    # Printing mean distances
    for i in range(1, len(clusters)):
        dist = torch.norm(all_lvs[0] - all_lvs[i], dim=1)
        # print(f"Mean template-to-{clusters[i]} mean L2 distance: {torch.mean(dist):.3f}\tStd: {torch.std(dist):.3f}")

    # Running PCA
    pca = PCA(n_components=dimensions)
    all_coords = torch.vstack(all_lvs)
    new_coords = pca.fit(all_coords.T).components_.T

    if dimensions == 3:
        # Plotting in 3D
        assert ((len(new_coords) / len(clusters)) % 1 == 0)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        samples_per_cluster = len(new_coords) // len(clusters)
        for i, name in enumerate(clusters):
            if 'temp' in name and discard_templates:
                continue

            start = i * samples_per_cluster
            finish = (i + 1) * samples_per_cluster

            ax.scatter3D(new_coords[start:finish, 0],
                         new_coords[start:finish, 1],
                         new_coords[start:finish, 2],
                         label=name)
        plt.legend()
        plt.title(title)
        plt.show()
    else:
        samples_per_cluster = len(new_coords) // len(clusters)
        for i, name in enumerate(clusters):
            if 'temp' in name and discard_templates:
                continue

            start = i * samples_per_cluster
            finish = (i + 1) * samples_per_cluster

            sns.scatterplot(x=new_coords[start:finish, 0], y=new_coords[start:finish, 1], label=f"{name}")
        plt.legend()
        plt.title(title)
        plt.show()


def get_classification_stats(lvs, names, verbose=False):
    """Given classification scores, returns the AUC scores and FPR + FNR """
    sigmoid = Sigmoid()
    o_lvs = [[lvs[i], names[i]] for i in range(len(lvs)) if names[i].startswith('o')]
    f_lvs = [[lvs[i], names[i]] for i in range(len(lvs)) if names[i].startswith('f')]
    stats = []

    if verbose:
        print("Binary performance (assuming threshold = 0.5)")

    for o_lv, o_name in o_lvs:
        Pmiss = torch.sum(sigmoid(o_lv) < .5) / len(o_lv)
        for f_lv, f_name in f_lvs:
            Pfa = torch.sum(sigmoid(f_lv) >= .5) / len(f_lv)

            # Computing AUC
            y_true = [*([1] * len(o_lv)), *([0] * len(f_lv))]
            y_score = [*(sigmoid(o_lv).numpy()), *(sigmoid(f_lv).numpy())]
            auc = roc_auc_score(y_true, y_score)

            # Appending stats to result
            stats.append([o_name, f_name, Pmiss.item(), Pfa.item(), auc])

            if verbose:
                print(f"\t{o_name}-{f_name}\tPmiss: {Pmiss:.3f}\tPfa: {Pfa:.3f}\t(AUC-Score: {auc:.3f})")
    return stats


def get_tln_stats(lvs, names, verbose=False):
    # Getting L2 Distances
    distances = [torch.norm(lvs[0] - lvs[i], dim=1) for i in range(1, len(lvs))]
    n_orig = len([n for n in names if n.startswith('o_')])
    stats = []

    for o_lv, o_name in zip(distances[:n_orig], names[1:][:n_orig]):
        for f_lv, f_name in zip(distances[n_orig:], names[1:][n_orig:]):
            # Computing AUC
            y_true = [*([0] * len(o_lv)), *([1] * len(f_lv))]
            y_score = [*(o_lv.numpy()), *(f_lv.numpy())]
            auc = roc_auc_score(y_true, y_score)

            fpr, tpr, thresholds = roc_curve(y_true, y_score)

            # Getting the best possible threshold and Pmiss / Pfa
            t_idx = np.argmax(tpr - fpr)
            Pmiss, Pfa = 1 - tpr[t_idx], fpr[t_idx]

            # Appending stats to the result
            stats.append([o_name, f_name, Pmiss, Pfa, auc])

            if verbose:
                print(f"\t{o_name}-{f_name}\tPmiss: {Pmiss:.3f}\tPfa: {Pfa:.3f}\t(AUC-Score: {auc:.3f})")
    return stats


def get_tln_osvm_stats(lvs, names):
    # Dividing names
    o_names = [n for n in names if n.startswith("o_")]
    f_names = [n for n in names if n.startswith("f_")]

    # Getting latent vectors
    o_lvs = [lvs[i].numpy() for i in range(len(lvs)) if names[i].startswith('o_')]
    f_lvs = [lvs[i].numpy() for i in range(len(lvs)) if names[i].startswith('f_')]

    # Getting result
    result = []

    for o_lv, o_name in zip(o_lvs, o_names):
        # Fitting one-class SVM
        oc_svm = OneClassSVM().fit(o_lv)

        for f_lv, f_name in zip(f_lvs, f_names):
            # Getting latent vectors scores
            o_scores = oc_svm.score_samples(o_lv)
            f_scores = oc_svm.score_samples(f_lv)

            # Getting roc score
            labels = [1] * len(o_scores)
            labels.extend([0] * len(f_scores))
            auc = roc_auc_score(labels, [*o_scores, *f_scores])

            # Getting Pmiss and Pfa
            fpr, tpr, t = roc_curve(labels, [*o_scores, *f_scores])

            idx = np.argmax(tpr - fpr)
            Pmiss, Pfa = 1 - tpr[idx], fpr[idx]

            result.append([o_name, f_name, Pmiss, Pfa, auc])
    return result


def experiment_summary(run_path, train=False, plot=True):
    """Gets FNR and FPR for a single experiment (multiple runs), using both TLN and TLN + OSVM"""
    experiment_name = run_path.split('/')[-1]
    sub_folders = get_subfolders(run_path, runs_only=True)

    all_tln_stats, all_tln_osvm_stats = [], []
    all_class_stats = []

    print(f"\n\n\n\n\nEXPERIMENT: {experiment_name} ({len(sub_folders)} runs)")
    for run_nr, run_dir in enumerate(sub_folders):
        lvs, names = get_all_lvs(run_dir, train)
        d = len(lvs[0][0])

        if d > 1:
            # If a multi-dimensional vector is stored, print L2 distances and OSVM classification stats
            all_tln_stats.append(get_tln_stats(lvs, names))
            all_tln_osvm_stats.append(get_tln_osvm_stats(lvs, names))

            if plot:
                # Plotting histogram of distances
                distances = [torch.norm(lvs[0] - lvs[i], dim=1) for i in range(1, len(lvs))]
                title = f"{experiment_name} (run {run_nr + 1}/{len(sub_folders)})"
                plot_histograms(distances, names[1:], title)
                scatter_plot(lvs, names, title)
        else:
            # If model was binary classifier, get Pmiss and Pfa with threshold = 0.5
            all_class_stats.append(get_classification_stats(lvs, names))

    for final_stat, name in zip([all_class_stats, all_tln_stats, all_tln_osvm_stats],
                                ['CLASSIFICATION', 'TLN', 'TLN + OSVM']):
        if len(final_stat) > 0:
            print(f"\n\t{name}")
            final_stat = np.array(final_stat)
            for r_idx, row in enumerate(final_stat[0]):
                o_name, f_name, _, _, _ = row
                pmisses, pfas, aucs = final_stat[:, r_idx, 2], final_stat[:, r_idx, 3], final_stat[:, r_idx, 4]
                pmisses = pmisses.astype(np.float32)
                pfas = pfas.astype(np.float32)
                aucs = aucs.astype(np.float32)
                print(f"\t\t{o_name} - {f_name}: "
                      f"Pmiss {np.mean(pmisses):.2f} (±{np.std(pmisses):.2f})\t"
                      f"Pfa {np.mean(pfas):.2f} (±{np.std(pfas):.2f})\t"
                      f"AUC {np.mean(aucs):.2f} (±{np.std(aucs):.2f})")


def recursive_summary(path, train=False, plot=True):
    """Given a the path containing one, computes the results """
    if is_experiment_folder(path):
        experiment_summary(path, train=train, plot=plot)

    for sub_fold in get_subfolders(path):
        recursive_summary(sub_fold, train, plot)


def main():
    """Recursively prints the results searching for experiments in sub-folders"""
    parser = ArgumentParser()
    parser.add_argument("dirs", nargs='+', help="List of directories")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--plot", action='store_false')
    args = vars(parser.parse_args())

    for d in args['dirs']:
        recursive_summary(d, args['train'], args['plot'])


if __name__ == '__main__':
    main()
