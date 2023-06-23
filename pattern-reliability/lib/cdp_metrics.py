import numpy as np
from skimage.util import view_as_blocks
from skimage.filters import threshold_otsu
from tqdm import trange
import multiprocessing as mp

# Metrics


def batch_metric(t, y, metric, k, w=None, mode='no_processing', nb_cores=1):
    """
    A function that computes a weighted metric between the template t and the probe y
    on batch of images.
    
    t        a binary template
    y        a printed template of same size, or a binary estimate t_tilda of t
    metric   a string describing the metric either
             'mse', 'l1', 'lls', 'dhamm', 'pcor', 'nc_pcor', 'ssim'
    k        the magnification factor from t to y
    w        the weight mask
    mode     A string indicating the type of processing that should be applied to the dataset
                no_processing:  (default) no processing is applied.
                normalize:      a masked (x - mean) / std transformation is applied to y.
                stretch:        a masked histogram stretching is applied to y.
    """
    
    metrics_dict = {
        'mse': batch_mse,
        'l1': batch_l1_normalized,
        'dhamm': batch_dhamm,
        'pcor': batch_pcor,
        'nc_pcor': batch_non_centered_pcor,
        'lls2': batch_log_likelihood_score_pb,
        'lls': batch_log_likelihood_score,
    }
    
    assert (metric in metrics_dict.keys()),\
        f"The metric {metric} is not a correct choice. Possibilities are {metrics_dict.keys()}"

    assert (mode in {'no_processing', 'normalize', 'stretch'}), \
        f'The mode {mode} is not a correct choice. Possibilities are "no_processing", "normalize" and "stretch".'
    
    if k != 1:
        t = t.repeat(k, axis=2).repeat(k, axis=1)
        
    if w is None:
        w = np.ones_like(t)
    else:
        if k != 1:
            w = w.repeat(k, axis=2).repeat(k, axis=1)
    
    u = batch_flatten_image(t)
    v = batch_flatten_image(y)
    w = batch_flatten_image(w)
    
    masked_u = np.ma.array(u, mask = np.logical_not(w))
    masked_v = np.ma.array(v, mask = np.logical_not(w))

    if mode != 'no_processing' and metric not in ['lls', 'lls2', 'dhamm']:
        masked_u = batch_processing(masked_u, mode)
        masked_v = batch_processing(masked_v, mode)

    pool = mp.Pool(nb_cores)

    if masked_u.shape[0] < nb_cores:
        batch_masked_u = [masked_u]
        batch_masked_v = [masked_v]
    else:
        batch_masked_u = np.array_split(masked_u, nb_cores)
        batch_masked_v = np.array_split(masked_v, nb_cores)

    results = pool.starmap(metrics_dict[metric], zip(batch_masked_u, batch_masked_v))
    pool.close()

    results = np.concatenate(results)

    return np.array(results)


def batch_log_likelihood_score(u,v):

    err = np.finfo('float64').eps
    return - (np.log(1 - np.abs(u-v) + err)).mean(-1)


def batch_log_likelihood_score_pb(u,v):

    true_flip = np.abs(u-v)
    err = np.finfo('float64').eps
    return - (v*np.log(err + u) + (1-v)*np.log(err + 1 - u) ).mean(-1)


def batch_mse(u,v):
    
    return np.mean((u-v)**2, axis=-1)
    

def batch_l1_normalized(u,v):
    
    return np.mean(np.abs(u-v), axis=-1)


def batch_dhamm(u,v):
    return (np.logical_xor(u, v)).mean(-1)

    
def batch_pcor(u,v):
    
    cov = ((u - u.mean(-1, keepdims=True))*(v - v.mean(-1, keepdims=True))).mean(-1)
    pCor = cov / (u.std(-1) * v.std(-1))
    
    return pCor


def batch_non_centered_pcor(u,v):
    
    nc_cov = (u*v).mean(-1)
    nc_pCor = nc_cov / (u.std(-1) * v.std(-1))
    
    return nc_pCor

# Utilities


def batch_flatten_image(img_batch):
    return np.reshape(img_batch, (img_batch.shape[0], -1))


def batch_processing(vector_batch, mode):
    """
    This function computes a processing on a batch of masked vectors based on a certain mode.

    Args:
        img_batch:  a masked numpy array of shape [nb_samples, dimension] representing a batch of vectors
        mode:       A string indicating the type of processing that should be applied to the dataset
                        normalize:  (img_batch - mean) / std transformation is applied batch-wise to img_batch.
                        stretch:    a histogram stretching is applied batch-wise to img_batch

    Returns:
        processed:  a masked numpy array of shape [nb_samples, dimension] representing the processed batch of vectors
    """

    if mode == 'normalize':
        mean = vector_batch.mean(-1, keepdims=True)
        std = vector_batch.std(-1, keepdims=True)

        processed = (vector_batch - mean) / std

    elif mode == 'stretch':
        m = vector_batch.min(-1, keepdims=True)
        M = vector_batch.max(-1, keepdims=True)

        processed = (vector_batch - m) / (M - m)

    else:
        print(f'{mode} is not a valid processing. Possibilities are normalize and stretch.')
        exit(0)

    return processed


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    t = (np.random.random([10,50,50]) > .5)*1
    x = np.random.random([10,3*50,3*50])

    v = np.ma.array([1,2,3,4.6,2.3], mask=[1,0,1,0,0])

    v_stretched = batch_processing(v, mode='normalize')

    d1 = batch_metric(t, x, 'l1', 3)
    print(d1)
