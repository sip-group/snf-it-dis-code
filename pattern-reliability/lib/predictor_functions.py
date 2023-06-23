import numpy as np
from skimage.util.shape import view_as_blocks, view_as_windows
from skimage.filters import threshold_otsu
from tqdm import trange, tqdm
import multiprocessing as mp

"""
In this file, we implement the core functions for the Codebook algorithm and codebook-based authentication.
"""


def train_codebook(template, target, estimator=None, block_size=3, verbose=True):

    """
    This function computes a trained codebook C on 3x3 neighbourhoods. (Generalization should come later)

    Args:
        template: A numpy array representing a batch of binary templates
        target: A numpy array representing a batch of printed templates
        estimator: A boolean value telling whether target needs to be binarized before processing
        block_size: An integer representing the magnification factor from template to target
        verbose: A boolean value asking the function to explicit its steps
        A pair (codebook, local_stat) which represents the codebook indexed by neighbourhood.

    Returns:
        local_prob: A dictionary of python list of integers indexed by neighbourhoods representing the codebook.
                    The dictionary contains the two codebooks: P_b(w) and P(w).
        local_stat: A python list of lists indexed by neighbourhoods.
                    Each sublist represents the output t_tilda_ij of a particular realisation of omega_ij.
                    The codebook at index omega is the mean value of this sublist.
    """

    local_stat = { 'error': {}, 'distr': {} }
    local_prob = { 'error': {}, 'distr': {} }
    
    for i in range(2 ** 9):
        code = str(bin(i))[2:].zfill(9)
        local_stat['error'][code] = []
        local_stat['distr'][code] = []
    
    t = template

    if estimator is None:
        t_tilda = target
    elif estimator == 'otsu':
        if verbose:
            print('Applying Otsu')
        
        t_tilda = apply_otsu_threshold(target, block_size, verbose)
    else:
        raise NameError('Invalid estimator')

    t_windows = view_as_windows(t, window_shape=(t.shape[0], 3, 3), step=1)
    t_windows = t_windows.squeeze(axis=0)
    t_windows = np.moveaxis(t_windows, 2, 0)

    if verbose:
        print('Computing the dictionary')

    if verbose:
        for i in trange(t.shape[0]):
            for v in range(t.shape[1]-2):
                for h in range(t.shape[2]-2):
                    current_flipped = (t_windows[i, v, h, 1, 1] != t_tilda[i, v+1, h+1]) * 1
                    current_distr = t_tilda[i, v+1, h+1]
                    code_array = t_windows[i, v, h].flatten()
                    current_code = ''.join(str(e) for e in code_array * 1)

                    local_stat['error'][current_code].append(current_flipped)
                    local_stat['distr'][current_code].append(current_distr)

    else:
        for i in range(t.shape[0]):
            for v in range(t.shape[1]-2):
                for h in range(t.shape[2]-2):
                    current_flipped = (t_windows[i, v, h, 1, 1] != t_tilda[i, v+1, h+1]) * 1
                    current_distr = t_tilda[i, v+1, h+1]
                    code_array = t_windows[i, v, h].flatten()
                    current_code = ''.join(str(e) for e in code_array * 1)

                    local_stat['error'][current_code].append(current_flipped)
                    local_stat['distr'][current_code].append(current_distr)

    for key in local_stat['error'].keys():
        local_prob['error'][key] = np.mean(local_stat['error'][key])
        local_prob['distr'][key] = np.mean(local_stat['distr'][key])
    
    return local_prob, local_stat


def batch_train_codebook(template, target, nb_cores, estimator=None, block_size=1):

    """
    This function applies train_codebook() but uses multiprocessing acceleration.

    Args:
        template: A numpy array representing a batch of binary templates
        target: A numpy array representing a batch of printed templates
        poolsize: An integer representing the size of the batches for pooling on multiprocessing
        estimator: A boolean value telling whether target needs to be binarized before processing
        block_size: An integer representing the magnification factor from template to target

    Returns:
        local_prob: A dictionary of python list of integers indexed by neighbourhoods representing the codebook.
                    The dictionary contains the two codebooks: P_b(w) and P(w).
        local_stat: A python list of lists indexed by neighbourhoods.
                    Each sublist represents the output t_tilda_ij of a particular realisation of omega_ij.
                    The codebook at index omega is the mean value of this sublist.
    """

    assert nb_cores <= mp.cpu_count(), print(f'{nb_cores} is too big. Only {mp.cpu_count()} are available for training')
    # Multiprocessing computation of train_codebook()

    pool = mp.Pool(nb_cores)

    if template.shape[0] < nb_cores:
        batch_template = [template]
        batch_target = [target]
    else:
        batch_template = np.array_split(template, nb_cores)
        batch_target = np.array_split(target, nb_cores)

    L = len(batch_template)

    results = pool.starmap(train_codebook,
                           zip(batch_template,
                               batch_target,
                               [estimator]*L,
                               [block_size]*L,
                               [False]*L))

    pool.close()

    local_stat = {'error': {}, 'distr': {}}
    local_prob = {'error': {}, 'distr': {}}

    # Unfolding the results

    for i in range(512):
            code = bin(i)[2:].zfill(9)
            local_stat['error'][code] = []
            local_stat['distr'][code] = []
        
    for res in results:
        for i in range(512):
            code = bin(i)[2:].zfill(9)
            local_stat['error'][code] += res[1]['error'][code]
            local_stat['distr'][code] += res[1]['distr'][code]
    
    for i in range(512):
        code = bin(i)[2:].zfill(9)
        local_prob['error'][code] = np.mean(local_stat['error'][code])
        local_prob['distr'][code] = np.mean(local_stat['distr'][code])
    
    return local_prob, local_stat


def train_sample_codebook(template, target, estimator=None, block_size=3, verbose=True):

    """
    This function computes a trained codebook C on 3x3 neighbourhoods for each sample in the batch individually.

    Args:
        template: A numpy array representing a batch of binary templates
        target: A numpy array representing a batch of printed templates
        estimator: A boolean value telling whether target needs to be binarized before processing
        block_size: An integer representing the magnification factor from template to target
        verbose: A boolean value asking the function to explicit its steps
        A pair (codebook, local_stat) which represents the codebook indexed by neighbourhood.

    Returns:
        local_prob: A dictionary of python list of integers indexed by neighbourhoods representing the codebook.
                    Each neighbourhood is then split with all training samples.
                    The dictionary contains the two codebooks: P_b(w) and P(w).
        local_stat: A python list of lists indexed by neighbourhoods.
                    Each sublist represents a particular sample.
                    Each subsublist represents the output t_tilda_ij of a particular realisation of omega_ij.
                    The codebook at index omega is the mean value of this sublist.
    """

    local_stat = { 'error': {}, 'distr': {} }
    local_prob = { 'error': {}, 'distr': {} }

    t = template

    for i in range(2 ** 9):
        code = str(bin(i))[2:].zfill(9)
        local_stat['error'][code] = [[] for j in range(t.shape[0])]
        local_stat['distr'][code] = [[] for j in range(t.shape[0])]

        local_prob['error'][code] = []
        local_prob['distr'][code] = []

    if estimator is None:
        t_tilda = target
    elif estimator == 'otsu':
        if verbose:
            print('Applying Otsu')

        t_tilda = apply_otsu_threshold(target, block_size, verbose)
    else:
        raise NameError('Invalid estimator')

    t_windows = view_as_windows(t, window_shape=(t.shape[0], 3, 3), step=1)
    t_windows = t_windows.squeeze(axis=0)
    t_windows = np.moveaxis(t_windows, 2, 0)

    if verbose:
        print('Computing the dictionary')

    if verbose:
        for i in trange(t.shape[0]):
            for v in range(t.shape[1]-2):
                for h in range(t.shape[2]-2):
                    current_flipped = (t_windows[i, v, h, 1, 1] != t_tilda[i, v+1, h+1]) * 1
                    current_distr = t_tilda[i, v+1, h+1]
                    code_array = t_windows[i, v, h].flatten()
                    current_code = ''.join(str(e) for e in code_array * 1)

                    local_stat['error'][current_code][i].append(current_flipped)
                    local_stat['distr'][current_code][i].append(current_distr)

    else:
        for i in range(t.shape[0]):
            for v in range(t.shape[1]-2):
                for h in range(t.shape[2]-2):
                    current_flipped = (t_windows[i, v, h, 1, 1] != t_tilda[i, v+1, h+1]) * 1
                    current_distr = t_tilda[i, v+1, h+1]
                    code_array = t_windows[i, v, h].flatten()
                    current_code = ''.join(str(e) for e in code_array * 1)

                    local_stat['error'][current_code][i].append(current_flipped)
                    local_stat['distr'][current_code][i].append(current_distr)

    for key in local_stat['error'].keys():
        for i in range(t.shape[0]):
            local_prob['error'][key] += [np.mean(local_stat['error'][key][i])]
            local_prob['distr'][key] += [np.mean(local_stat['distr'][key][i])]

    return local_prob, local_stat


def batch_train_sample_codebook(template, target, poolsize, estimator=None, block_size=1):

    """
    This function applies train_sample_codebook() but uses multiprocessing acceleration.

    Args:
        template: A numpy array representing a batch of binary templates
        target: A numpy array representing a batch of printed templates
        poolsize: An integer representing the size of the batches for pooling on multiprocessing
        estimator: A boolean value telling whether target needs to be binarized before processing
        block_size: An integer representing the magnification factor from template to target

    Returns:
        local_prob: A dictionary of python list of integers indexed by neighbourhoods representing the codebook.
                    The dictionary contains the two codebooks: P_b(w) and P(w).
        local_stat: A python list of lists indexed by neighbourhoods.
                    Each sublist represents the output t_tilda_ij of a particular realisation of omega_ij.
                    The codebook at index omega is the mean value of this sublist.
    """

    assert (template.shape[0] % poolsize) == 0, print('Poolsize should divide the number of samples.')

    # Multiprocessing computation of train_codebook()

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(train_sample_codebook,
                           [(template[poolsize*i:poolsize*i+poolsize], target[poolsize*i:poolsize*i+poolsize],
                             estimator,
                             block_size,
                             False) for i in range(template.shape[0] // poolsize)])
    pool.close()

    local_stat = {'error': {}, 'distr': {}}
    local_prob = {'error': {}, 'distr': {}}

    # Unfolding the results

    for i in range(2 ** 9):
        code = str(bin(i))[2:].zfill(9)
        local_stat['error'][code] = [[] for j in range(template.shape[0])]
        local_stat['distr'][code] = [[] for j in range(template.shape[0])]

        local_prob['error'][code] = []
        local_prob['distr'][code] = []

    for res in results:
        for i in range(512):
            code = bin(i)[2:].zfill(9)
            for j in range(template.shape[0]):
                local_stat['error'][code][j] += res[1]['error'][code][j]
                local_stat['distr'][code][j] += res[1]['distr'][code][j]

    for i in range(512):
        code = bin(i)[2:].zfill(9)
        for j in range(template.shape[0]):
            local_prob['error'][code] += [np.mean(local_stat['error'][code][j])]
            local_prob['distr'][code] += [np.mean(local_stat['distr'][code][j])]

    return local_prob, local_stat


def predict(template, local_prob, verbose=True):

    """
    This function applies the predictor algorithm to a batch of template, based on a trained codebook.

    Args:
        template: A numpy array representing a batch of binary templates
        local_prob: A dictionary of python list of integers indexed by neighbourhoods representing the codebook.
                    The dictionary contains the two codebooks: P_b(w) and P(w).
        verbose: A boolean value asking the function to explicit its steps.

    Returns:
        t_predicted: A numpy array representing the probability P(w_ij) for each pixel t_ij in template.
        eps_predicted: A numpy array representing the probability P_b(w_ij) for each pixel t_ij in template.
    """

    assert len(template.shape) == 3, print('Expected template to have 3 dimensions [Batch, Width, Height]')

    eps_predicted = np.zeros_like(template) * 1.

    template_pad = np.pad(template, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=1)
    template_windows = view_as_windows(template_pad, window_shape=(template.shape[0], 3, 3), step=1).squeeze()
    template_windows = np.moveaxis(template_windows, 2, 0)

    if verbose:
        for i in trange(template_windows.shape[0]):
            for v in range(template_windows.shape[1]):
                for h in range(template_windows.shape[2]):
                    code = template_windows[i, v, h].flatten()
                    code = ''.join(str(e) for e in code * 1)
                    code = int(code, 2)
                    eps_predicted[i, v, h] = local_prob[code]
    else:
        for i in range(template_windows.shape[0]):
            for v in range(template_windows.shape[1]):
                for h in range(template_windows.shape[2]):
                    code = template_windows[i, v, h].flatten()
                    code = ''.join(str(e) for e in code * 1)
                    code = int(code, 2)
                    eps_predicted[i, v, h] = local_prob[code]

    return eps_predicted

    
def pool_predict(template, local_prob, nb_cores):

    """
    This function applies predict() but uses multiprocessing acceleration.

    Args:
        template: A numpy array representing a batch of binary templates
        local_prob: A dictionary of python list of integers indexed by neighbourhoods representing the codebook.
                    The dictionary contains the two codebooks: P_b(w) and P(w).
        poolsize: An integer representing the size of the batches for pooling on multiprocessing

    Returns:
        t_predicted: A numpy array representing the probability P(w_ij) for each pixel t_ij in template.
        eps_predicted: A numpy array representing the probability P_b(w_ij) for each pixel t_ij in template.
    """

    assert nb_cores <= mp.cpu_count(), print(f'{nb_cores} is too big. Only {mp.cpu_count()} are available for training')

    # Multiprocessing computation of predict()
    pool = mp.Pool(nb_cores)

    if template.shape[0] < nb_cores:
        batch_template = [template]
    else:
        batch_template = np.array_split(template, nb_cores)

    L = len(batch_template)

    results = pool.starmap(predict,
                           zip(batch_template,
                               [local_prob]*L,
                               [False]*L))

    pool.close()

    # Unfolding the results

    eps_pred = np.concatenate(results)

    return eps_pred


# Utility functions

def apply_otsu_threshold(y, block_size=3, verbose=True):

    """
    This function applies the binarization algorithm of Otsu + Majority voting to a batch of probes y.

    Args:
        y: A numpy array representing a batch of printed templates (either original or fake)
        block_size: An integer representing the magnification factor from template to target
        verbose: A boolean value asking the function to explicit its steps

    Returns:
        t_tilda: A numpy array of boolean values representing y after binarization by Otsu and majority voting.
    """

    assert block_size in {1,3}, 'Otsu estimator only implemented for block size 1 and 3.'

    y_otsu, _ = batch_otsu(y, verbose)

    if block_size == 1:
        t_tilda = (y_otsu > .5)
        return t_tilda

    elif block_size == 3:
        y_blocks = view_as_blocks(y_otsu, (y.shape[0], 3, 3)).squeeze(axis=0)
        t_tilda = np.moveaxis((y_blocks.sum(-1).sum(-1) > 4.5), -1, 0)
        t_tilda = (t_tilda > .5)
        return t_tilda
    else:
        print(f'problem with block_size {block_size}')


def batch_otsu(img_batch, verbose=True):
    """

    This function applies threshold_otsu() to a batch of images.

    Args:
        img_batch: A numpy array representing a batch of printed templates.
        verbose: A boolean value asking the function to explicit its steps

    Returns:
        img_batch_thresh: A numpy array representing a batch of binary images estimated by otsu
        thresh: A python list representing the optimal thresh chosen by otsu's algorithm for each image in the batch.

    """

    nb_samples = img_batch.shape[0]

    thresh = []

    if verbose:
        for i in trange(nb_samples):
            thresh += [threshold_otsu(img_batch[i])]
    else:
        for i in range(nb_samples):
            thresh += [threshold_otsu(img_batch[i])]

    thresh = np.asarray(thresh).reshape([-1,1,1])
    img_batch_thresh = (img_batch > thresh)

    return img_batch_thresh, thresh


if __name__ == '__main__':

    import numpy as np

    template = np.random.randint(0, 2, [100,100,100])
    target = np.random.randint(0, 2, [100,100,100])

    local_prob, local_stat = batch_train_codebook(template, target, poolsize=10, estimator=None, block_size=1)
