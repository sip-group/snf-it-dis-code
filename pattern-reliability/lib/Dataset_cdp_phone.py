import torch
import torchvision.transforms.functional
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage.exposure import match_histograms

import numpy as np

import random
import os
import cv2
from PIL import Image

import matplotlib.pyplot as plt

src_path = "/media/tuttj/phd_data/datasets/2023_Indigo_1x1_mobile"


class Dataset_cdp(Dataset):

    def __init__(self, dataset_name, run=1, nb_samples=1440, load_unet=False, root_dir=src_path, grayscale=True, normalized=False, hist_match=False):

        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.grayscale = grayscale
        self.normalized = normalized
        self.nb_samples = nb_samples
        self.run = run
        self.load_unet = load_unet
        self.hist_match = hist_match

        self.paths = {}
        self.extensions = {}
        self.missing = []

        self.list_IDs = list(range(1, 1+nb_samples))

        self.total_samples = {'iphone': 1440, 'samsung': 1440}

        self.name_dict = {
            't': 'templates',

            'x_iphone': f'iphone/run_{run}/original',
            'f_iphone': f'iphone/run_{run}/fake',
            'x_samsung': f'samsung/run_{run}/original',
            'f_samsung': f'samsung/run_{run}/fake',

        }

        self._check_arguments()
        self._init_dataset()

    def _check_arguments(self):

        assert self.dataset_name in ['scanner', 'iphone', 'samsung'], \
            print(f'{self.dataset_name} is not a valid dataset name. Possibilities are : scanner, iphone, samsung.')

        if self.dataset_name == 'scanner':

            assert self.run in {0,1}, \
                print(f'{self.run} is not a valid run number for dataset scanner. Possibility is 0 or 1.')

            assert 0 < self.nb_samples < 1441, \
                print(f'{self.nb_samples} is not a valid nb of samples for {self.dataset_name}. '
                      f'Choose an integer between 1 and 1440')
        else:

            assert self.run in [1, 2, 3, 4, 5, 6], \
                print(f'{self.run} is not a valid run number for dataset iphone or samsung. Possibilities are 1,2, ..., 6.')

            assert 0 < self.nb_samples < 1441, \
                print(f'{self.nb_samples} is not a valid nb of samples for {self.dataset_name}. '
                      f'Choose an integer between 1 and 1440')

    def _init_dataset(self):

        keys = ['t', 'x', 'f']
        if self.load_unet:
            keys += ['t_x', 't_f']

        for key in keys:
            if key == 't':
                self.paths[key] = os.path.join(self.root_dir, self.name_dict[key])
            else:
                self.paths[key] = os.path.join(self.root_dir, self.name_dict[f'{key}_{self.dataset_name}'])

        self._check_format()
        #  self._find_missing()
        self._remove_missing()

    def _check_format(self):

        for key in self.paths.keys():
            for fname in os.listdir(self.paths[key]):
                if fname.endswith('.tiff'):
                    self.extensions[key] = ('.tiff', len(fname[:-5]))  # extension + zfill format of name
                    break
                elif fname.endswith('.png'):
                    self.extensions[key] = ('.png', len(fname[:-4]))
                    break
                else:
                    print(f'No tiff or png file in {self.paths[key]}')

    def _remove_missing(self):
        with open(os.path.join(self.root_dir, 'removed.csv')) as f:
            lines = f.readlines()

        line = lines[self.run - 1]
        for s in line.split(','):
            i = int(s)

            if i <= self.nb_samples:
                self.missing.append(int(s))
                self.list_IDs.remove(int(s))
                print(f'Removed sample {i}')

        self.length = len(self.list_IDs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        ID = self.list_IDs[index]
        t = transforms.ToTensor()

        images = {}

        for key in self.paths.keys():

            cur_path = os.path.join(self.paths[key], str(ID).zfill(self.extensions[key][1]) + self.extensions[key][0])
            img = cv2.imread(cur_path, -1)

            if img.dtype == 'uint8':
                img = (img / (2**8-1)).astype('float32')
            elif img.dtype == 'uint16':
                img = (img / (2**16-1)).astype('float32')
            elif img.dtype == 'float64':
                img = img.astype('float32')

            if self.grayscale:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                assert len(img.shape) == 3, print('Original image in grayscale, unable to load RGB.')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.normalized:
                img = (img - img.mean()) / img.std()

            if self.hist_match and self.dataset_name == 'scanner' and key != 't':

                ref_path = os.path.join(self.paths['x'], str(12).zfill(self.extensions['x'][1]) + self.extensions['x'][0])
                ref = cv2.imread(ref_path, -1)

                if ref.dtype == 'uint8':
                    ref = (ref / (2**8-1)).astype('float32')
                elif ref.dtype == 'uint16':
                    ref = (ref / (2**16-1)).astype('float32')
                elif ref.dtype == 'float64':
                    ref = ref.astype('float32')

                if self.grayscale:
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    img = match_histograms(img, ref)
                else:
                    assert len(img.shape) == 3, print('Original image in grayscale, unable to load RGB.')
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = match_histograms(img, ref, channel_axis=-1)

            if key in ['t_x', 't_f'] and self.dataset_name == 'scanner':
                img = img[::3,::3]

            images[key] = img

        return images


if __name__ == '__main__':

    dataset = Dataset_cdp(dataset_name='scanner', load_unet=False, run=1, nb_samples=10)
    dataset_matched = Dataset_cdp(dataset_name='scanner', load_unet=False, run=1, nb_samples=10, hist_match=True)

    i = 3

    plt.imshow(dataset[i]['f'][30:90,30:90], cmap='Greys_r')
    plt.colorbar()
    plt.show()
    plt.imshow(dataset_matched[i]['f'][30:90,30:90], cmap='Greys_r')

    print(np.abs(dataset[i]['f'] - dataset_matched[i]['f']).mean())

    """
    for key in dataset[0].keys():
        print(key, dataset[0][key].shape)

        plt.figure(figsize=(20,20))
        plt.subplot(1, 5, i)
        plt.imshow(dataset[0][key][20:40,20:40], cmap='Greys')
        i += 1
    """
    plt.show()
