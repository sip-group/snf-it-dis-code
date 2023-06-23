from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import cv2

src_path = "/media/tuttj/phd_data/datasets/2023_Indigo_1x1_mobile"
originals = "orig_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG"
fakes = "fake_phone/HPI55_printdpi812.8_printrun1_session1_InvercoteG_EHPI55"


class DatasetCDP(Dataset):

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
            't': 'orig_templates/rcod',
            'iphone': f'iPhone12Pro_run{run}_ss100_focal12_apperture1',
            'samsung': f'SamsungGN20U_run{run}_ss100_focal12_apperture1',
        }

        self._check_arguments()
        self._init_dataset()

    def _check_arguments(self):

        assert self.dataset_name in ['iphone', 'samsung'], \
            print(f'{self.dataset_name} is not a valid dataset name. Possibilities are : iphone, samsung.')

        assert self.run in [1, 2, 3, 4, 5, 6], \
            print(f'{self.run} is not a valid run number for dataset iphone or samsung. Possibilities are 1,2, ..., 6.')

        assert 0 < self.nb_samples < 1441, \
            print(f'{self.nb_samples} is not a valid nb of samples for {self.dataset_name}. '
                  f'Choose an integer between 1 and 1440')

    def _init_dataset(self):

        keys = ['t', 'x', 'f']

        if self.hist_match:
            folder_name = 'rcod_hist'
        else:
            folder_name = 'rcod'

        self.paths['t'] = os.path.join(self.root_dir, self.name_dict['t'])
        self.paths['x'] = os.path.join(self.root_dir, originals, self.name_dict[self.dataset_name], folder_name)
        self.paths['f'] = os.path.join(self.root_dir, fakes, self.name_dict[self.dataset_name], folder_name)

        self._check_format()
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
                    exit(0)

    def _remove_missing(self):
        with open(os.path.join(self.root_dir, f'removed.csv')) as f:
            line = f.readlines()
            line = line[0]

        for s in line.split(','):
            if s != '\n':
                i = int(s)

                if i <= self.nb_samples:
                    self.missing.append(int(s))
                    self.list_IDs.remove(int(s))

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

            images[key] = img

        return images


if __name__ == '__main__':

    dataset = DatasetCDP(dataset_name='iphone', load_unet=False, run=1, nb_samples=1440, hist_match=True)

    i = 3

    print(np.abs(dataset[i]['f']).mean())

    import matplotlib.pyplot as plt

    plt.imshow(dataset[i]['f'][30:90,30:90], cmap='Greys_r')
    plt.colorbar()
    plt.show()