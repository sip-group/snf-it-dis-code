import os
import cv2
import json
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
from torch.nn import BCELoss
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from torchvision.models import resnet18

# Definitions
MODEL_STATE_DICT_NAME = "model_sd.pt"


def get_idxs(file_path):
    file = open(file_path, "r")
    idxs = [int(line.split(".tiff")[0]) for line in file]
    file.close()
    return idxs


class CDPDataset(Dataset):
    def __init__(self, t_dir, x_dir, f_dir, indices, return_stack=False, load=True):
        """
            Copy Detection Pattern (CDP) dataset. Data is loaded in triplets of templates, originals and fakes (t, x, f).

            :param t_dir: Directory containing images of the digital templates
            :param x_dir: Directories containing images of the originals
            :param f_dir: Directories containing images of the counterfeits
            :param indices: List of img numbers to be taken (e.g. [1, 7, 8, 13, ...])
            :param return_stack: Whether to stack templates with originals and fakes for authentication or not.
            :param load: Whether to load all images in memory in one go or not.
        """
        super(CDPDataset, self).__init__()

        # Local variables
        self.t_dir = t_dir
        self.x_dir = x_dir
        self.f_dir = f_dir
        self.indices = indices
        self.return_stack = return_stack
        self.all_loaded = load

        # Keeping only file names that exists in all folders
        self.file_names = []
        all_dirs = [t_dir, x_dir, f_dir]
        for fn in os.listdir(t_dir):
            if np.all([os.path.isfile(os.path.join(d, fn)) for d in all_dirs]) and int(fn.split(".")[0]) in indices:
                self.file_names.append(fn)

        if load:
            self.all_images = []
            for idx in range(self.__len__()):
                images = self._idx_to_images(idx)
                self.all_images.append(images)

    def __getitem__(self, item):
        """Returns the template image, followed by the several originals and fakes images.
        The original and fakes images come, in order, from the x_dirs and f_dirs used to initialize this CDPDataset
         """
        if self.all_loaded:
            images = self.all_images[item]
        else:
            images = self._idx_to_images(item)

        if self.return_stack:
            images = [torch.cat((images[0], img)) for img in images]

        return {
            'template': images[0],
            'original': images[1],
            'fake': images[2],
            'code name': self.file_names[item]
        }

    def __len__(self):
        return len(self.file_names)

    def _idx_to_images(self, idx):
        """
        Given an index, returns the template, all originals and all fakes for the same CDP (as a list).
        Also, applies the deterministic pre_transform, if specified, to all the images.
        """
        file_name = self.file_names[idx]
        images = [self._load_image(os.path.join(d, file_name)) for d in [self.t_dir, self.x_dir, self.f_dir]]

        return images

    def _load_image(self, path):
        # Loading image as Grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        # Min-Max normalization
        img = img - np.min(img)
        img = img / np.max(img)

        # Conversion to tensor
        img = torch.as_tensor(img).unsqueeze(0).float()
        return img


def get_custom_resnet(in_channels, output_size=1, resnet_fn=resnet18):
    """Resnet18 Custom model"""
    # Creating model
    resnet = resnet_fn(pretrained=False)

    # Modifying first conv layer's expected input channels
    resnet.conv1 = nn.Conv2d(
        in_channels,
        resnet.conv1.out_channels,
        resnet.conv1.kernel_size,
        stride=resnet.conv1.stride,
        padding=resnet.conv1.padding,
        bias=resnet.conv1.bias
    )

    # Separating feature extractor and classification head
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

    # Modifying output's dimensionality
    classifier = nn.Sequential(
        nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
        nn.ReLU(),
        nn.Linear(resnet.fc.in_features, output_size),
        nn.Sigmoid()
    )

    return nn.Sequential(feature_extractor, nn.Flatten(), classifier)


def training_loop(model, device, n_epochs, lr, train_loader, val_loader, store_path):
    model = model.to(device).train()
    optim = Adam(model.parameters(), lr=lr)
    bce = BCELoss()

    def get_loss(x, f):
        l = 0.0
        x = x.to(device)
        y = torch.ones(len(x), 1).to(device)
        l += bce(model(x), y)

        f = f.to(device)
        y = torch.zeros(len(f), 1).to(device)
        l += bce(model(f), y)

        return l

    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        train_loss, val_loss = 0, 0

        for batch in train_loader:
            loss = get_loss(batch["original"], batch["fake"])
            train_loss += loss.item() / len(train_loader)

            optim.zero_grad()
            loss.backward()
            optim.step()

        for batch in val_loader:
            loss = get_loss(batch["original"], batch["fake"])
            val_loss += loss.item() / len(val_loader)

        epoch_log = f"Epoch {epoch + 1}/{n_epochs}: Train loss -> {train_loss:.3f}\t Val loss -> {val_loss:.3f}"
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(store_path, MODEL_STATE_DICT_NAME))
            epoch_log += " --> Best model ever stored"

        print(epoch_log)
    print("Training finished.")


@torch.no_grad()
def test_loop(model, device, test_set):
    model = model.to(device).eval()
    pred = {"originals": {}, "fakes": {}}

    for element in test_set:
        name = element["code name"]
        x = element["original"].unsqueeze(0).to(device)
        f = element["fake"].unsqueeze(0).to(device)
        pred["originals"][name] = model(x).item()
        pred["fakes"][name] = model(f).item()
    return pred


def main():
    # Parsing program arguments
    parser = ArgumentParser()
    parser.add_argument("--t_dir", type=str, help="Path to directory containing templates")
    parser.add_argument("--x_dir", type=str, help="Path to directory containing original printed codes")
    parser.add_argument("--f_dir", type=str, help="Path to directory containing fake printed codes")
    parser.add_argument("--extra_dir", type=str, help="Path to directory containing extra codes")
    parser.add_argument("--train_indices", type=str, help="Path to training indices")
    parser.add_argument("--val_indices", type=str, help="Path to validation indices")
    parser.add_argument("--concat_ty", action="store_true", help="Whether the supervised model uses both t and y")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of trianing epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate during training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--result_dir", type=str, help="Path to the directory where model and results are stored")
    args = vars(parser.parse_args())

    # Getting data
    t_dir, x_dir, f_dir = args["t_dir"], args["x_dir"], args["f_dir"]
    concatenating = args["concat_ty"]
    train_idxs, val_idxs = get_idxs(args["train_indices"]), get_idxs(args["val_indices"])

    train_set = CDPDataset(t_dir, x_dir, f_dir, train_idxs, return_stack=concatenating, load=False)
    val_set = CDPDataset(t_dir, x_dir, f_dir, val_idxs, return_stack=concatenating, load=False)

    extra_dir = args["extra_dir"]
    extra_test_set = CDPDataset(t_dir, x_dir, extra_dir, np.arange(721), return_stack=concatenating, load=False)

    batch_size = args["batch_size"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, drop_last=True)

    # Getting device and creating model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_custom_resnet(1 if not concatenating else 2, 1, resnet_fn=resnet18).to(device)

    # Training loop
    n_epochs, result_dir = args["n_epochs"], args["result_dir"]
    os.makedirs(result_dir, exist_ok=True)
    lr = args["lr"]
    training_loop(model, device, n_epochs, lr, train_loader, val_loader, result_dir)

    # Extra test loop with best model
    state_dict = torch.load(os.path.join(result_dir, MODEL_STATE_DICT_NAME), map_location=device)
    model.load_state_dict(state_dict)
    predictions = test_loop(model, device, extra_test_set)

    # Storing predictions
    with open(os.path.join(result_dir, "predictions.json"), "w") as fp:
        json.dump(predictions, fp)

    print("Program completed successfully")


if __name__ == '__main__':
    main()
