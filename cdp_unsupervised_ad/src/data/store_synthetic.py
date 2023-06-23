import os
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torchvision.utils import save_image


# Paths
PROJECT_DIR = "/Users/bp/Desktop/Projects/cdp_unsupervised_ad"
TEMPLATE_DIR = os.path.join(PROJECT_DIR, "datasets/scanner/templates")
MODELS_DIR = os.path.join(PROJECT_DIR, "results/scanner")


def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading data
    images = {}
    to_tensor = ToTensor()
    for file_name in os.listdir(TEMPLATE_DIR):
        file_path = os.path.join(TEMPLATE_DIR, file_name)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img = to_tensor(np.expand_dims(img, 2))
        img -= img.min()
        img /= img.max()
        images[file_name] = img

    # Obtaining synthetic codes for each model
    for orig in ["55", "76"]:
        for seed in range(10):
            # Store directory
            store_dir = f"./synthetic_codes_scanner/{orig}/seed_{seed}"
            os.makedirs(store_dir, exist_ok=True)

            # Loading model
            model_path = os.path.join(MODELS_DIR, f"{orig}/seed_{seed}/t2x.pt")
            model = torch.load(model_path, map_location=device)

            # Running templates through model
            for name in images.keys():
                img = images[name].unsqueeze(0)
                out = model(img)[0]

                save_image(out, os.path.join(store_dir, name))


if __name__ == '__main__':
    main()
