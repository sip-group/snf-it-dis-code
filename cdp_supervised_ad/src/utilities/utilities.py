import torch
import numpy as np
import random


def set_reproducibility(seed):
    """Given the seed, makes the code reproducible given the same seed"""
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # torch.set_deterministic(True)               # Pytorch 1.7
    # torch.use_deterministic_algorithms(True)    # Pytorch 1.8 or higher


def get_device():
    """Gets current execution device"""
    # Finding current device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Found {torch.cuda.device_count()} devices.")
        print(f"Current device: {torch.cuda.get_device_name(device)}.")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA is not available. Code will run on CPU.")
    return device


def tln_forward(batch, template_net, verification_net, criterion, device, on_features=False,
                no_template=False):
    """
    Runs a forward pass with the given batch, using the given template and verification networks.
    Returns the loss value and the latent vectors for template, originals and fakes.

    :param batch: Batch of data
    :param template_net: Template Network
    :param verification_net: Verification Network
    :param criterion: Loss function that takes the latent vectors of template, originals and fakes as input
    :param device: Device onto which data will be moved
    :param on_features: Whether to use the extracted features as latent vectors (True) or the final head (False)
    :param no_template: Whether to not run images through the template network (for classification)
    """
    t_lv, x_lvs, f_lvs = None, [], []

    def get_lv(net, img):
        img = img.to(device)

        # Make batch out of single image
        if len(img.shape) == 3:
            img = torch.unsqueeze(img, 0)

        out = net(img)[1 if on_features else 0]

        if on_features:
            return out.squeeze()
        return out

    if not no_template:
        t_lv = get_lv(template_net, batch['template'])

    for x in batch['originals']:
        x_lvs.append(get_lv(verification_net, x))

    for f in batch['fakes']:
        f_lvs.append(get_lv(verification_net, f))

    return criterion(t_lv, x_lvs, f_lvs), (t_lv, x_lvs, f_lvs)
