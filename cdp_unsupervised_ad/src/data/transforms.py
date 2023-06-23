import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, CenterCrop
from torchvision.transforms.functional import adjust_gamma, rotate, vflip, hflip


# Custom transforms
class ComposeAll:
    """Composes all transforms that take as input multiple arguments"""

    def __init__(self, all_transforms):
        self.transforms = all_transforms

    def __call__(self, *args):
        result = args
        for t in self.transforms:
            result = t(*result)
        return result


class ToTensorAll:
    """Converts all the given inputs to pytorch tensors."""

    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, *args):
        result = []
        for arg in args:
            result.append(self.to_tensor(arg))
        return result


class NormalizeAll:
    """Makes sure that  all values in a sensor are within [0, 1] by dividing its values by 2**16 if necessary (1x1)"""

    def __call__(self, *args):
        result = []
        for arg in args:
            # mu, sigma = torch.mean(arg), torch.std(arg)
            # arg = (arg - mu) / sigma
            arg -= arg.min()
            arg /= arg.max()
            result.append(arg)
        return result


class CenterCropAll:
    """Center crops each image with the specified size"""

    def __init__(self, size):
        self.t = CenterCrop(size)

    def __call__(self, *args):
        result = []
        for arg in args:
            result.append(self.t(arg))
        return result


class ResizeAll:
    def __init__(self, size=(228, 228)):
        self.size = size

    def __call__(self, *args):
        result = []
        for arg in args:
            result.append(cv2.resize(arg, self.size))
        return result


class RandomRotationAll:
    """Applies the same random rotation to all of its inputs"""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, *args):
        angle = int(np.random.choice(self.angles))
        result = []
        for arg in args:
            result.append(rotate(arg, angle))
        return result


class RandomGammaCorrectionAll:
    """Applies the same random gamma correction to all of its inputs"""

    def __init__(self, bounds=(0.4, 1.3)):
        self.range = bounds
        self.interval = bounds[1] - bounds[0]

    def __call__(self, *args):
        gamma = float(self.range[0] + torch.rand(1) * self.interval)
        result = []
        for arg in args:
            result.append(adjust_gamma(arg, gamma))
        return result


class RandomFlip:
    def __init__(self, p=0.5, horizontal=True):
        self.p = p
        self.flip_fn = hflip if horizontal else vflip

    def __call__(self, *args):
        if torch.rand(1) >= self.p:
            return list(args)

        result = []
        for arg in args:
            result.append(self.flip_fn(arg))
        return result


class NormalizedTensorTransform:
    """Testing transform that simply converts anything to a Tensor and normalizes it"""

    def __init__(self):
        self.to_tensor = ToTensorAll()
        self.normalize = NormalizeAll()

    def __call__(self, *args):
        # Collecting tensors
        tensors = self.to_tensor(*args)

        # Removing duplicated channels
        for i in range(len(tensors)):
            if tensors[i].shape[0] > 1:
                tensors[i] = torch.unsqueeze(tensors[i][0], 0)

        # Returning the normalized version of the tensors
        return self.normalize(*tensors)


class NormalizedResizeTensorTransform:
    def __init__(self, size=(228, 228)):
        self.resize = ResizeAll(size)
        self.norm_tens = NormalizedTensorTransform()

    def __call__(self, *args):
        # Collecting tensors
        resized = self.resize(*args)
        return self.norm_tens(*resized)


class AllRandomTransforms:
    """Training transform that converts to Tensor, Normalizes, Rotates randomly and applies a gamma correction"""

    def __init__(self, angles=(0, 90, 180, 270), bounds=(0.4, 1.3)):
        self.h_flip = RandomFlip()
        self.v_flip = RandomFlip(horizontal=False)
        self.rotate = RandomRotationAll(angles)
        self.gamma = RandomGammaCorrectionAll(bounds)

    def __call__(self, *args):
        return self.gamma(*self.rotate(*self.v_flip(*self.h_flip(*args))))
