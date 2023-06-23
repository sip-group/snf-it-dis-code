import torch
import numpy as np
from skimage.util import img_as_ubyte
from torchvision.transforms.functional import adjust_gamma, rotate, vflip, hflip


# Custom transforms
class ToTensorAll:
    """Converts all of the given inputs to pytorch tensors."""

    def __init__(self):
        pass

    def __call__(self, *args):
        result = []
        for arg in args:
            result.append(torch.as_tensor(arg))
        return result


class NormalizeAll:
    """Makes sure that  all values in a sensor are within [0, 1] by dividing its values by 2**16 if necessary (1x1)"""

    def __call__(self, *args):        
        result = []
        for img in args:
            if not np.max(img) > 255:
                img = img.astype(np.uint8)
            
            if img.ndim == 3:
                img = img[:,:,0]
            
            img = img_as_ubyte(img).astype(np.float32)
            result.append(img)
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
        self.normalize = NormalizeAll()
        self.to_tensor = ToTensorAll()

    def __call__(self, *args):
        # Normalizing
        normalized = self.normalize(*args)

        # Removing duplicated channels
        for i in range(len(normalized)):
            if len(normalized[i].shape) > 2:
                if normalized[i].shape[2] > 1:
                    normalized[i] = normalized[i][:, :, 0]
                    
            normalized[i] = np.expand_dims(normalized[i], axis=0)

        # Making normalized images pytorch tensors
        return self.to_tensor(*normalized)


class AllRandomTransforms:
    """Training transform that converts to Tensor, Normalizes, Rotates randomly and applies a gamma correction"""

    def __init__(self, angles=(0, 90, 180, 270), bounds=(0.4, 1.3)):
        self.h_flip = RandomFlip()
        self.v_flip = RandomFlip(horizontal=False)
        self.rotate = RandomRotationAll(angles)
        self.gamma = RandomGammaCorrectionAll(bounds)

    def __call__(self, *args):
        return self.gamma(*self.rotate(*self.v_flip(*self.h_flip(*args))))
