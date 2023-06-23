import os

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

# Definitions
SUPPORTED_MODELS = {'resnet18': resnet18, 'resnet50': resnet50}


class EmbeddingNetwork(nn.Module):
    """Custom embedding model composed of a feature extractor and a 'classifier' (MLP) on top."""

    def __init__(self, input_size, latent_dimension, model_name='resnet18', mlp_head=False):
        # Super constructor
        super(EmbeddingNetwork, self).__init__()

        # Local variables
        self.input_size = input_size
        self.latent_dimension = latent_dimension

        # Checking if model is supported
        assert model_name in SUPPORTED_MODELS

        # Creating custom model
        in_channels = self.input_size[0]
        self.fm, self.classifier = self._name_to_model(model_name, in_channels, self.latent_dimension, mlp_head)

    def forward(self, x):
        features = self.fm(x)
        return self.classifier(features.squeeze()), features

    def _name_to_model(self, model_name, in_channels, output_size, mlp_classifier=False):
        for k in SUPPORTED_MODELS.keys():
            if k == model_name.lower():
                return self._get_custom_resnet(in_channels, output_size, SUPPORTED_MODELS[k], mlp_classifier)
        raise KeyError(f"Model with name {model_name} is not supported. Supported models are: {SUPPORTED_MODELS}.")

    def _get_custom_resnet(self, in_channels, output_size, resnet_fn, mlp_head=False):
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
        if mlp_head:
            classifier = nn.Sequential(
                nn.Linear(resnet.fc.in_features, resnet.fc.in_features),
                nn.ReLU(),
                nn.Linear(resnet.fc.in_features, output_size)
            )
        else:
            classifier = nn.Linear(resnet.fc.in_features, output_size)

        return feature_extractor, classifier


class ModelsManager:
    """Manages creation, storage and loading of Template and Verification networks."""

    def __init__(self, t_in_size, v_in_size, latent_d, model_name, multi_gpu, device, store_dir, mlp_head=False):
        self.t_in_size = t_in_size
        self.v_in_size = v_in_size
        self.latent_d = latent_d
        self.model_name = model_name
        self.multi_gpu = multi_gpu
        self.device = device
        self.store_dir = store_dir
        self.t_store_path = os.path.join(store_dir, 'template_net.pt')
        self.v_store_path = os.path.join(store_dir, 'verification_net.pt')
        self.mlp_head = mlp_head

        # TODO: Consider wrapping models around torch.jit.script()
        self.t_net, self.v_net = self._get_empty_models()
        if self.multi_gpu:
            self.t_net = nn.DataParallel(self.t_net)
            self.v_net = nn.DataParallel(self.v_net)

    def _get_empty_models(self):
        t_net = EmbeddingNetwork(self.t_in_size, self.latent_d, self.model_name, self.mlp_head).to(self.device)
        v_net = EmbeddingNetwork(self.v_in_size, self.latent_d, self.model_name, self.mlp_head).to(self.device)
        return t_net, v_net

    def get_models(self):
        return self.t_net, self.v_net

    def store_networks(self):
        # Storing best model ever
        if self.multi_gpu:
            torch.save(self.t_net.module.state_dict(), self.t_store_path)
            torch.save(self.v_net.module.state_dict(), self.v_store_path)
        else:
            torch.save(self.t_net.state_dict(), self.t_store_path)
            torch.save(self.v_net.state_dict(), self.v_store_path)

    def load_networks(self):
        self.t_net, self.v_net = self._get_empty_models()

        self.t_net.load_state_dict(torch.load(self.t_store_path, map_location=self.device))
        self.v_net.load_state_dict(torch.load(self.v_store_path, map_location=self.device))

        if self.multi_gpu and torch.cuda.is_available():
            self.t_net = nn.DataParallel(self.t_net)
            self.v_net = nn.DataParallel(self.v_net)
