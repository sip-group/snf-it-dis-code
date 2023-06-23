import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block with instance normalization
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return x + self.residual_block(x)


def _get_bottleneck_model(mode, conv_dim, repeat_num):
    out_channels = 1 if mode in ["t2x", "x2t", "both"] else 2

    layers = [nn.Conv2d(1, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
              nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True), nn.ReLU(inplace=True)]

    # Down-sampling
    curr_dim = conv_dim
    for _ in range(2):
        layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

    # Bottleneck
    for _ in range(repeat_num):
        layers.append(ResidualBlock(in_channels=curr_dim, out_channels=curr_dim))

    # Up-sampling
    for _ in range(2):
        layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2

    layers.append(nn.Conv2d(curr_dim, out_channels, kernel_size=7, stride=1, padding=3, bias=False))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


def get_models(mode, hidden_channels=10, n_res_bottleneck_blocks=1, device=None):
    """Creates a list of simple models based on the mode."""
    if mode == "both":
        return [
            _get_bottleneck_model("t2x", hidden_channels, n_res_bottleneck_blocks).to(device),
            _get_bottleneck_model("x2t", hidden_channels, n_res_bottleneck_blocks).to(device)
        ]
    elif mode == "both_a":
        return [
            _get_bottleneck_model("t2xa", hidden_channels, n_res_bottleneck_blocks).to(device),
            _get_bottleneck_model("x2ta", hidden_channels, n_res_bottleneck_blocks).to(device)
        ]

    return [_get_bottleneck_model(mode, hidden_channels, n_res_bottleneck_blocks).to(device)]
