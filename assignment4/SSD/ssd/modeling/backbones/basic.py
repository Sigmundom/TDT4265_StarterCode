from torch import nn
import torch
from typing import Tuple, List



class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.image_channels = image_channels
        self.output_feature_shape = output_feature_sizes
        self.feature_extractors = [
            #1
            nn.Sequential(
                nn.Conv2d(self.image_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, self.out_channels[0], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            # 2
            nn.Sequential(
                nn.Conv2d(self.out_channels[0], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                # nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, self.out_channels[1], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            # 3
            nn.Sequential(
                nn.Conv2d(self.out_channels[1], 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, self.out_channels[2], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            # 4
            nn.Sequential(
                nn.Conv2d(self.out_channels[2], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.out_channels[3], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            # 5
            nn.Sequential(
                nn.Conv2d(self.out_channels[3], 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.out_channels[4], kernel_size=3, padding=1, stride=2),
                nn.ReLU(),
            ),
            # 6
            nn.Sequential(
                nn.Conv2d(self.out_channels[4], 128, kernel_size=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, self.out_channels[5], kernel_size=2, padding=0, stride=2),
                nn.ReLU(),
            )
        ]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for feature_extractor in self.feature_extractors:
            feature_extractor.to(device)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        in_features = x
        out_features = []

        for feature_extractor in self.feature_extractors:
            # print(in_features.shape)
            out_features.append(feature_extractor(in_features))
            in_features = out_features[-1]

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

