from torch import nn
import torch
import torchvision
from typing import OrderedDict, Tuple, List



class FPN(torch.nn.Module):
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
            output_channels: int,
            image_channels: int,
            output_feature_sizes: List[Tuple[int]], 
            type, 
            pretrained=True
        ):
        super().__init__()
        self.image_channels = image_channels
        self.output_feature_shape = output_feature_sizes
        # self.resnet.

        print(type)
        if type == 'resnet34':
            self.in_channels_list = [64, 64, 128, 256, 512, 64]
            self.resnet = torchvision.models.resnet34(pretrained=pretrained)
        elif type == 'resnet50':
            self.in_channels_list = [64, 256, 512, 1024, 2048, 64]
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        else:
            raise Exception(f'FPN does not support model type {type}')

        self.out_channels = 6*[output_channels]
        self.fpn = torchvision.ops.FeaturePyramidNetwork(self.in_channels_list, output_channels)

        self.extra = nn.Sequential(
                nn.Conv2d(self.in_channels_list[4], 128, kernel_size=2, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, self.in_channels_list[5], kernel_size=2, padding=0),
                nn.ReLU(),
            )




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
        out_features = OrderedDict()

 
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        out_features['feat0'] = x

        x = self.resnet.layer1(x)
        x = self.resnet.maxpool(x)
        out_features['feat1'] = x

        x = self.resnet.layer2(x)
        out_features['feat2'] = x

        x = self.resnet.layer3(x)
        out_features['feat3'] = x


        x = self.resnet.layer4(x)
        out_features['feat4'] = x

        x = self.extra(x)
        out_features['feat5'] = x

        out_features = self.fpn(out_features)
         

        for idx, feature in enumerate(out_features.values()):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features.values())

