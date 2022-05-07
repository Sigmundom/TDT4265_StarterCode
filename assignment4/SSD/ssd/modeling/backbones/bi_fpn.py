import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from typing import OrderedDict, Tuple, List



class DepthwiseConvBlock(nn.Module):
    """
    Depthwise seperable convolution. 
    
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(DepthwiseConvBlock,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, 
                               padding, dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, dilation=1, groups=1, bias=False)
        
        
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()
        
    def forward(self, inputs):
        x = self.depthwise(inputs)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
    
class ConvBlock(nn.Module):
    """
    Convolution block with Batch Normalization and ReLU activation.
    
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, freeze_bn=False):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9997, eps=4e-5)
        self.act = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.act(x)

class BiFPNBlock(nn.Module):
    """
    Bi-directional Feature Pyramid Network
    """
    def __init__(self, feature_size=64, epsilon=0.0001):
        super(BiFPNBlock, self).__init__()
        self.epsilon = epsilon
        
        self.p3_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p4_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_td = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_td = DepthwiseConvBlock(feature_size, feature_size)
        
        self.p4_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p5_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p6_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p7_out = DepthwiseConvBlock(feature_size, feature_size)
        self.p8_out = DepthwiseConvBlock(feature_size, feature_size)
        
        # TODO: Init weights
        self.w1 = nn.Parameter(torch.Tensor(2, 5))
        self.w1_relu = nn.ReLU()
        self.w2 = nn.Parameter(torch.Tensor(3, 5))
        self.w2_relu = nn.ReLU()
    
    def forward(self, inputs):
        p3_x, p4_x, p5_x, p6_x, p7_x, p8_x = inputs
        
        # Calculate Top-Down Pathway
        w1 = self.w1_relu(self.w1)
        w1 = w1 / (torch.sum(w1, dim=0) + self.epsilon)
        w2 = self.w2_relu(self.w2)
        w2 = w2 / (torch.sum(w2, dim=0) + self.epsilon)
        
        p8_td = p8_x
        p7_td = self.p7_td(w1[0, 0] * p7_x + w1[1, 0] * F.interpolate(p8_td, scale_factor=2))        
        p6_td = self.p6_td(w1[0, 1] * p6_x + w1[1, 1] * F.interpolate(p7_td, scale_factor=2))        
        p5_td = self.p5_td(w1[0, 2] * p5_x + w1[1, 2] * F.interpolate(p6_td, scale_factor=2))
        p4_td = self.p4_td(w1[0, 3] * p4_x + w1[1, 3] * F.interpolate(p5_td, scale_factor=2))
        p3_td = self.p3_td(w1[0, 4] * p3_x + w1[1, 4] * F.interpolate(p4_td, scale_factor=2))
        
        # Calculate Bottom-Up Pathway
        p3_out = p3_td
        p4_out = self.p4_out(w2[0, 0] * p4_x + w2[1, 0] * p4_td + w2[2, 0] * nn.Upsample(scale_factor=0.5)(p3_out))
        p5_out = self.p5_out(w2[0, 1] * p5_x + w2[1, 1] * p5_td + w2[2, 1] * nn.Upsample(scale_factor=0.5)(p4_out))
        p6_out = self.p6_out(w2[0, 2] * p6_x + w2[1, 2] * p6_td + w2[2, 2] * nn.Upsample(scale_factor=0.5)(p5_out))
        p7_out = self.p7_out(w2[0, 3] * p7_x + w2[1, 3] * p7_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p6_out))
        p8_out = self.p8_out(w2[0, 4] * p8_x + w2[1, 4] * p8_td + w2[2, 3] * nn.Upsample(scale_factor=0.5)(p7_out))

        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]
    
class BiFPNLayer(nn.Module):
    def __init__(self, size, feature_size=64, num_layers=2, epsilon=0.0001):
        super(BiFPNLayer, self).__init__()
        self.prepare_input = nn.ModuleList([nn.Conv2d(s, feature_size, kernel_size=1, stride=1, padding=0) for s in size])
        
        # # p6 is obtained via a 3x3 stride-2 conv on C5
        # self.p6 = nn.Conv2d(size[2], feature_size, kernel_size=3, stride=2, padding=1)
        
        # # p7 is computed by applying ReLU followed by a 3x3 stride-2 conv on p6
        # self.p7 = ConvBlock(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        # bifpns = []
        # for _ in range(num_layers):
        #     bifpns.append(BiFPNBlock(feature_size, epsilon))
        # self.bifpn = nn.Sequential(*bifpns)
        self.bifpn = nn.Sequential(*[BiFPNBlock(feature_size, epsilon) for _ in range(num_layers)])
    
    def forward(self, inputs):
        
        # Calculate the input column of BiFPN
        # p3_x = self.p3(c3)        
        # p4_x = self.p4(c4)
        # p5_x = self.p5(c5)
        # p6_x = self.p6(c5)
        # p7_x = self.p7(p6_x)
        
        # features = [p3_x, p4_x, p5_x, p6_x, p7_x]
        features = [self.prepare_input[i](inputs[i]) for i in range(len(inputs))]
        for f in features:
            print(f.shape)
        return self.bifpn(features)



# class BiFPNLayer(torch.nn.Module):
#     def __init__(self, in_channels_list):
#         super().__init__()

#         self.out_5 = nn.Conv2d(in_channels_list[5], in_channels_list[5], kernel_size=3, padding=1)
#         resample = ResampleFeatureMap(in_channels=40, out_channels=112, reduction_ratio=2)
#         self.ff_6 = nn.Sequential(

#         )

        
#     def __call__(self, in_features):
#         out_features = OrderedDict()

#         out_features['f5'] = self.out_5(in_features['f5'])

#         td_6 = torch.concat((out_features['f4'], out_features['f5']




class BiFPN(torch.nn.Module):
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
        elif type == 'resnet101':
            self.in_channels_list = [64, 256, 512, 1024, 2048, 64]
            self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        else:
            raise Exception(f'FPN does not support model type {type}')

        self.out_channels = 6*[output_channels]
       

        self.bi_fpn = BiFPNLayer(self.in_channels_list, feature_size=output_channels, num_layers=3)

        self.extra = nn.Sequential(
                nn.Conv2d(self.in_channels_list[4], 128, kernel_size=2, padding=1, stride=2),
                nn.ReLU(),
                nn.Conv2d(128, self.in_channels_list[5], kernel_size=2, padding=0),
                nn.ReLU(),
            )

    def _resnet_forward(self, x):
        # features = OrderedDict()
        features = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        # features['f0'] = x
        features.append(x)

        x = self.resnet.layer1(x)
        x = self.resnet.maxpool(x)
        # features['f1'] = x
        features.append(x)

        x = self.resnet.layer2(x)
        # features['f2'] = x
        features.append(x)

        x = self.resnet.layer3(x)
        # features['f3'] = x
        features.append(x)


        x = self.resnet.layer4(x)
        # features['f4'] = x
        features.append(x)

        x = self.extra(x)
        # features['f5'] = x
        features.append(x)
        
        return features


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

        features = self._resnet_forward(x)


        out_features = self.bi_fpn(features)
         

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

