# https://amaarora.github.io/2021/01/13/efficientdet-pytorch.html
class ResampleFeatureMap(nn.Sequential):
    def __init__(
            self, in_channels, out_channels, reduction_ratio=1., pad_type='', downsample=None, upsample=None,
            norm_layer=nn.BatchNorm2d, apply_bn=False, conv_after_downsample=False, redundant_bias=False):
        super(ResampleFeatureMap, self).__init__()
        downsample = downsample or 'max'
        upsample = upsample or 'nearest'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio
        self.conv_after_downsample = conv_after_downsample

        conv = None
        if in_channels != out_channels:
            conv = nn.ConvBnAct2d(
                in_channels, out_channels, kernel_size=1, padding=pad_type,
                norm_layer=norm_layer if apply_bn else None,
                bias=not apply_bn or redundant_bias, act_layer=None)

        if reduction_ratio > 1:
            if conv is not None and not self.conv_after_downsample:
                self.add_module('conv', conv)
            if downsample in ('max', 'avg'):
                stride_size = int(reduction_ratio)
                downsample = nn.create_pool2d(
                     downsample, kernel_size=stride_size + 1, stride=stride_size, padding=pad_type)
            else:
                downsample = nn.Interpolate2d(scale_factor=1./reduction_ratio, mode=downsample)
            self.add_module('downsample', downsample)
            if conv is not None and self.conv_after_downsample:
                self.add_module('conv', conv)
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                scale = int(1 // reduction_ratio)
                self.add_module('upsample', nn.Interpolate2d(scale_factor=scale, mode=upsample))

class ConvBnAct2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding='', bias=False,
                 norm_layer=nn.BatchNorm2d, act_layer=None):
                #  norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER):
        super(ConvBnAct2d, self).__init__()
        self.conv = nn.create_conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.bn = None if norm_layer is None else norm_layer(out_channels)
        self.act = None if act_layer is None else act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class Fnode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(Fnode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return self.after_combine(self.combine(x))


class FpnCombine(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, inputs_offsets, target_reduction, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, apply_resample_bn=False,
                 conv_after_downsample=False, redundant_bias=False, weight_method='attn'):
        super(FpnCombine, self).__init__()
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        for idx, offset in enumerate(inputs_offsets):
            in_channels = fpn_channels
            if offset < len(feature_info):
                in_channels = feature_info[offset]['num_chs']
                input_reduction = feature_info[offset]['reduction']
            else:
                node_idx = offset - len(feature_info)
                input_reduction = fpn_config.nodes[node_idx]['reduction']
            reduction_ratio = target_reduction / input_reduction
            self.resample[str(offset)] = ResampleFeatureMap(
                in_channels, fpn_channels, reduction_ratio=reduction_ratio, pad_type=pad_type,
                downsample=downsample, upsample=upsample, norm_layer=norm_layer, apply_bn=apply_resample_bn,
                conv_after_downsample=conv_after_downsample, redundant_bias=redundant_bias)

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x: List[torch.Tensor]):
        dtype = x[0].dtype
        nodes = []
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        out = torch.sum(out, dim=-1)
        return out

class SeparableConv2d(torch.nn.Module):
    def __init__(self, 
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 padding_mode='zeros',
                 depth_multiplier=1,
        ):
        super().__init__()
        
        intermediate_channels = in_channels * depth_multiplier
        self.spatialConv = torch.nn.Conv2d(
             in_channels=in_channels,
             out_channels=intermediate_channels,
             kernel_size=kernel_size,
             stride=stride,
             padding=padding,
             dilation=dilation,
             groups=in_channels,
             bias=bias,
             padding_mode=padding_mode
        )
        self.pointConv = torch.nn.Conv2d(
             in_channels=intermediate_channels,
             out_channels=out_channels,
             kernel_size=1,
             stride=1,
             padding=0,
             dilation=1,
             bias=bias,
             padding_mode=padding_mode,
        )
    
    def forward(self, x):
        return self.pointConv(self.spatialConv(x))


class BiFpnLayer(nn.Module):
    def __init__(self, feature_info, fpn_config, fpn_channels, num_levels=5, pad_type='',
                 downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=None,
                #  downsample=None, upsample=None, norm_layer=nn.BatchNorm2d, act_layer=_ACT_LAYER,
                 apply_resample_bn=False, conv_after_downsample=True, conv_bn_relu_pattern=False,
                 separable_conv=True, redundant_bias=False):
        super(BiFpnLayer, self).__init__()
        self.num_levels = num_levels
        self.conv_bn_relu_pattern = False

        self.feature_info = []
        self.fnode = nn.ModuleList()
        for i, fnode_cfg in enumerate(fpn_config.nodes):
            logging.debug('fnode {} : {}'.format(i, fnode_cfg))
            reduction = fnode_cfg['reduction']
            combine = FpnCombine(
                feature_info, fpn_config, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                target_reduction=reduction, pad_type=pad_type, downsample=downsample, upsample=upsample,
                norm_layer=norm_layer, apply_resample_bn=apply_resample_bn, conv_after_downsample=conv_after_downsample,
                redundant_bias=redundant_bias, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()
            conv_kwargs = dict(
                in_channels=fpn_channels, out_channels=fpn_channels, kernel_size=3, padding=pad_type,
                bias=False, norm_layer=norm_layer, act_layer=act_layer)
            if not conv_bn_relu_pattern:
                conv_kwargs['bias'] = redundant_bias
                conv_kwargs['act_layer'] = None
                after_combine.add_module('act', act_layer(inplace=True))
            after_combine.add_module(
                'conv', nn.SeparableConv2d(**conv_kwargs) if separable_conv else ConvBnAct2d(**conv_kwargs))

            self.fnode.append(Fnode(combine=combine, after_combine=after_combine))
            self.feature_info.append(dict(num_chs=fpn_channels, reduction=reduction))

        self.feature_info = self.feature_info[-num_levels::]

    def forward(self, x: List[torch.Tensor]):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class BiFpn(nn.Module):

    def __init__(self, config, feature_info):
        super(BiFpn, self).__init__()
        self.num_levels = config.num_levels
        norm_layer = config.norm_layer or nn.BatchNorm2d
        if config.norm_kwargs:
            norm_layer = partial(norm_layer, **config.norm_kwargs)
        act_layer = get_act_layer(config.act_type) or _ACT_LAYER
        fpn_config = config.fpn_config or get_fpn_config(
            config.fpn_name, min_level=config.min_level, max_level=config.max_level)

        self.resample = nn.ModuleDict()
        for level in range(config.num_levels):
            if level < len(feature_info):
                in_chs = feature_info[level]['num_chs']
                reduction = feature_info[level]['reduction']
            else:
                # Adds a coarser level by downsampling the last feature map
                reduction_ratio = 2
                self.resample[str(level)] = ResampleFeatureMap(
                    in_channels=in_chs,
                    out_channels=config.fpn_channels,
                    pad_type=config.pad_type,
                    downsample=config.downsample_type,
                    upsample=config.upsample_type,
                    norm_layer=norm_layer,
                    reduction_ratio=reduction_ratio,
                    apply_bn=config.apply_resample_bn,
                    conv_after_downsample=config.conv_after_downsample,
                    redundant_bias=config.redundant_bias,
                )
                in_chs = config.fpn_channels
                reduction = int(reduction * reduction_ratio)
                feature_info.append(dict(num_chs=in_chs, reduction=reduction))

        self.cell = SequentialList()
        for rep in range(config.fpn_cell_repeats):
            logging.debug('building cell {}'.format(rep))
            fpn_layer = BiFpnLayer(
                feature_info=feature_info,
                fpn_config=fpn_config,
                fpn_channels=config.fpn_channels,
                num_levels=config.num_levels,
                pad_type=config.pad_type,
                downsample=config.downsample_type,
                upsample=config.upsample_type,
                norm_layer=norm_layer,
                act_layer=act_layer,
                separable_conv=config.separable_conv,
                apply_resample_bn=config.apply_resample_bn,
                conv_after_downsample=config.conv_after_downsample,
                conv_bn_relu_pattern=config.conv_bn_relu_pattern,
                redundant_bias=config.redundant_bias,
            )
            self.cell.add_module(str(rep), fpn_layer)
            feature_info = fpn_layer.feature_info

    def forward(self, x: List[torch.Tensor]):
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        x = self.cell(x)
        return x