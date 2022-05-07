import torch
from ssd.modeling.backbones import FPN
from tops.config import LazyCall as L

from .adjust_parameters import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    # backbone,
    data_train,
    data_val,
    val_cpu_transform,
    # train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

from ..task2_2_augmentation.t_2_2_data_augmentation_erase import train_cpu_transform

backbone = L(FPN)(
    output_channels=64,
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
    type="resnet50", 
    pretrained=True
    )

optimizer = L(torch.optim.SGD)(
    # Tip: Scale the learning rate by batch size! 2.6e-3 is set for a batch size of 32. use 2*2.6e-3 if you use 64
    lr=7e-3, momentum=0.9, weight_decay=0.0005
)
