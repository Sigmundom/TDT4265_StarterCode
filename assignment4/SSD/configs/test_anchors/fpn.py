from tops.config import LazyCall as L
from ssd.modeling.backbones import FPN

# The line belows inherits the configuration set for the tdt4265 dataset
from .data_augmentation import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    # backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

backbone = L(FPN)(
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
    type="resnet34", 
    pretrained=True
    )
