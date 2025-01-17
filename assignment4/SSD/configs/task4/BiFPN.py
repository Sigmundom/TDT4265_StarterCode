from ssd.modeling.backbones import BiFPN
from tops.config import LazyCall as L
# The line belows inherits the configuration set for the tdt4265 dataset
from ..task2_5_extended_dataset.extended_dataset import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    # backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors,
)

backbone = L(BiFPN)(
    output_channels=128,
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}",
    type="resnet34", 
    pretrained=True
    )




