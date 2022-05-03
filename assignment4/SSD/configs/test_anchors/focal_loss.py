from tops.config import LazyCall as L
from ssd.modeling.focal_loss import FocalLoss

# The line belows inherits the configuration set for the tdt4265 dataset
from .fpn import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

loss_objective = L(FocalLoss)(anchors="${anchors}", alpha=[0.01, *[1 for _ in range(model.num_classes-1)]], gamma=1)
