from ssd.modeling import RetinaNet
from tops.config import LazyCall as L

from .fpn import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    # model,
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1,  # Add 1 for background
    improved_weight_init=True
)

anchors.aspect_ratios=[[2,3], [2, 3], [2, 3], [2, 3], [2,3], [2,3]] 