from ssd.modeling import FocalLoss
from tops.config import LazyCall as L
# The line belows inherits the configuration set for the tdt4265 dataset
from .anchors_AR import (
    train,
    optimizer,
    schedulers,
    # loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors,
)

# ("background", "car", "truck", "bus", "motorcycle", "bicycle", "scooter", "person", "rider")
alpha = [0.01, 1, 2, 2, 2, 2, 2, 1, 2]
loss_objective = L(FocalLoss)(anchors="${anchors}", alpha=alpha, gamma=1)




