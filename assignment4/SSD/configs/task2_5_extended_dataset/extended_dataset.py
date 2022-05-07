from tops.config import LazyCall as L

from ..task2_4_adapt.anchors_AR import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    # data_train,
    # data_val,
    val_cpu_transform,
    train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)

from ..tdt4265_updated import data_train, data_val

