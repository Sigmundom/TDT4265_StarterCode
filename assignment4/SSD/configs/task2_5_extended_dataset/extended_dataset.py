import torch
from tops.config import LazyCall as L

from ..task2_4_adapt.anchors_AR import (
    train,
    # optimizer,
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

train.batch_size = 64

optimizer = L(torch.optim.SGD)(
    # Tip: Scale the learning rate by batch size! 2.6e-3 is set for a batch size of 32. use 2*2.6e-3 if you use 64
    lr=1e-2, momentum=0.9, weight_decay=0.0005
)

from ..tdt4265_updated import data_train, data_val

