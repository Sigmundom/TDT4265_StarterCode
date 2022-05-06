from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomHorizontalFlip, RandomSampleCrop, RandomErasing)
import torchvision
# The line belows inherits the configuration set for the tdt4265 dataset
from ..task2_4_adapt.fl_alpha import (
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

