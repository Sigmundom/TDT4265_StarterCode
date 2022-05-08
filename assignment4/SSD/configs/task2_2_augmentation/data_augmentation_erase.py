from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Resize,
    GroundTruthBoxesToAnchors, RandomHorizontalFlip, RandomSampleCrop, RandomErasing)
import torchvision
# The line belows inherits the configuration set for the tdt4265 dataset
from ..test_anchors.base import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    val_cpu_transform,
    # train_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    # L(RandomErasing)(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    L(RandomErasing)(p=0.5, scale=(0.005, 0.05), ratio=(0.33, 3.3)),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])