import torchvision
from torch import nn
from task2 import create_plots
# import pathlib
# import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = torchvision.models.resnet18(pretrained=True)
    self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
    # as this is done in nn.CrossEntropyLoss
    for param in self.model.parameters(): # Freeze all parameters
      param.requires_grad = False
    for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
      param.requires_grad = True # layer
    for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
      param.requires_grad = True # layers

  def forward(self, x):
    x = self.model(x)
    return x


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result!
    utils.set_seed(10)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model()
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    trainer.final_accuracy()
    create_plots(trainer, "task4")

if __name__ == "__main__":
    main()