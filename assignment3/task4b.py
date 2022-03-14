import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]

def task4b():
    activation = first_conv_layer(image)
    print("Activation shape:", activation.shape)

    weights = first_conv_layer.weight[indices]
    filters = activation[0, indices]
    print(filters.shape)

    fig = plt.figure(figsize=(10,10))

    columns = 5
    rows = 2

    for i, weight in enumerate(weights):
        fig.add_subplot(rows, columns, i+1)
        img = torch_image_to_numpy(weight)
        plt.imshow(img)

    for i, filter in enumerate(filters):
        fig.add_subplot(rows, columns, i+columns+1)
        img = torch_image_to_numpy(filter)
        plt.imshow(img)
    print('Plotting')
    plt.savefig('task4b.png')
    plt.show()

def task4c():
    layers = list(model.children())[:-2]
    print(layers)
    activation = image
    for layer in layers:
        activation = layer(activation)

    print("Activation shape:", activation.shape)
    filters = activation[0, :10]

    fig  = plt.figure(figsize=(7,7))

    for i, filter in enumerate(filters):
        fig.add_subplot(2, 5, i+1)
        img = torch_image_to_numpy(filter)
        plt.imshow(img)

    print('Plotting')
    plt.savefig('task4c.png')
    plt.show()




if __name__ == '__main__':
    # task4b()
    task4c()
