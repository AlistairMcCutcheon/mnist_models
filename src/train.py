import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import DatasetMNIST
import torchvision
from convolutional_network import ConvNet

image_size = 28
data_path = "data/"
dataset = DatasetMNIST(data_path, image_size)
print(len(dataset))

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

batch_images, batch_labels = next(iter(dataloader))

# image_grid = torchvision.utils.make_grid(batch_images, padding=2, pad_value=255)
# plt.imshow(np.transpose(image_grid, (1, 2, 0)))
# plt.show()

network = ConvNet()
out = network.forward(batch_images)
print(type(out))
print(out.shape)
print(out)
