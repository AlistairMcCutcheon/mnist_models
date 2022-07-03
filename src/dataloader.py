import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from dataset import DatasetMNIST

image_size = 28
data_path = "data/"
dataset = DatasetMNIST(data_path, image_size)
print(len(dataset))

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

batch_images, batch_labels = next(iter(dataloader))
print(batch_images.shape)
print(batch_labels.shape)
for image, label in zip(batch_images, batch_labels):
    print(label)
    image = np.asarray(image).squeeze()
    plt.imshow(image)
    plt.show()
