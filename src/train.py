import numpy as np
from matplotlib import pyplot as plt

from dataset import DatasetMNIST
import torch.optim as optim
from convolutional_network import ConvNet
import torch.nn as nn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Model

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomRotation(30),
        transforms.Normalize(0.5, 0.5),
    ]
)
dataset = DatasetMNIST(data_path="data/", image_size=28, transform=transform)
train_dataloader, test_dataloader, val_dataloader = dataset.get_dataloaders(
    batch_size=32, train_test_val_split=(0.8, 0.2, 0)
)

network = ConvNet()
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
model = Model(network, loss_function, optimiser, train_dataloader, test_dataloader)

writer = SummaryWriter()

images, _ = model.get_arbitrary_batch()
image_grid = model.get_image_grid(images)
writer.add_image("Batch of images", image_grid)
writer.add_graph(model.network, images)

epochs = 100
for epoch in range(epochs):
    print(epoch)
    train_metrics = model.train_one_epoch()
    writer.add_scalar("Loss/train", train_metrics.get_average_loss(), epoch)
    writer.add_scalar("Accuracy/train", train_metrics.get_average_accuracies(), epoch)

    test_metrics = model.test_one_epoch()
    writer.add_scalar("Loss/test", test_metrics.get_average_loss(), epoch)
    writer.add_scalar("Accuracy/test", test_metrics.get_average_accuracies(), epoch)

image_grid = model.get_image_grid(test_metrics.get_incorrect_batch_output_images()[:64])
writer.add_image("Sample of Incorrectly Labelled Images", image_grid)
writer.close()
