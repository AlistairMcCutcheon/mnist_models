import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import DatasetMNIST
import torchvision
import torch.optim as optim
from convolutional_network import ConvNet
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
from model import Model


def get_dataloaders(dataset, batch_size, train_test_val_split):
    assert sum(train_test_val_split) == 1
    dataset_split_numbers = np.multiply(train_test_val_split, len(dataset)).astype(
        np.int32
    )
    assert sum(dataset_split_numbers) == len(dataset)

    datasets = random_split(dataset, dataset_split_numbers)
    train_dataset, test_dataset, val_dataset = datasets

    train_dataloader = (
        DataLoader(train_dataset, batch_size, shuffle=True)
        if train_test_val_split[0] != 0
        else None
    )
    test_dataloader = (
        DataLoader(test_dataset, batch_size, shuffle=True)
        if train_test_val_split[1] != 0
        else None
    )
    val_dataloader = (
        DataLoader(val_dataset, batch_size, shuffle=True)
        if train_test_val_split[2] != 0
        else None
    )
    return train_dataloader, test_dataloader, val_dataloader


dataset = DatasetMNIST(data_path="data/", image_size=28)
batch_size = 32
train_test_val_split = (0.8, 0.2, 0)
train_dataloader, test_dataloader, val_dataloader = get_dataloaders(
    dataset, batch_size, train_test_val_split
)

network = ConvNet()
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(network.parameters(), lr=0.0001, momentum=0.9)
model = Model(network, loss_function, optimiser, train_dataloader, test_dataloader)

writer = SummaryWriter()
images, _ = model.get_arbitrary_batch()
image_grid = model.get_image_grid(images)
writer.add_image("Batch of images", image_grid)
writer.add_graph(model.network, images)

epochs = 2
for epoch in range(epochs):
    print(epoch)
    train_metrics = model.train_one_epoch()
    average_train_loss = np.mean(train_metrics["epoch_losses"])
    average_train_accuracy = np.mean(train_metrics["epoch_accuracies"])

    writer.add_scalar("Loss/train", average_train_loss, epoch)
    writer.add_scalar("Accuracy/train", average_train_accuracy, epoch)

    test_metrics = model.test_one_epoch()
    average_test_loss = np.mean(test_metrics["epoch_losses"])
    average_test_accuracy = np.mean(test_metrics["epoch_accuracies"])

    writer.add_scalar("Loss/test", average_test_loss, epoch)
    writer.add_scalar("Accuracy/test", average_test_accuracy, epoch)

image_grid = model.get_image_grid(images[:batch_size])
writer.add_image("Sample of Incorrectly Labelled Images", image_grid)
writer.close()
