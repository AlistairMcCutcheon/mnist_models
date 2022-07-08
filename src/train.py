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


def get_accuracy_metrics(labels, outputs):
    labels = np.array(labels)
    correct_array = labels[np.arange(len(outputs)), outputs.argmax(1)]
    average_accuracy = np.mean(correct_array)
    accuracy_metrics = {
        "average_accuracy": average_accuracy,
        "correct_array": correct_array,
    }
    return accuracy_metrics


def train_one_epoch(model):
    epoch_losses = []
    epoch_accuracies = []
    for batch in model.train_dataloader:
        images, labels = batch

        model.optimiser.zero_grad()

        outputs = model.network(images)
        loss = model.loss_function(outputs, labels)
        loss.backward()
        model.optimiser.step()

        epoch_losses.append(loss.item())
        accuracy_metrics = get_accuracy_metrics(labels, outputs)
        epoch_accuracies.append(accuracy_metrics["average_accuracy"])

    train_metrics = {"epoch_losses": epoch_losses, "epoch_accuracies": epoch_accuracies}
    return train_metrics


def test_one_epoch(model):
    epoch_losses = []
    epoch_accuracies = []

    incorrect_images = []
    incorrect_labels = []

    with torch.no_grad():
        for batch in model.test_dataloader:
            images, labels = batch

            outputs = model.network(images)
            loss = model.loss_function(outputs, labels)

            epoch_losses.append(loss.item())
            accuracy_metrics = get_accuracy_metrics(labels, outputs)
            epoch_accuracies.append(accuracy_metrics["average_accuracy"])

            incorrect_indices = np.where(accuracy_metrics["correct_array"] == 0)
            incorrect_images.extend(images[incorrect_indices])
            incorrect_labels.extend(outputs[incorrect_indices])

    test_metrics = {
        "epoch_losses": epoch_losses,
        "epoch_accuracies": epoch_accuracies,
        "incorrect_images": incorrect_images,
        "incorrect_labels": incorrect_labels,
    }

    return test_metrics


image_size = 28
data_path = "data/"
dataset = DatasetMNIST(data_path, image_size)

data_split_fractions = [0.8, 0.2]
assert sum(data_split_fractions) == 1
data_split_numbers = np.multiply(data_split_fractions, len(dataset)).astype(np.int32)
assert sum(data_split_numbers) == len(dataset)
datasets = random_split(dataset, data_split_numbers)
train_dataset, test_dataset = datasets

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

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
    train_metrics = train_one_epoch(model)
    average_train_loss = np.mean(train_metrics["epoch_losses"])
    average_train_accuracy = np.mean(train_metrics["epoch_accuracies"])

    writer.add_scalar("Loss/train", average_train_loss, epoch)
    writer.add_scalar("Accuracy/train", average_train_accuracy, epoch)

    test_metrics = test_one_epoch(model)
    average_test_loss = np.mean(test_metrics["epoch_losses"])
    average_test_accuracy = np.mean(test_metrics["epoch_accuracies"])

    writer.add_scalar("Loss/test", average_test_loss, epoch)
    writer.add_scalar("Accuracy/test", average_test_accuracy, epoch)

image_grid = model.get_image_grid(images[:batch_size])
writer.add_image("Sample of Incorrectly Labelled Images", image_grid)
writer.close()
