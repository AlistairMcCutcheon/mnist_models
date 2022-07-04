import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import DatasetMNIST
import torchvision
import torch.optim as optim
from convolutional_network import ConvNet
import torch.nn as nn
import torch


def get_accuracy_metrics(labels, outputs):
    labels = np.array(labels)
    correct_array = labels[np.arange(len(outputs)), outputs.argmax(1)]
    average_accuracy = np.mean(correct_array)
    accuracy_metrics = {
        "average_accuracy": average_accuracy,
        "correct_array": correct_array,
    }
    return accuracy_metrics


def train_one_epoch(train_dataloader, optimiser, loss_function):
    epoch_losses = []
    epoch_accuracies = []
    for batch in train_dataloader:
        images, labels = batch

        optimiser.zero_grad()

        outputs = network(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()

        epoch_losses.append(loss.item())
        accuracy_metrics = get_accuracy_metrics(labels, outputs)
        epoch_accuracies.append(accuracy_metrics["average_accuracy"])

    train_metrics = {"epoch_losses": epoch_losses, "epoch_accuracies": epoch_accuracies}
    return train_metrics


def test_one_epoch(test_dataloader, loss_function):
    epoch_losses = []
    epoch_accuracies = []

    incorrect_images = []
    incorrect_labels = []

    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch

            outputs = network(images)
            loss = loss_function(outputs, labels)

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

learning_rate = 0.0001
momentum = 0.9
network = ConvNet()
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(network.parameters(), learning_rate, momentum)

train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

epochs = 5
for epoch in range(epochs):
    train_metrics = train_one_epoch(train_dataloader, optimiser, loss_function)
    average_train_loss = np.mean(train_metrics["epoch_losses"])
    average_train_accuracy = np.mean(train_metrics["epoch_accuracies"])

    train_loss_history.append(average_train_loss)
    train_accuracy_history.append(average_train_accuracy)

    print(f"Epoch: {epoch}")
    print(f"Train Loss: {average_train_loss}")
    print(f"Train Accuracy: {average_train_accuracy}")

    test_metrics = test_one_epoch(test_dataloader, loss_function)
    average_test_loss = np.mean(test_metrics["epoch_losses"])
    average_test_accuracy = np.mean(test_metrics["epoch_accuracies"])

    test_loss_history.append(average_test_loss)
    test_accuracy_history.append(average_test_accuracy)

    print(f"Test Loss: {average_test_loss}")
    print(f"Test Accuracy: {average_test_accuracy}")


plt.plot(range(epochs), train_loss_history, color="red")
plt.plot(range(epochs), test_loss_history, color="blue")
plt.title("Loss per Epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

plt.plot(range(epochs), train_accuracy_history, color="red")
plt.plot(range(epochs), test_accuracy_history, color="blue")
plt.title("Accuracy per Epoch")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.show()

incorrect_dataset = zip(
    test_metrics["incorrect_images"], test_metrics["incorrect_labels"]
)
for image, incorrect_label in incorrect_dataset:
    print(incorrect_label)
    print(image.shape)
    plt.imshow(np.transpose(image, (1, 2, 0)))
    plt.show()

# image_grid = torchvision.utils.make_grid(
#     batch_images.int(), padding=2, pad_value=255
# )
# plt.imshow(np.transpose(image_grid, (1, 2, 0)))
# plt.show()
