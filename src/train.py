import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import DatasetMNIST
import torchvision
import torch.optim as optim
from convolutional_network import ConvNet
import torch.nn as nn

image_size = 28
data_path = "data/"
dataset = DatasetMNIST(data_path, image_size)

data_split_fractions = [0.8, 0.2]
assert sum(data_split_fractions) == 1
data_split_numbers = np.multiply(data_split_fractions, len(dataset)).astype(np.int32)
assert sum(data_split_numbers) == len(dataset)
datasets = random_split(dataset, data_split_numbers)
train_dataset, test_dataset = datasets

batch_size = 4
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

learning_rate = 0.001
momentum = 0.9
network = ConvNet()
loss_function = nn.CrossEntropyLoss()
optimiser = optim.SGD(network.parameters(), learning_rate, momentum)

epochs = 10
for epoch in range(epochs):
    loss_history = []
    accuracy_history = []
    for i, data in enumerate(train_dataloader):
        images, labels = data

        optimiser.zero_grad()

        outputs = network(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()

        loss_history.append(loss.item())

        labels = np.array(labels)
        correct_array = labels[np.arange(len(outputs)), outputs.argmax(1)]
        accuracy_history.append(np.mean(correct_array))

    print(f"Epoch: {epoch}")
    print(f"Running Loss: {np.mean(loss_history)}")
    print(f"Running Accuracy: {np.mean(accuracy_history)}")


# batch_images, batch_labels = next(iter(train_dataloader))
# print(batch_labels)
# image_grid = torchvision.utils.make_grid(batch_images.int(), padding=2, pad_value=255)
# plt.imshow(np.transpose(image_grid, (1, 2, 0)))
# plt.show()
