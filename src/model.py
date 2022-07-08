import torchvision
import torch
import numpy as np


class Model:
    def __init__(
        self,
        network,
        loss_function,
        optimiser,
        train_dataloader,
        test_dataloader,
    ) -> None:
        self.network = network
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def get_arbitrary_batch(self):
        return next(iter(self.train_dataloader))

    def get_image_grid(self, images):
        grid = torchvision.utils.make_grid(images, padding=2, pad_value=255)
        return grid.to(torch.uint8)

    def get_accuracy_metrics(self, labels, outputs):
        labels = np.array(labels)
        correct_array = labels[np.arange(len(outputs)), outputs.argmax(1)]
        average_accuracy = np.mean(correct_array)
        accuracy_metrics = {
            "average_accuracy": average_accuracy,
            "correct_array": correct_array,
        }
        return accuracy_metrics

    def train_one_epoch(self):
        epoch_losses = []
        epoch_accuracies = []
        for batch in self.train_dataloader:
            images, labels = batch

            self.optimiser.zero_grad()

            outputs = self.network(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimiser.step()

            epoch_losses.append(loss.item())
            accuracy_metrics = self.get_accuracy_metrics(labels, outputs)
            epoch_accuracies.append(accuracy_metrics["average_accuracy"])

        train_metrics = {
            "epoch_losses": epoch_losses,
            "epoch_accuracies": epoch_accuracies,
        }
        return train_metrics
