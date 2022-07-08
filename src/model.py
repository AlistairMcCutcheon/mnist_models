import torchvision
import torch
import numpy as np
from batch_metrics import BatchMetrics


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

            batch_metrics = BatchMetrics(labels, outputs)
            accuracy_metrics = batch_metrics.get_accuracy_metrics()
            epoch_accuracies.append(accuracy_metrics["average_accuracy"])

        train_metrics = {
            "epoch_losses": epoch_losses,
            "epoch_accuracies": epoch_accuracies,
        }
        return train_metrics

    def test_one_epoch(self):
        epoch_losses = []
        epoch_accuracies = []

        incorrect_images = []
        incorrect_labels = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                images, labels = batch

                outputs = self.network(images)
                loss = self.loss_function(outputs, labels)

                epoch_losses.append(loss.item())

                batch_metrics = BatchMetrics(labels, outputs)
                accuracy_metrics = batch_metrics.get_accuracy_metrics()
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
