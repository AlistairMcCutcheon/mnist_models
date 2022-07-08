import torchvision
import torch
import numpy as np
from batch_metrics import BatchMetrics
from epoch_metrics import EpochMetrics


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
        epoch_metrics = EpochMetrics()
        for batch in self.train_dataloader:
            images, labels = batch

            self.optimiser.zero_grad()
            outputs = self.network(images)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimiser.step()

            epoch_metrics.batches_metrics.append(
                BatchMetrics(images, labels, outputs, loss.item())
            )
        return epoch_metrics

    def test_one_epoch(self):
        epoch_metrics = EpochMetrics()
        with torch.no_grad():
            for batch in self.test_dataloader:
                images, labels = batch

                self.optimiser.zero_grad()
                outputs = self.network(images)
                loss = self.loss_function(outputs, labels)

                epoch_metrics.batches_metrics.append(
                    BatchMetrics(images, labels, outputs, loss.item())
                )
        return epoch_metrics
