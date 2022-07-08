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
