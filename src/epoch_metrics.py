import itertools
import numpy as np


class EpochMetrics:
    def __init__(self):
        self.batches_metrics = []

    def get_average_loss(self):
        return np.mean([batch_metrics.loss for batch_metrics in self.batches_metrics])

    def get_average_accuracies(self):
        return np.mean(
            [batch_metrics.average_accuracy for batch_metrics in self.batches_metrics]
        )

    def get_incorrect_batch_output_images(self):
        incorrect_batches_output_images = [
            batch_metrics.incorrect_batch_output_images
            for batch_metrics in self.batches_metrics
        ]
        return list(itertools.chain(*incorrect_batches_output_images))
