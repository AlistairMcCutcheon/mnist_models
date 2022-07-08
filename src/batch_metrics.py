import numpy as np


class BatchMetrics:
    def __init__(self, images, labels, batch_outputs, loss):
        self.images = images
        self.labels = np.array(labels)
        self.loss = loss
        self.batch_outputs = batch_outputs
        (
            self.average_accuracy,
            self.correct_batch_outputs_bool_array,
        ) = self.get_accuracy_metrics()
        self.incorrect_batch_output_images = self.get_incorrect_batch_output_images()

    def get_accuracy_metrics(self):
        correct_array = self.labels[
            np.arange(len(self.batch_outputs)), self.batch_outputs.argmax(1)
        ]
        average_accuracy = np.mean(correct_array)
        return average_accuracy, correct_array

    def get_incorrect_batch_output_images(self):
        incorrect_batch_outputs_indices = np.where(
            self.correct_batch_outputs_bool_array == 0
        )
        return self.images[incorrect_batch_outputs_indices]
