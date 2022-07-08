import numpy as np


class BatchMetrics:
    def __init__(self, labels, batch_outputs):
        self.labels = np.array(labels)
        self.batch_outputs = batch_outputs

    def get_accuracy_metrics(self):
        correct_array = self.labels[
            np.arange(len(self.batch_outputs)), self.batch_outputs.argmax(1)
        ]
        average_accuracy = np.mean(correct_array)
        accuracy_metrics = {
            "average_accuracy": average_accuracy,
            "correct_array": correct_array,
        }
        return accuracy_metrics
