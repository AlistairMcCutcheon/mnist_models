import gzip
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os


class DatasetMNIST:
    def __init__(self, data_path, image_size=28):
        # http://yann.lecun.com/exdb/mnist/
        self.data_path = data_path

        train_images_path = os.path.join(data_path, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_path, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(data_path, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_path, "t10k-labels-idx1-ubyte.gz")

        train_images = self.get_images(train_images_path, image_size)
        train_labels = self.get_labels(train_labels_path)
        test_images = self.get_images(test_images_path, image_size)
        test_labels = self.get_labels(test_labels_path)

        print("Train images shape:")
        print(train_images.shape)
        print(train_labels.shape)
        print("Test images shape")
        print(test_images.shape)
        print(test_labels.shape)

        all_images = np.concatenate((train_images, test_images))
        all_labels = np.concatenate((train_labels, test_labels))

        print("All images shape:")
        print(all_images.shape)
        print(all_labels.shape)

        for image, label in zip(
            np.flip(all_images, axis=0), np.flip(all_labels, axis=0)
        ):
            print(label)
            image = np.asarray(image).squeeze()
            plt.imshow(image)
            plt.show()

    def get_images(self, images_path, image_size):
        with gzip.open(images_path) as file:
            file.read(16)
            buffer = file.read()
        train_images = np.frombuffer(buffer, dtype=np.uint8)
        train_images = train_images.reshape(-1, 1, image_size, image_size)
        return train_images

    def get_labels(self, labels_path):
        with gzip.open(labels_path) as file:
            file.read(8)
            buffer = file.read()
        train_labels = np.frombuffer(buffer, dtype=np.uint8)
        return train_labels


image_size = 28
data_path = "data/"
DatasetMNIST(data_path, image_size)
