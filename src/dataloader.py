import gzip
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import os


class DatasetMNIST:
    def __init__(self, data_path, image_size=28):
        def get_images(images_path, image_size):
            with gzip.open(images_path) as file:
                file.read(16)
                buffer = file.read()
            train_images = np.frombuffer(buffer, dtype=np.uint8)
            train_images = train_images.reshape(-1, 1, image_size, image_size)
            return train_images

        def get_labels(labels_path):
            with gzip.open(labels_path) as file:
                file.read(8)
                buffer = file.read()
            train_labels = np.frombuffer(buffer, dtype=np.uint8)
            return train_labels

        # http://yann.lecun.com/exdb/mnist/
        self.data_path = data_path

        train_images_path = os.path.join(data_path, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_path, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(data_path, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_path, "t10k-labels-idx1-ubyte.gz")

        train_images = get_images(train_images_path, image_size)
        train_labels = get_labels(train_labels_path)
        test_images = get_images(test_images_path, image_size)
        test_labels = get_labels(test_labels_path)

        print("Train images shape:")
        print(train_images.shape)
        print(train_labels.shape)
        print("Test images shape")
        print(test_images.shape)
        print(test_labels.shape)

        self.all_images = np.concatenate((train_images, test_images))
        self.all_labels = np.concatenate((train_labels, test_labels))
        assert len(self.all_images) == len(self.all_labels)

        print("All images shape:")
        print(self.all_images.shape)
        print(self.all_labels.shape)

        # for image, label in zip(
        #     np.flip(self.all_images, axis=0), np.flip(self.all_labels, axis=0)
        # ):
        #     print(label)
        #     image = np.asarray(image).squeeze()
        #     plt.imshow(image)
        #     plt.show()

    def __len__(self):
        assert len(self.all_images) == len(self.all_labels)
        return len(self.all_images)


image_size = 28
data_path = "data/"
dataset = DatasetMNIST(data_path, image_size)
print(len(dataset))
