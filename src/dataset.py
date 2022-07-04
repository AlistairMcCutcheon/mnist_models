import numpy as np
import gzip
import os
from torch.utils.data import Dataset


class DatasetMNIST(Dataset):
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
        self.image_size = image_size

        train_images_path = os.path.join(data_path, "train-images-idx3-ubyte.gz")
        train_labels_path = os.path.join(data_path, "train-labels-idx1-ubyte.gz")
        test_images_path = os.path.join(data_path, "t10k-images-idx3-ubyte.gz")
        test_labels_path = os.path.join(data_path, "t10k-labels-idx1-ubyte.gz")

        train_images = get_images(train_images_path, image_size)
        train_labels = get_labels(train_labels_path)
        test_images = get_images(test_images_path, image_size)
        test_labels = get_labels(test_labels_path)

        self.all_images = np.concatenate((train_images, test_images))
        all_labels = np.concatenate((train_labels, test_labels))
        assert len(self.all_images) == len(all_labels)

        self.one_hot_labels = np.zeros((len(all_labels), 10))
        self.one_hot_labels[np.arange(len(all_labels)), all_labels] = 1

        self.all_images = self.all_images.astype(np.float32)
        self.one_hot_labels = self.one_hot_labels.astype(np.float32)

    def __len__(self):
        assert len(self.all_images) == len(self.one_hot_labels)
        return len(self.all_images)

    def __getitem__(self, index):
        image = self.all_images[index]
        label = self.one_hot_labels[index]
        return image, label
