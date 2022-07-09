import numpy as np
import gzip
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class DatasetMNIST(Dataset):
    def __init__(self, data_path, image_size=28, transform=None):
        def get_images(images_path, image_size):
            with gzip.open(images_path) as file:
                file.read(16)
                buffer = file.read()
            images = np.frombuffer(buffer, dtype=np.uint8)
            images = images.reshape(-1, image_size, image_size, 1)
            # images = np.divide(images, 255)
            return images

        def get_labels(labels_path):
            with gzip.open(labels_path) as file:
                file.read(8)
                buffer = file.read()
            labels = np.frombuffer(buffer, dtype=np.uint8)
            return labels

        # http://yann.lecun.com/exdb/mnist/
        self.data_path = data_path
        self.image_size = image_size
        self.transform = transform

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
        if self.transform is not None:
            image = self.transform(image)
        label = self.one_hot_labels[index]
        return image, label

    def get_dataloaders(self, batch_size, train_test_val_split):
        assert sum(train_test_val_split) == 1
        dataset_split_numbers = np.multiply(train_test_val_split, len(self)).astype(
            np.int32
        )
        assert sum(dataset_split_numbers) == len(self)

        datasets = random_split(self, dataset_split_numbers)
        train_dataset, test_dataset, val_dataset = datasets

        train_dataloader = (
            DataLoader(train_dataset, batch_size, shuffle=True)
            if train_test_val_split[0] != 0
            else None
        )
        test_dataloader = (
            DataLoader(test_dataset, batch_size, shuffle=True)
            if train_test_val_split[1] != 0
            else None
        )
        val_dataloader = (
            DataLoader(val_dataset, batch_size, shuffle=True)
            if train_test_val_split[2] != 0
            else None
        )
        return train_dataloader, test_dataloader, val_dataloader
