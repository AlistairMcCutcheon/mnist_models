import gzip
import numpy as np
from matplotlib import pyplot as plt


def get_images(images_path, image_size, num_images):
    with gzip.open(images_path) as file:
        file.read(16)
        buffer = file.read(image_size * image_size * num_images)
    train_images = np.frombuffer(buffer, dtype=np.uint8)
    train_images = train_images.reshape(num_images, 1, image_size, image_size)
    return train_images


def get_labels(labels_path, num_images):
    with gzip.open(labels_path) as file:
        file.read(8)
        buffer = file.read(num_images)
    train_labels = np.frombuffer(buffer, dtype=np.uint8)
    return train_labels


image_size = 28
num_images = 5
# http://yann.lecun.com/exdb/mnist/
train_images_path = "data/train-images-idx3-ubyte.gz"
train_labels_path = "data/train-labels-idx1-ubyte.gz"
train_images = get_train_images(train_images_path, image_size, num_images)
train_labels = get_train_labels(train_labels_path, num_images)

print(train_images.shape)
print(train_labels.shape)

print(train_labels)
for image, label in zip(train_images, train_labels):
    print(label)
    image = np.asarray(image).squeeze()
    plt.imshow(image)
    plt.show()
