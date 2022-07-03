import gzip
import numpy as np
from matplotlib import pyplot as plt


def get_train_images(train_images_path, image_size, num_images):
    with gzip.open(train_images_path) as file:
        file.read(16)
        buffer = file.read(image_size * image_size * num_images)
    train_images = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
    train_images = train_images.reshape(num_images, 1, image_size, image_size)
    return train_images


image_size = 28
num_images = 5
# http://yann.lecun.com/exdb/mnist/
train_images_path = "data/mnist/train-images-idx3-ubyte.gz"
train_images = get_train_images(train_images_path, image_size, num_images)

print(train_images.shape)

image = np.asarray(train_images[0]).squeeze()
plt.imshow(image)
plt.show()



