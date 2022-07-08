import torchvision
import torch


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
