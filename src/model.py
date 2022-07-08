class Model:
    def __init__(
        self, network, loss_function, optimiser, train_dataloader, test_dataloader
    ) -> None:
        self.network = network
        self.loss_function = loss_function
        self.optimiser = optimiser
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
