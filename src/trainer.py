class Trainer:
    def __init__(
        self,
        epochs=200,
        lr=0.0002,
        adam=True,
        SGD=False,
        device="mps",
        lr_scheduler=False,
        l1_loss=False,
        l2_loss=False,
        elastic_net=False,
        is_display=False,
    ):

        self.epochs = epochs
        self.lr = lr
        self.adam = adam
        self.SGD = SGD
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.l1_loss = l1_loss
        self.l2_loss = l2_loss
        self.elastic_net = elastic_net
        self.is_display = is_display

    def l1(self, model):
        pass

    def l2(self, model):
        pass

    def elastic_net(self, model):
        pass

    def saved_checkpoints(self, **kwargs):
        pass

    def saved_metrics(self, **kwargs):
        pass

    def saved_train_images(self, **kwargs):
        pass

    def update_train_model(self, **kwargs):
        pass

    def show_progress(self, **kwargs):
        pass

    def train(self):
        pass

    @staticmethod
    def plot_history():
        pass
