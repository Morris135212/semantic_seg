import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


import transforms
from data import CustomDataset
from eval import Evaluator
from utils.torchtools import EarlyStopping


class Trainer:
    def __init__(self,
                 train: tuple,
                 val: tuple,
                 model,
                 batch_size=32,
                 epochs=10,
                 optimizer="adam",
                 lr=1e-2,
                 momentum=0.5,
                 step_size=20,
                 interval=1,
                 patience=20,
                 include_weight=True,
                 path="output/checkpoints/checkpoint.pt"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        train_data = CustomDataset(train[0], train[1], transforms.default_transform["train"])
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_data = CustomDataset(val[0], val[1], transforms.default_transform["val"])
        self.val_loader = DataLoader(val_data, batch_size=batch_size)
        self.batch_size = batch_size
        self.epochs = epochs
        # Loss function
        # self.criterion = torch.nn.CrossEntropyLoss().to(device=self.device)
        # self.criterion = torch.nn.BCELoss().to(device=self.device)
        # Model
        self.model = model.to(device=self.device)
        # Optimizer
        if optimizer == "adam":
            self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=step_size, gamma=0.5)
        self.interval = interval
        self.include_weight = include_weight
        self.early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)

    def __call__(self, *args, **kwargs):
        print("Start Training")
        for epoch in range(self.epochs):
            print(f"At epoch: {epoch}")
            epoch_loss, epoch_acc = 0., 0.
            length = 0
            for i, data in enumerate(tqdm(self.train_loader), 0):
                self.model.train()
                img, ground, weight = data
                if self.include_weight:
                    weight = weight.to(self.device)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight).to(self.device)
                else:
                    criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
                img = Variable(img.float()).to(self.device)
                ground = Variable(ground.float()).to(self.device)
                output = self.model(img)
                output = output.view((output.shape[0], -1))
                loss = criterion(output, ground)
                epoch_loss += loss.item()*ground.shape[0]
                length += ground.shape[0]

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.scheduler.step()
                del img, ground
                if i % self.interval == self.interval - 1:
                    evaluator = Evaluator(val_loader=self.val_loader,
                                          model=self.model,
                                          device=self.device)
                    results = evaluator()
                    loss = results["loss"]
                    print(f"train loss: {epoch_loss / length}, eval loss: {loss}", flush=True)
                    self.early_stopping(loss, self.model)
            if self.early_stopping.early_stop:
                break



