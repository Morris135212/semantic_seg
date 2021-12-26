import torch.nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    def __init__(self, val_loader, model, device, criterion=torch.nn.BCELoss()):
        self.val_loader = val_loader
        self.model = model
        self.model.eval()
        self.criterion = criterion
        self.device = device

    def __call__(self, *args, **kwargs):
        total_loss = 0.
        length = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                img, ground = data

                img = Variable(img.float()).to(self.device)
                ground = Variable(ground.float()).to(self.device)
                output = self.model(img)
                output = output.view((output.shape[0], -1))
                total_loss += self.criterion(torch.sigmoid(output), ground)*img.shape[0]
                length += img.shape[0]
            del img, ground
        return {"loss": total_loss/length}