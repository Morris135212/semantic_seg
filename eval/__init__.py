import torch.nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score


def acc_score(y_pred, y):
    y_pred = torch.round(y_pred.reshape(-1)).cpu().detach().numpy()
    y = y.reshape(-1).cpu().detach().numpy()
    score = accuracy_score(y, y_pred)
    return score


class Evaluator:
    def __init__(self, val_loader, model, device, criterion=torch.nn.BCEWithLogitsLoss()):
        self.val_loader = val_loader
        self.model = model
        self.model.eval()
        self.criterion = criterion
        self.device = device

    def __call__(self, *args, **kwargs):
        total_loss = 0.
        total_acc = 0.
        length = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                img, ground = data

                img = Variable(img.float()).to(self.device)
                ground = Variable(ground.float()).to(self.device)
                output = self.model(img)
                output = output.view((output.shape[0], -1))
                total_loss += self.criterion(output, ground)*img.shape[0]
                total_acc += acc_score(output, ground)*img.shape[0]
                length += img.shape[0]
            del img, ground
        return {"loss": total_loss/length, "acc": total_acc/length}