import torch
from torch.autograd import Variable
use_gpu = torch.cuda.is_available()


def pick_top_n(preds, vocab_size, top_n=5):
    pass


def train_epoch(model, dataloader, optimizer, criterion):
    running_loss = 0.0
    for batch in dataloader:
        x, y = batch
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)
