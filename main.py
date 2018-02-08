# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com
"""
import numpy as np
import torch
from mxtorch import meter
from mxtorch.trainer import Trainer, ScheduledOptim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
from config import opt
from data import TextConverter, TextDataset

convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)


def get_data():
    dataset = TextDataset(opt.txt, opt.len, convert.text_to_arr)
    return DataLoader(dataset, opt.batch_size, shuffle=True, num_workers=opt.num_workers)


def get_model(num_classes=convert.vocab_size, embed_dim=512, hidden_size=512, num_layers=2, dropout=0.5):
    model = getattr(models, opt.model)(num_classes, embed_dim, hidden_size, num_layers, dropout)
    if opt.use_gpu:
        model = model.cuda(opt.ctx)
    return model


def get_loss(score, label):
    return nn.CrossEntropyLoss()(score, label.view(-1))


def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    return ScheduledOptim(optimizer)


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


class CharRNNTrainer(Trainer):
    def __init__(self):
        model = get_model()
        criterion = get_loss
        optimizer = get_optimizer(model)
        super().__init__(model, criterion, optimizer)

        self.metric_meter['loss'] = meter.AverageValueMeter()

    def train(self, train_data):
        for data in tqdm(train_data):
            x, y = data
            y = y.long()
            if opt.use_gpu:
                x = x.cuda(opt.ctx)
                y = y.cuda(opt.ctx)
            x, y = Variable(x), Variable(y)

            # Forward.
            score, _ = self.model(x)
            loss = self.criterion(score, y)

            # Backward.
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradient.
            nn.utils.clip_grad_norm(self.model.parameters(), 5)
            self.optimizer.step()

            self.metric_meter['loss'].add(loss.data[0])

            # Update to tensorboard.
            # if (self.n_iter + 1) % opt.print_freq == 0:
            #     self.writer.add_scalar('loss', self.metric_meter['loss'].value()[0], self.n_plot)
            #     self.n_plot += 1

            self.n_iter += 1

        # Log the train metrics to dict.
        self.metric_log['train loss'] = self.metric_meter['loss'].value()[0]
        self.metric_log['perplexity'] = np.exp(self.metric_meter['loss'].value()[0])

    def load_state_dict(self, checkpoints):
        self.model.load_state_dict(torch.load(checkpoints))

    def predict(self, begin, text_len=20):
        """Set beginning word and predicted length, using model to generate texts.

        Args:
            begin (torch.LongTensor): index of begin words, shape is :math:`[1, len]`
            text_len (int): length of generate text

        Returns:

        """
        self.model.eval()
        samples = [convert(c) for c in begin]
        input_txt = torch.LongTensor(samples)[None]
        if opt.use_gpu:
            input_txt = input_txt.cuda(opt.ctx)
        input_txt = Variable(input_txt)
        _, init_state = self.model(input_txt)
        result = samples
        model_input = input_txt[:, -1][:, None]
        for i in range(text_len):
            out, init_state = self.model(model_input, init_state)
            pred = pick_top_n(out.data)
            model_input = Variable(torch.LongTensor(pred))[None]
            if opt.use_gpu:
                model_input = model_input.cuda(opt.ctx)
            result.append(pred[0])
        return result


def train(**kwargs):
    opt._parse(kwargs)
    train_data = get_data()
    char_rnn_trainer = CharRNNTrainer()
    char_rnn_trainer.fit(train_data)


if __name__ == '__main__':
    import fire

    fire.Fire()
