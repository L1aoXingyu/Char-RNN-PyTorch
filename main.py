import argparse
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils import data

from CharRNN import CharRNN
from data_utils import TextConverter, TextData

use_gpu = torch.cuda.is_available()


def train_epoch(model, dataloader, criterion, optimizer):
    running_loss = 0.0
    n_total = 0.0
    for batch in dataloader:
        x, y = batch
        y = y.type(torch.LongTensor)
        mb_size = x.size(0)
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x, y = Variable(x), Variable(y)
        out, _ = model(x)
        batch_loss = criterion(out, y.view(-1))
        # 反向传播
        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 5)
        optimizer.step()
        running_loss += batch_loss.data[0]
        n_total += mb_size
    return running_loss / n_total


def train(n_epoch, model, dataloader, optimizer, criterion):
    for e in range(n_epoch):
        print('{}/{}'.format(e + 1, n_epoch))
        since = time.time()
        loss = train_epoch(model, dataloader, criterion, optimizer)
        print('Loss: {}, Time: {:.4f}'.format(loss, time.time() - since))
        if (e + 1) % 100 == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(),
                       './checkpoints/model_{}.pth'.format(e + 1))


def pick_top_n(preds, top_n=5):
    top_pred_prob, top_pred_label = torch.topk(preds, top_n, 1)
    top_pred_prob /= torch.sum(top_pred_prob)
    top_pred_prob = top_pred_prob.squeeze(0).cpu().numpy()
    top_pred_label = top_pred_label.squeeze(0).cpu().numpy()
    c = np.random.choice(top_pred_label, size=1, p=top_pred_prob)
    return c


def sample(model, checkpoint, convert, arr_to_text, prime, text_len=20):
    '''
    将载入好权重的模型读入，指定开始字符和长度进行生成，将生成的结果保存到txt文件中
    checkpoint: 载入的模型
    convert: 文本和下标转换
    prime: 起始文本
    text_len: 生成文本长度
    '''
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    samples = [convert(c) for c in prime]
    input_txt = torch.LongTensor(samples).unsqueeze(0)
    if use_gpu:
        input_txt = input_txt.cuda()
    input_txt = Variable(input_txt)
    _, init_state = model(input_txt)
    result = samples
    model_input = input_txt[:, -1].unsqueeze(1)
    for i in range(text_len):
        # out是输出的字符，大小为1 x vocab
        # init_state是RNN传递的hidden state
        out, init_state = model(model_input, init_state)
        pred = pick_top_n(out.data)
        model_input = Variable(torch.LongTensor(pred)).unsqueeze(0)
        if use_gpu:
            model_input = model_input.cuda()
        result.append(pred[0])
    return arr_to_text(result)


def main():
    '''main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', required=True, help='训练还是预测, train or eval')
    parser.add_argument('--txt', required=True, help='进行训练的txt文件')
    parser.add_argument('--batch', default=128, type=int, help='训练的batch size')
    parser.add_argument('--epoch', default=5000, type=int, help='跑多少个epoch')
    parser.add_argument('--len', default=100, type=int, help='输入模型的序列长度')
    parser.add_argument(
        '--max_vocab', default=5000, type=int, help='最多存储的字符数目')
    parser.add_argument('--embed', default=512, type=int, help='词向量的维度')
    parser.add_argument('--hidden', default=512, type=int, help='RNN的输出维度')
    parser.add_argument('--n_layer', default=2, type=int, help='RNN的层数')
    parser.add_argument('--dropout', default=0.5, help='RNN中drop的概率')
    parser.add_argument('--begin', default='我', help='给出生成文本的开始')
    parser.add_argument('--pred_len', default=20, type=int, help='生成文本的长度')
    parser.add_argument('--checkpoint', help='载入模型的位置')
    opt = parser.parse_args()
    print(opt)

    convert = TextConverter(opt.txt, max_vocab=opt.max_vocab)
    model = CharRNN(convert.vocab_size, opt.embed, opt.hidden, opt.n_layer,
                    opt.dropout)
    model.load_state_dict(torch.load('./poetry_checkpoints/model_300.pth'))
    if use_gpu:
        model = model.cuda()

    if opt.state == 'train':
        model.train()
        dataset = TextData(opt.txt, opt.len, convert.text_to_arr)
        dataloader = data.DataLoader(
            dataset, opt.batch, shuffle=True, num_workers=4)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(size_average=False)
        train(opt.epoch, model, dataloader, optimizer, criterion)

    elif opt.state == 'eval':
        pred_text = sample(model, opt.checkpoint, convert.word_to_int,
                           convert.arr_to_text, opt.begin, opt.pred_len)
        print(pred_text)
        with open('./generate.txt', 'a') as f:
            f.write(pred_text)
            f.write('\n')
    else:
        print('Error state, must choose from train and eval!')


if __name__ == '__main__':
    main()
