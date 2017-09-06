#建立Char RNN模型

import torch
from torch import nn
from torch.autograd import Variable


class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers,
                 dropout):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hs=None):
        batch = x.size(0)
        if hs is None:
            hs = Variable(
                torch.zeros(self.num_layers, batch, self.hidden_size))
            if torch.cuda.is_available():
                hs = hs.cuda()
        word_embed = self.word_to_vec(x)  # batch x len x embed
        word_embed = word_embed.permute(1, 0, 2)  # len x batch x embed
        out, h0 = self.rnn(word_embed, hs)  # len x batch x hidden
        out = out.permute(1, 0, 2).contiguous()  # batch x len x hidden
        return out.view(-1, out.size(2)), h0
