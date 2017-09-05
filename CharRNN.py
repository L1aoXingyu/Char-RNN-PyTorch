#建立Char RNN模型

import torch
from torch import nn


class CharRNN(nn.Module):
    def __init__(self, num_classes, embed_dim, hidden_size, num_layers,
                 dropout):
        super(CharRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.word_to_vec = nn.Embedding(num_classes, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_size, num_layers, dropout)
        self.proj = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch = x.size(0)
        word_embed = self.word_to_vec(x)  # batch x len x embed
        word_embed = torch.transpose(word_embed, 0, 1)  # len x batch x embed
        out, _ = self.rnn(word_embed)  # len x batch x hidden
        out = torch.transpose(out, 0, 1)  # batch x len x hidden
        return out.view(-1, out.size(2))
