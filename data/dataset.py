# encoding: utf-8
"""
@author: xyliao
@contact: xyliao1993@qq.com

This is the document of the file.
"""

import numpy as np
import torch


class TextConverter(object):
    def __init__(self, text_path, max_vocab=5000):
        with open(text_path, 'r') as f:
            text_file = f.readlines()
        word_list = [v for s in text_file for v in s]
        vocab = set(word_list)

        # If the number of words is larger than limit, clip the words with minimum frequency.
        vocab_count = {}
        for word in vocab:
            vocab_count[word] = 0
        for word in word_list:
            vocab_count[word] += 1
        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab

        self.word_to_int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int_to_word_table = dict(enumerate(self.vocab))

    @property
    def vocab_size(self):
        return len(self.vocab) + 1

    def word_to_int(self, word):
        if word in self.word_to_int_table:
            return self.word_to_int_table[word]
        else:
            return len(self.vocab)

    def int_to_word(self, index):
        if index == len(self.vocab):
            return '<unk>'
        elif index < len(self.vocab):
            return self.int_to_word_table[index]
        else:
            raise Exception('Unknown index!')

    def text_to_arr(self, text):
        arr = []
        for word in text:
            arr.append(self.word_to_int(word))
        return np.array(arr)

    def arr_to_text(self, arr):
        words = []
        for index in arr:
            words.append(self.int_to_word(index))
        return "".join(words)


class TextDataset(object):
    def __init__(self, text_path, n_step, arr_to_idx):

        with open(text_path, 'r') as f:
            content = f.readline()
        text = [v for s in content for v in s]
        num_seq = int(len(text) / n_step)
        self.num_seq = num_seq
        self.n_step = n_step
        # Clip more than maximum length.
        text = text[:num_seq * n_step]
        arr = arr_to_idx(text)
        arr = arr.reshape((num_seq, -1))
        self.arr = torch.from_numpy(arr)

    def __getitem__(self, item):
        x = self.arr[item, :]
        y = torch.zeros(x.shape)
        y[:-1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.num_seq
