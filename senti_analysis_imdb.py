import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torchtext import data
from torchtext import vocab
from torchtext import datasets
import numpy as np
from matplotlib import pyplot as plt

#ハイパーパラメータ
batch_size = 32
output_size = 2 #positive or negative
hidden_size = 256 #lstmの出力次元
embedding_length = 300

#前処理用機能のFieldのセットアップ
tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True, include_lengths=True, batch_first=True, fix_length=200)

#LabelField
LABEL = data.LabelField()

train_dataset, test_dataset = datasets.IMDB.splits(TEXT, LABEL)
train_dataset, val_dataset = train_dataset.split()

TEXT.build_vocab(train_dataset, min_freq=3, vectors=vocab.GloVe(name='6B', dim=300)) #学習済み埋め込みベクトル glove.6B.300d.txtを使用
LABEL.build_vocab(train_dataset) #単語の辞書を作成

print(TEXT.vocab.freqs.most_common(10)) #単語件数top10
print(LABEL.vocab.freqs) #ラベルごとの件数
print(TEXT.vocab.itos[:10]) #単語10個

#バッチ単位にする
train_iter, val_dataset, test_dataset = data.BucketIterator.splits((train_dataset, val_dataset, test_dataset), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True)

vocab_size = len(TEXT.vocab) #単語数
print(vocab_size) #単語数のサイズ

word_embeddings = TEXT.vocab.vectors #埋め込みベクトル
print(word_embeddings.size()) #埋め込みベクトルのサイズ

