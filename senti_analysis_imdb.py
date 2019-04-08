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

