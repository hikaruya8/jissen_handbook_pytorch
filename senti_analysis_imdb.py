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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

#データのTensor形状の確認
for i, batch in enumerate(train_iter):
  print(batch.text[0].size())
  #ラベル
  print(batch.text[1].size())
  print(batch.label.size())
  print('1データ目の単語列を表示')
  print(batch.text[0][0])
  print(batch.text[1][0])
  print(batch.label[0])
  print([TEXT.vocab.itos[data] for data in batch.text[0][0].tolist()])
  print("ラベル")
  print(batch.label[0].item())
  break

class LstmClassifier(nn.Module):
  def __init__(self, batch_size, hidden_size, output_size, vocab_size, embedding_length, weights):
    super(LstmClassifier, self).__init__()
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.vocab_size = vocab_size
    self.embed = nn.Embedding(vocab_size, embedding_length)

    #use pre-trained Embedded Vector
    self.embed.weight.data.copy_(weights)
    self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    #初期隠れ状態とセル状態を設定
    h0 = torch.zeros(1, self.batch_size, self.hidden_size).to(device)
    c0 = torch.zeros(1, self.batch_size, self.hidden_size).to(device)

    #LSTMの伝播
    output_seq, (h_n, c_n) = self.lstm(x, (h0, c0)) # output_seqの出力形状：（バッチサイズ、シーケンス長、出力次元）
    out = self.fc(h_n[-1]) # 最後のタイムステップの隠れ状態をデコード
    return out

net = LstmClassifier(batch_size, hidden_size, output_size, vocab_size, embedding_length, word_embeddings)
net = net.to(device)

#損失関数、最適化関数
criterion = nn.CrossEntropyLoss()
optim = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))





