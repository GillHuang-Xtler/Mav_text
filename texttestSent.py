import spacy
import torch
from torchtext import data, datasets
from torchtext.vocab import Vectors
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import spacy.cli

# spacy.cli.download("en_core_web_md")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_data = pd.read_csv('./data/Sentiment/train.tsv', sep='\t')
raw_test = pd.read_csv('./data/Sentiment/test.tsv', sep='\t')

# create train and validation set
train, val = train_test_split(raw_data, test_size=0.2)
train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)

spacy_en = spacy.load('en')


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


# Field
LABEL = data.Field(sequential=False, use_vocab=False)
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True)

# Dataset
train, val = data.TabularDataset.splits(
    path='.', train='train.csv', validation='val.csv', format='csv', skip_header=True,
    fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT), ('Sentiment', LABEL)])

test = data.TabularDataset('test.tsv', format='tsv', skip_header=True,
                           fields=[('PhraseId', None), ('SentenceId', None), ('Phrase', TEXT)])
# build vocab
TEXT.build_vocab(train, vectors='glove.6B.100d')  # , max_size=30000)
TEXT.vocab.vectors.unk_init = init.xavier_uniform

# Iterator
train_iter = data.BucketIterator(train, batch_size=128, sort_key=lambda x: len(x.Phrase),
                                 shuffle=True, device=DEVICE)

val_iter = data.BucketIterator(val, batch_size=128, sort_key=lambda x: len(x.Phrase),
                               shuffle=True, device=DEVICE)

# 在 test_iter , sort一定要设置成 False, 要不然会被 torchtext 搞乱样本顺序
test_iter = data.Iterator(dataset=test, batch_size=128, train=False,
                          sort=False, device=DEVICE)
"""
由于目的是学习torchtext的使用，所以只定义了一个简单模型
"""
len_vocab = len(TEXT.vocab)


class Enet(nn.Module):
    def __init__(self):
        super(Enet, self).__init__()
        self.embedding = nn.Embedding(len_vocab, 100)
        self.lstm = nn.LSTM(100, 128, 3, batch_first=True)  # ,bidirectional=True)
        self.linear = nn.Linear(128, 5)

    def forward(self, x):
        batch_size, seq_num = x.shape
        vec = self.embedding(x)
        out, (hn, cn) = self.lstm(vec)
        out = self.linear(out[:, -1, :])
        out = F.softmax(out, -1)
        return out


model = Enet()
"""
将前面生成的词向量矩阵拷贝到模型的embedding层
这样就自动的可以将输入的word index转为词向量
"""
model.embedding.weight.data.copy_(TEXT.vocab.vectors)
model.to(DEVICE)

# 训练
optimizer = optim.Adam(model.parameters())  # ,lr=0.000001)

n_epoch = 5

best_val_acc = 0

for epoch in range(n_epoch):

    for batch_idx, batch in enumerate(train_iter):
        data = batch.Phrase
        target = batch.Sentiment
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1, 0)
        optimizer.zero_grad()

        out = model(data)
        loss = -target * torch.log(out) - (1 - target) * torch.log(1 - out)
        loss = loss.sum(-1).mean()

        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 200 == 0:
            _, y_pre = torch.max(out, -1)
            acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
            print('epoch: %d \t batch_idx : %d \t loss: %.4f \t train acc: %.4f'
                  % (epoch, batch_idx, loss, acc))

    val_accs = []
    for batch_idx, batch in enumerate(val_iter):
        data = batch.Phrase
        target = batch.Sentiment
        target = torch.sparse.torch.eye(5).index_select(dim=0, index=target.cpu().data)
        target = target.to(DEVICE)
        data = data.permute(1, 0)
        out = model(data)

        _, y_pre = torch.max(out, -1)
        acc = torch.mean((torch.tensor(y_pre == batch.Sentiment, dtype=torch.float)))
        val_accs.append(acc)

    acc = np.array(val_accs).mean()
    if acc > best_val_acc:
        print('val acc : %.4f > %.4f saving model' % (acc, best_val_acc))
        torch.save(model.state_dict(), 'params.pkl')
        best_val_acc = acc
    print('val acc: %.4f' % (acc))