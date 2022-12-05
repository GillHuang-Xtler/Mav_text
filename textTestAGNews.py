# import torch
# import torch.nn as nn
# import torch.optim as optim
# import os
# import logging
# import pandas as pd
# from torchtext.data import Iterator, BucketIterator, TabularDataset
# from torchtext import data
# from torchtext.vocab import Vectors
#
# class FastText(nn.Module):
#     def __init__(self, vocab, vec_dim, label_size, hidden_size):
#         super(FastText, self).__init__()
#         #创建embedding
#         # self.vocab =
#         self.embed = nn.Embedding(len(vocab), vec_dim)
#         # 若使用预训练的词向量，需在此处指定预训练的权重
#         self.embed.weight.data.copy_(vocab.vectors)
#         self.embed.weight.requires_grad = True
#         self.fc = nn.Sequential(
#             nn.Linear(vec_dim, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, label_size)
#         )
#
#     def forward(self, x):
#         x = self.embed(x)
#         out = self.fc(torch.mean(x, dim=1))
#         return out
#
# def train_model(net, train_iter, epoch, lr, batch_size):
#     print("begin training")
#     net.train()
#     optimizer = optim.Adam(net.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     for i in range(epoch):
#         for batch_idx, batch in enumerate(train_iter):
#             # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
#             data, target = batch.text, batch.label - 1
#             optimizer.zero_grad()
#             output = net(data)
#             loss = criterion(output, target)
#             loss.backward()
#             optimizer.step()
#
#             # 打印状态信息
#             logging.info(
#                 "train epoch=" + str(i) + ",batch_id=" + str(batch_idx) + ",loss=" + str(loss.item() / batch_size))
#     print('Finished Training')
#
#
# def model_test(net, test_iter):
#     net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for i, batch in enumerate(test_iter):
#             # 注意target=batch.label - 1，因为数据集中的label是1，2，3，4，但是pytorch的label默认是从0开始，所以这里需要减1
#             data, label = batch.text, batch.label - 1
#             logging.info("test batch_id=" + str(i))
#             outputs = net(data)
#             # torch.max()[0]表示最大值的值，troch.max()[1]表示回最大值的每个索引
#             _, predicted = torch.max(outputs.data, 1)  # 每个output是一行n列的数据，取一行中最大的值
#             total += label.size(0)
#             correct += (predicted == label).sum().item()
#             print('Accuracy of the network on test set: %d %%' % (100 * correct / total))
#             # test_acc += accuracy_score(torch.argmax(outputs.data, dim=1), label)
#             # logging.info("test_acc=" + str(test_acc))
#
#
# def get_data_iter(train_csv, test_csv, fix_length):
#     TEXT = data.Field(sequential=True, lower=True, fix_length=fix_length, batch_first=True)
#     LABEL = data.Field(sequential=False, use_vocab=False)
#     train_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
#     train = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True)
#     train_iter = BucketIterator(train, batch_size=batch_size, device=-1, sort_key=lambda x: len(x.text),
#                                 sort_within_batch=False, repeat=False)
#     test_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
#     test = TabularDataset(path=test_csv, format="csv", fields=test_fields, skip_header=True)
#     test_iter = Iterator(test, batch_size=batch_size, device=-1, sort=False, sort_within_batch=False, repeat=False)
#
#     vectors = Vectors(name=word2vec_dir)
#     TEXT.build_vocab(train, vectors=vectors)
#     vocab = TEXT.vocab
#     return train_iter, test_iter, vocab
#
#
# if __name__ == "__main__":
#     logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
#     train_csv = "./data/AG_News/train.csv"
#     test_csv = "./data/AG_News/test.csv"
#     word2vec_dir = "./data/AG_News/glove.6B.300d.txt"  # 训练好的词向量文件,写成相对路径好像会报错
#     net_dir = "model/ag_fasttext_model.pkl"
#     sentence_max_size = 50  # 每篇文章的最大词数量
#     batch_size = 64
#     epoch = 10  # 迭代次数
#     emb_dim = 300  # 词向量维度
#     lr = 0.001
#     hidden_size = 200
#     label_size = 4
#
#     train_iter, test_iter, vocab = get_data_iter(train_csv, test_csv, sentence_max_size)
#     print(vocab)
#     net = FastText(vocab=vocab, vec_dim=emb_dim, label_size=label_size, hidden_size=hidden_size)
#
#     logging.info("start training")
#     train_model(net, train_iter, epoch, lr, batch_size)
#     torch.save(net, net_dir)
#     logging.info("start testing")
#     model_test(net, test_iter)
#

# -----------------------------------------------------
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torchtext import datasets
from torchtext import data
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

tokenizer = get_tokenizer('basic_english')
train_iter = datasets.AG_NEWS(split='train')
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: int(x) - 1


from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)


from torch import nn

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

import time

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count
# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_iter, test_iter = datasets.AG_NEWS()
train_dataset = list(train_iter)
# print(train_dataset)
# print(len(train_dataset))
test_dataset = list(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])
# print(split_train_)
# print(len(split_train_))
# 
train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)

valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


print(len(next(iter(train_dataloader))[0]))
print(len(next(iter(train_dataloader))[1]))
print(len(next(iter(train_dataloader))[2]))

# for epoch in range(1, EPOCHS + 1):
#     epoch_start_time = time.time()
#     train(train_dataloader)
#     accu_val = evaluate(valid_dataloader)
#     if total_accu is not None and total_accu > accu_val:
#       scheduler.step()
#     else:
#        total_accu = accu_val
#     print('-' * 59)
#     print('| end of epoch {:3d} | time: {:5.2f}s | '
#           'valid accuracy {:8.3f} '.format(epoch,
#                                            time.time() - epoch_start_time,
#                                            accu_val))
#     print('-' * 59)