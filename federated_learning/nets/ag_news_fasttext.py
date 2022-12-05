import torch
import torch.nn as nn


class AGNewsFastText(nn.Module):
    # def __init__(self, vocab, vec_dim = 300, label_size = 4, hidden_size = 200):
    #     super(AGNewsFastText, self).__init__()
    #
    #     self.embed = nn.Embedding(len(vocab), vec_dim)
    #     self.embed.weight.data.copy_(vocab.vectors)
    #     self.embed.weight.requires_grad = True
    #     self.fc = nn.Sequential(
    #         nn.Linear(vec_dim, hidden_size),
    #         nn.BatchNorm1d(hidden_size),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(hidden_size, label_size)
    #     )
    #
    # def forward(self, x):
    #     x = self.embed(x)
    #     out = self.fc(torch.mean(x, dim=1))
    #     return out
    def __init__(self, vocab_size, embed_dim, num_class):
        super(AGNewsFastText, self).__init__()
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