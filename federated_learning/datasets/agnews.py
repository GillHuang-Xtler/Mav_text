from .dataset import Dataset
from torchtext import datasets
from torchtext import data
import torch
from torch.utils.data import DataLoader
# from torchtext.data import Iterator, BucketIterator, TabularDataset
from torchtext.vocab import Vectors
from torch.utils.data.dataset import random_split

class AGNewsDataset(Dataset):

    def __init__(self, args):
        super(AGNewsDataset, self).__init__(args)

    # def load_train_dataset(self):
    #     self.get_args().get_logger().debug("Loading AGNEWS train data")
    #
    #     train_csv = "./data/AG_News/train.csv"
    #     word2vec_dir = "./data/AG_News/glove.6B.300d.txt"
    #
    #     TEXT = data.Field(sequential=True, lower=True, fix_length=50, batch_first=True)
    #     LABEL = data.Field(sequential=False, use_vocab=False)
    #     train_fields = [("label", LABEL), ("title", None), ("text", TEXT)]
    #     train_dataset = TabularDataset(path=train_csv, format="csv", fields=train_fields, skip_header=True) # dataset object, consists of dicts {label: x, data: xxx}
    #     vectors = Vectors(name=word2vec_dir)
    #     TEXT.build_vocab(train_dataset, vectors=vectors)
    #
    #
    #
    #
    #     train_iter = BucketIterator(train_dataset, batch_size=len(train_dataset), device=-1, sort_key=lambda x: len(x.text),
    #                                 sort_within_batch=False, repeat=False)
    #     train_iter.create_batches()
    #     print('PyTorchText BuketIterator\n')
    #     for batch in train_iter.batches:
    #
    #         # Let's check batch size.
    #         print('Batch size: %d\n' % len(batch))
    #         print('LABEL\tLENGTH\tTEXT'.ljust(10))
    #
    #         # Print each example.
    #         for example in batch:
    #             print('%s\t%d\t%s'.ljust(10) % (example.label, len(example.text), example.text))
    #         print('\n')
    #
    #         # Only look at first batch. Reuse this code in training models.
    #         break
    #
    #     train_loader = DataLoader(train_iter, batch_size=len(train_dataset))
    #     print(next(iter(train_loader)))
    #     # print(next(iter(train_iter))[0].shape, next(iter(train_iter))[1].shape)
    #
    #     train_data = self.get_tuple_from_data_loader(train_loader)
    #
    #     self.get_args().get_logger().debug("Finished loading AGNEWS train data")
    #
    #     return train_data

    from torch.utils.data import DataLoader


    def collate_batch(self, batch):
        from torchtext.data.utils import get_tokenizer
        from collections import Counter
        from torchtext.vocab import Vocab

        tokenizer = get_tokenizer('basic_english')
        train_iter = datasets.AG_NEWS(split='train')
        counter = Counter()
        for (label, line) in train_iter:
            counter.update(tokenizer(line))
        vocab = Vocab(counter)
        # print("vocab size :" + str(len(vocab)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
        label_pipeline = lambda x: int(x) - 1
        label_list, text_list, offsets = [], [], [0]
        for (_text, _label) in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list.to(device), label_list.to(device), offsets.to(device)

    def load_train_dataset(self):
        train_iter, test_iter = datasets.AG_NEWS()
        # from torchtext.data import to_map_style_dataset
        # train_dataset = to_map_style_dataset(train_iter)
        train_dataset = list(train_iter)
        for i in range(len(train_dataset)):
            _label = train_dataset[i][0]
            _data = train_dataset[i][1]
            train_dataset[i] = (_data,_label)
        # print(type(train_dataset[0]))
        # print(len(train_dataset))
        # train_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
        #                               shuffle=True, collate_fn=self.collate_batch)
        # train_data = self.get_tuple_from_textdata_loader(train_loader)
        # print(train_data[0])
        # print(train_data[1])
        # print(train_data[2])
        # print(len(train_data[0]))
        # print(len(train_data[1]))
        # print(len(train_data[2]))
        train_data_loader = DataLoader(train_dataset, batch_size=64,
                                      shuffle=True, collate_fn=self.collate_batch)
        self.get_args().get_logger().debug("Finished loading AGNEWS train data")
        # print((next(iter(train_data_loader))[0]))
        # print((next(iter(train_data_loader))[1]))
        # print((next(iter(train_data_loader))[2]))
        # print(next(iter(train_data_loader)))
        return train_data_loader

    def load_small_train_dataset(self):
        train_iter, test_iter = datasets.AG_NEWS()

        train_dataset = list(train_iter)[:5000]
        for i in range(5000):
            _label = train_dataset[i][0]
            _data = train_dataset[i][1]
            train_dataset[i] = (_data,_label)

        train_data_loader = DataLoader(train_dataset, batch_size=64,
                                      shuffle=True, collate_fn=self.collate_batch)
        self.get_args().get_logger().debug("Finished loading AGNEWS train data")
        print((next(iter(train_data_loader))[0]))
        print((next(iter(train_data_loader))[1]))
        print((next(iter(train_data_loader))[2]))
        return train_data_loader



    def load_test_dataset(self):
        train_iter, test_iter = datasets.AG_NEWS()
        test_dataset = list(test_iter)
        for i in range(len(test_dataset)):
            _label = test_dataset[i][0]
            _data = test_dataset[i][1]
            test_dataset[i] = (_data,_label)
        test_data_loader = DataLoader(test_dataset, batch_size=64,
                                      shuffle=True, collate_fn=self.collate_batch)
        self.get_args().get_logger().debug("Finished loading AGNEWS test data")
        return test_data_loader
