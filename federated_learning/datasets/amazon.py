from .dataset import Dataset
from torchtext import datasets
from torch.utils.data import DataLoader

class AMAZONDataset(Dataset):

    def __init__(self, args):
        super(AMAZONDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Amazon train data")

        train_dataset = datasets.AmazonReviewPolarity(self.get_args().get_data_path(), train=True, download=True)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading Amazon train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading Amazon test data")

        test_dataset = datasets.AmazonReviewPolarity(self.get_args().get_data_path(), train=False, download=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading Amazon test data")

        return test_data
