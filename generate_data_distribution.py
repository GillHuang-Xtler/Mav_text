from loguru import logger
import pathlib
import os
from federated_learning.arguments import Arguments
from federated_learning.datasets import CIFAR10Dataset
from federated_learning.datasets import AmazonDataset
from federated_learning.datasets import MNISTDataset
from federated_learning.datasets import CIFAR100Dataset
from federated_learning.datasets import FashionMNISTDataset
# from federated_learning.datasets import TRECDataset
from federated_learning.datasets import STL10Dataset
from federated_learning.utils import generate_train_loader
from federated_learning.utils import generate_test_loader
from federated_learning.utils import save_data_loader_to_file


if __name__ == '__main__':
    args = Arguments(logger)

    # ---------------------------------
    # ------------ CIFAR10 ------------
    # ---------------------------------
    # dataset = CIFAR10Dataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/test_data_loader.pickle"
    #
    # if not os.path.exists("data_loaders/cifar10"):
    #     pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)
    #
    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)
    #
    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)
    #
    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # --------- Fashion-MNIST ---------
    # ---------------------------------
    # dataset = FashionMNISTDataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/test_data_loader.pickle"
    #
    # if not os.path.exists("data_loaders/fashion-mnist"):
    #     pathlib.Path("data_loaders/fashion-mnist").mkdir(parents=True, exist_ok=True)
    #
    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)
    #
    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)
    #
    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # ------------ MNIST --------------
    # ---------------------------------
    # dataset = MNISTDataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/mnist/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/mnist/test_data_loader.pickle"
    #
    # if not os.path.exists("data_loaders/mnist"):
    #     pathlib.Path("data_loaders/mnist").mkdir(parents=True, exist_ok=True)
    #
    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)
    #
    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)
    #
    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # ------------ STL10 --------------
    # ---------------------------------
    # dataset = STL10Dataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/stl10/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/stl10/test_data_loader.pickle"
    #
    # if not os.path.exists("data_loaders/stl10"):
    #     pathlib.Path("data_loaders/stl10").mkdir(parents=True, exist_ok=True)
    #
    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)
    #
    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)
    #
    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # ------------ CIFAR100 -----------
    # ---------------------------------
    # dataset = CIFAR100Dataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar100/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar100/test_data_loader.pickle"
    #
    # if not os.path.exists("data_loaders/cifar100"):
    #     pathlib.Path("data_loaders/cifar100").mkdir(parents=True, exist_ok=True)
    #
    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)
    #
    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)
    #
    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # ---------------TREC -------------
    # ---------------------------------
    # dataset = TRECDataset(args)
    # TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/trec/train_data_loader.pickle"
    # TEST_DATA_LOADER_FILE_PATH = "data_loaders/trec/test_data_loader.pickle"
    #
    # if not os.path.exists("data_loaders/trec"):
    #     pathlib.Path("data_loaders/trec").mkdir(parents=True, exist_ok=True)
    #
    # train_data_loader = generate_train_loader(args, dataset)
    # test_data_loader = generate_test_loader(args, dataset)
    #
    # with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(train_data_loader, f)
    #
    # with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
    #     save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # ----Amazon Review Polarity-------
    # ---------------------------------
    dataset = AmazonDataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/Amazon/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/Amazon/test_data_loader.pickle"

    if not os.path.exists("data_loaders/Amazon"):
        pathlib.Path("data_loaders/Amazon").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)

