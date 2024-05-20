import pandas as pd

from .dataset import Dataset


class SMD(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="SMD", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        train = pd.read_csv('../dataset/SMD/1-1/train.csv', header=None)
        test_1 = pd.read_csv('../dataset/SMD/1-1/test.csv', header=None)
        test_2 = pd.read_csv('../dataset/SMD/1-2/test.csv', header=None)
        test_3 = pd.read_csv('../dataset/SMD/1-3/test.csv', header=None)
        test_label_1 = pd.read_csv('../dataset/SMD/1-1/test_label.csv', header=None)
        test_label_2 = pd.read_csv('../dataset/SMD/1-2/test_label.csv', header=None)
        test_label_3 = pd.read_csv('../dataset/SMD/1-3/test_label.csv', header=None)

        x_train = train.to_numpy()
        y_train = pd.Series([0 for _ in range(x_train.shape[0])]).to_numpy()
        x_test = pd.concat([test_1, test_2, test_1, test_3], axis=0).to_numpy()
        y_test = pd.concat([test_label_1, test_label_2, test_label_1, test_label_3], axis=0).to_numpy()

        y_test_binary = y_test

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])
