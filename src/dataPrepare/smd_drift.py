import pandas as pd

from .dataset import Dataset
from pathlib import Path

base_path = Path(__file__).parent.parent.parent



class SMD_Small(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="SMD_Small", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        train = pd.read_csv(f'{base_path}/dataset/smd/1-1/train.csv', header=None)
        test_1 = pd.read_csv(f'{base_path}/dataset/smd/1-1/test.csv', header=None)
        test_2 = pd.read_csv(f'{base_path}/dataset/smd/1-2/test.csv', header=None)
        test_3 = pd.read_csv(f'{base_path}/dataset/smd/1-3/test.csv', header=None)
        test_label_1 = pd.read_csv(f'{base_path}/dataset/smd/1-1/test_label.csv', header=None)
        test_label_2 = pd.read_csv(f'{base_path}/dataset/smd/1-2/test_label.csv', header=None)
        test_label_3 = pd.read_csv(f'{base_path}/dataset/smd/1-3/test_label.csv', header=None)

        x_train = train.to_numpy()
        y_train = pd.Series([0 for _ in range(x_train.shape[0])]).to_numpy()
        x_test = pd.concat([test_1, test_2, test_1, test_3], axis=0).to_numpy()
        y_test = pd.concat([test_label_1, test_label_2, test_label_1, test_label_3], axis=0).to_numpy()

        y_test_binary = y_test

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])


class SMD_Large(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="SMD_Large", file_name='')
        self.win_len = win_len
        self.subset = subset

    def load(self):
        train = pd.read_csv(f'{base_path}/dataset/smd/1-1/train.csv', header=None)
        test_1 = pd.read_csv(f'{base_path}/dataset/smd/1-1/test.csv', header=None)
        test_2 = pd.read_csv(f'{base_path}/dataset/smd/1-2/test.csv', header=None)
        test_3 = pd.read_csv(f'{base_path}/dataset/smd/1-3/test.csv', header=None)
        test_4 = pd.read_csv(f'{base_path}/dataset/smd/1-4/test.csv', header=None)
        test_5 = pd.read_csv(f'{base_path}/dataset/smd/1-5/test.csv', header=None)
        test_6 = pd.read_csv(f'{base_path}/dataset/smd/1-6/test.csv', header=None)
        test_7 = pd.read_csv(f'{base_path}/dataset/smd/1-7/test.csv', header=None)
        test_8 = pd.read_csv(f'{base_path}/dataset/smd/1-8/test.csv', header=None)
        test_label_1 = pd.read_csv(f'{base_path}/dataset/smd/1-1/test_label.csv', header=None)
        test_label_2 = pd.read_csv(f'{base_path}/dataset/smd/1-2/test_label.csv', header=None)
        test_label_3 = pd.read_csv(f'{base_path}/dataset/smd/1-3/test_label.csv', header=None)
        test_label_4 = pd.read_csv(f'{base_path}/dataset/smd/1-4/test_label.csv', header=None)
        test_label_5 = pd.read_csv(f'{base_path}/dataset/smd/1-5/test_label.csv', header=None)
        test_label_6 = pd.read_csv(f'{base_path}/dataset/smd/1-6/test_label.csv', header=None)
        test_label_7 = pd.read_csv(f'{base_path}/dataset/smd/1-7/test_label.csv', header=None)
        test_label_8 = pd.read_csv(f'{base_path}/dataset/smd/1-8/test_label.csv', header=None)

        x_train = train.to_numpy()
        y_train = pd.Series([0 for _ in range(x_train.shape[0])]).to_numpy()
        x_test = pd.concat([test_1, test_2, test_1, test_3, test_1, test_4, test_1, test_5, test_1, test_6, test_1, test_7, test_1, test_8],
                           axis=0).to_numpy()
        y_test = pd.concat([test_label_1, test_label_2, test_label_1, test_label_3,
                            test_label_1, test_label_4, test_label_1, test_label_5,
                            test_label_1, test_label_6, test_label_1, test_label_7,
                            test_label_1, test_label_8], axis=0).to_numpy()

        y_test_binary = y_test

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])
