import pandas as pd

from .dataset import Dataset

class Forest(Dataset):
    def __init__(self, win_len):
        super().__init__(name="Forest", file_name='')
        self.win_len = win_len

    def _load_csv(self):
        return pd.read_csv(f'../dataset/forest/covtypeNorm.csv')

    def load(self):
        data = self._load_csv()

        x_train, x_test = data.iloc[:10000, :-1], data.iloc[10000:,
                                                             :-1].to_numpy()  # based on Table 2 in the original paper
        y_train_tmp, y_test_tmp = data.iloc[:10000, -1], data.iloc[10000:, -1]
        y_train = pd.Series([0 if y != 4 else 1 for y in y_train_tmp])
        y_test = pd.Series([0 if y != 4 else 1 for y in y_test_tmp]).to_numpy().reshape(-1, 1)
        y_test_binary = y_test

        x_train = x_train[y_train != 1].to_numpy()
        y_train = y_train[y_train != 1].to_numpy().reshape(-1, 1)

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])
