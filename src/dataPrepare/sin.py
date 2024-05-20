import pandas as pd

from .dataset import Dataset
from pathlib import Path

base_path = Path(__file__).parent.parent.parent


class Sin(Dataset):

    def __init__(self, win_len, subset=None):
        super().__init__(name="Sin", file_name='')
        self.win_len = win_len
        self.subset = subset

    def _load(self, subset):
        train = pd.read_csv(f'{base_path}/dataset/{subset}/train.csv', header=None)
        test = pd.read_csv(f'{base_path}/dataset/{subset}/test.csv', header=None)

        x_train = train.iloc[:, :-1].to_numpy()
        y_train = train.iloc[:, -1].to_numpy().reshape(-1, 1)
        x_test = test.iloc[:, :-1].to_numpy()
        y_test = test.iloc[:, -1].to_numpy().reshape(-1, 1)

        y_test_binary = y_test

        self._data = tuple(data for data in [x_train, y_train, x_test, y_test, y_test_binary])


class Sin_swap_easy(Sin):

    def load(self):
        self._load('sin_swap_easy')


class Sin_swap_hard(Sin):

    def load(self):
        self._load('sin_swap_hard')


class Sin_ampl_easy(Sin):

    def load(self):
        self._load('sin_ampl_easy')


class Sin_ampl_hard(Sin):

    def load(self):
        self._load('sin_ampl_hard')


class Sin_incremental_easy(Sin):

    def load(self):
        self._load('sin_incremental_easy')


class Sin_gradual_easy(Sin):

    def load(self):
        self._load('sin_gradual_easy')


class Sin_incremental_hard(Sin):

    def load(self):
        self._load('sin_incremental_hard')


class Sin_gradual_hard(Sin):

    def load(self):
        self._load('sin_gradual_hard')
