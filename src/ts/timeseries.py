import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    A customized dataset for time series. Sliding window with window_size
    will be applied to the dataset.
    """

    def __init__(self, dataframe, window_size, step):
        """

        Args:
            data_path: path to time series csv file
            window_size: size of sliding window
            step: number of steps sliding goes forward each time
        """
        self.data = dataframe
        self.window_size = window_size
        self.step = step

    def __getitem__(self, index):
        """

        Args:
            index: the index number of window

        Returns: the dataset and label in required sliding window

        """
        X = self.data.iloc[self.step * index: self.step * index + self.window_size, : -1]
        y = self.data.iloc[self.step * index: self.step * index + self.window_size, -1]

        timestamp = torch.FloatTensor(X.index.tolist())
        X = torch.FloatTensor(X.values)
        y = torch.LongTensor(y.values)


        return X, y, timestamp

    def __len__(self):
        """

        Returns: number of sliding windows

        """
        count = 0
        dataset_size = self.data.shape[0]
        while dataset_size >= self.window_size:
            dataset_size -= self.step
            count += 1
        return count

    def _scaling(self, df):
        label = df.iloc[:, -1]
        scaler = MinMaxScaler()
        scaler.fit(df.iloc[:, :-1])
        cont = pd.DataFrame(scaler.transform(df.iloc[:, :-1]))
        data = pd.concat((cont, label), axis=1)
        return data
