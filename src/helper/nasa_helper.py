import os
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from autoperiod import Autoperiod


class NasaHelper:

    def __init__(self, data_dir, output_dir, label_file_path, default_win_len, spacecraft='SMAP'):
        self.dataDir = data_dir
        self.outputDir = output_dir
        self.trainDir = os.path.join(data_dir, 'train')
        self.testDir = os.path.join(data_dir, 'test')
        self.labelFilePath = label_file_path
        self.spacecraft = spacecraft
        self.default_win_len = default_win_len

    def _load_label_df(self):
        self.label_df = pd.read_csv(self.labelFilePath)

    def _load_train_dfs(self):
        train_df_dict = {}

        for _, _, files in os.walk(self.trainDir):
            for file in files:
                channel = file.split('.')[0]
                if self.label_df[self.label_df.chan_id == channel].spacecraft.values[0] != self.spacecraft:
                    continue
                path = os.path.join(self.trainDir, file)
                npy = np.load(path)
                df = pd.DataFrame(npy)
                label = pd.Series(['normal' for _ in range(df.shape[0])])
                train_df_dict[channel] = pd.concat((df.iloc[:, 0], label), axis=1)

        return train_df_dict

    def _load_test_dfs(self):
        test_df_dict = {}
        for _, _, files in os.walk(self.testDir):
            for file in files:
                channel = file.split('.')[0]
                if self.label_df[self.label_df.chan_id == channel].spacecraft.values[0] != self.spacecraft:
                    continue
                path = os.path.join(self.testDir, file)
                npy = np.load(path)
                df = pd.DataFrame(npy)

                label = ['normal' for _ in range(df.shape[0])]
                if channel in self.label_df.chan_id.values:
                    anomaly_ranges = ast.literal_eval(
                        self.label_df[self.label_df.chan_id == channel].anomaly_sequences.values[0])
                    for idx, anomaly_range in enumerate(anomaly_ranges):
                        assert len(label) >= anomaly_range[1], (len(label), anomaly_range)
                        label[anomaly_range[0]: anomaly_range[1]] = [
                            'anomaly'
                            for _ in range(anomaly_range[1] - anomaly_range[0])]
                label = pd.Series(label)
                df = pd.concat((df.iloc[:, 0], label), axis=1)

                test_df_dict[channel] = df
        return test_df_dict

    def _get_period(self, dataset, timeseries, max_len=5000):
        max_len = timeseries.size if timeseries.size < max_len else max_len
        p = Autoperiod(np.array(range(timeseries.size)), timeseries[:max_len], plotter=None)
        win_len = p.period if (p.period is not None) and (p.period <= self.default_win_len) else self.default_win_len
        return win_len

    def load(self):
        """
        Label and load train and test data
        """

        self._load_label_df()
        train_df_dict = self._load_train_dfs()
        test_df_dict = self._load_test_dfs()

        split_pos_log = []
        for dataset in train_df_dict.keys():
            period = self._get_period(dataset, train_df_dict[dataset].iloc[:, 0])
            df = pd.concat((train_df_dict[dataset], test_df_dict[dataset]), axis=0)
            df.to_csv(os.path.join(self.outputDir, dataset + '.csv'), header=False, index=False)
            split_pos_log.append([dataset, train_df_dict[dataset].shape[0], period])

        pd.DataFrame(split_pos_log, columns=['dataset', 'position', 'win_len']) \
            .to_csv(os.path.join(Path(self.outputDir).resolve().parent, 'nasa_log.csv'), index=False)
