import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .lstmae import EncDecAD
import logging
from tqdm import tqdm

from ts.timeseries import TimeSeriesDataset
import time

log = logging.getLogger()
torch.manual_seed(0)


class Update:

    def __init__(self, drift_idx, config, experiment_dir, data_path, update_train_size=28, update_val_size=7):
        self.drift_idx = drift_idx
        self.config = config
        self.experiment_dir = experiment_dir
        self.data_path = data_path
        self.update_train_size = update_train_size
        self.update_val_size = update_val_size
        self.hidden_dim = config['hidden_dim']
        self.batch_size = config['batch_size']
        self.dropout_rate = config['dropout_rate']
        self.weight_decay = config['weight_decay']
        self.num_layers = config['num_layers']
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.early_stopping_patience = config['early_stopping_patience']
        self.window_size = config['window_size']
        self.steps = config['steps']
        self.ks_alpha = config['ks_alpha']
        self.ks_hist_min_size = config['ks_hist_min_size']
        self.min_val_loss = None

    def _load_raw_data(self):
        df = pd.read_csv(self.data_path, header=None)
        train_dataframe = df[self.drift_idx - self.update_train_size // 2:self.drift_idx + self.update_train_size // 2]

        val_dataframe = df[
                        self.drift_idx + self.update_train_size // 2:self.drift_idx + self.update_val_size + self.update_train_size // 2]

        train_set = TimeSeriesDataset(train_dataframe, self.window_size, self.steps)
        validation_set = TimeSeriesDataset(val_dataframe, self.window_size, self.steps)
        self.input_size = train_set.__getitem__(0)[0].shape[1]
        return train_set, validation_set

    def _padding(self, batch):
        """

        @param batch:
        @type batch: shape (less_batch_size, win_len, input_dim)
        """

        padding_zeros = torch.zeros(self.batch_size - batch.shape[0], self.window_size, self.input_size)

        return torch.cat((batch, padding_zeros), dim=0)

    def _early_stopping(self, val_loss, model, patience=0):
        """

        Args:
            val_loss (): epoch average validation loss
            model (): intermediate model
            patience (): number of epochs without improvement before early stopping
        """
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
            # self._save_checkpoint(model)
        elif val_loss >= self.min_val_loss:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                return True
        else:
            self.min_val_loss = val_loss
            # self._save_checkpoint(model)
            self.early_stopping_counter = 0
        return False

    def train(self):
        if torch.cuda.is_available():
            print("Using CUDA device")

        train_set, validation_set = self._load_raw_data()

        model = EncDecAD(self.input_size, self.hidden_dim, self.batch_size, self.num_layers, self.dropout_rate)
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        print("Beginning training.")

        training_loss = []
        validation_loss = []
        train_data_loader = DataLoader(train_set, self.batch_size, shuffle=True, drop_last=False)
        validation_data_loader = DataLoader(validation_set, self.batch_size, drop_last=False)

        for epoch in range(1, self.epochs + 1):

            start_time = time.time()

            print(f'Epoch {epoch}/{self.epochs - 1}')

            val_input_buf = []
            val_hidden_buf = []
            val_error_buf = []
            val_reconstruction_buf = []  # for parameter learning

            model.train()
            running_train_loss = 0.0

            for i, batch in enumerate(tqdm(iter(train_data_loader))):
                optimizer.zero_grad()

                # Due to that the loss is only based in the original and reconstructed sequences, the labels are ignored
                # during the model training phase.
                batch_X, _ = batch[0], batch[1]
                current_batch_size = self.batch_size if batch_X.shape[0] == self.batch_size else batch_X.shape[0]
                if batch_X.shape[0] < self.batch_size:  # last batch could have less windows than batch_size
                    batch_X = self._padding(batch_X)
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()

                outputs, _ = model.forward(batch_X)
                outputs = outputs[:current_batch_size]
                batch_X = batch_X[:current_batch_size]
                """
                As documented in the original paper, let the decoder predict the sequence in reversed order can minimize 
                the gradient difference between last encoder step and first decoder step.
                """
                outputs = torch.flip(outputs, [1])
                loss = criterion(
                    batch_X,
                    outputs)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item() * batch_X.size(0)
            epoch_training_loss = running_train_loss / len(train_data_loader.dataset)
            training_loss.append(epoch_training_loss)

            model.eval()
            running_val_loss = 0.0

            for i, (batch_X, _, _) in enumerate(validation_data_loader):
                current_batch_size = self.batch_size if batch_X.shape[0] == self.batch_size else batch_X.shape[0]

                if batch_X.shape[0] < self.batch_size:  # last batch could have less windows than batch_size
                    batch_X = self._padding(batch_X)

                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                outputs, hidden_state = model.forward(batch_X)
                outputs = outputs[:current_batch_size]
                batch_X = batch_X[:current_batch_size]

                outputs = torch.flip(outputs, [1])
                if torch.cuda.is_available():
                    outputs = outputs.cuda()
                val_loss = criterion(batch_X, outputs)
                running_val_loss += val_loss.item() * batch_X.size(0)
                errors = torch.abs(batch_X - outputs)
                val_error_buf += errors.reshape(-1, self.input_size).tolist()
                val_hidden_buf += hidden_state[0][-1].tolist()
                val_input_buf += batch_X.reshape(-1, self.input_size).tolist()
                val_reconstruction_buf += outputs.reshape(-1, self.input_size).tolist()

            epoch_val_loss = running_val_loss / len(validation_data_loader.dataset)
            validation_loss.append(epoch_val_loss)

            used_time = time.time() - start_time
            print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(
                epoch, self.epochs, epoch_training_loss, epoch_val_loss, used_time))
            if self._early_stopping(epoch_val_loss, model, self.early_stopping_patience):
                print(f'Early stopping at epoch {str(epoch)}.')
                break

        f = plt.figure()
        plt.plot(np.linspace(1, epoch, epoch).astype(int), training_loss, color='blue')
        plt.plot(np.linspace(1, epoch, epoch).astype(int), validation_loss, color='red')
        f.clear()
        plt.close(f)
        mu = np.mean(val_error_buf, axis=0)

        if self.input_size == 1:
            sigma = np.var(val_error_buf)
            sigma = np.array([sigma])
        else:
            sigma = np.cov(val_error_buf, rowvar=False)

        return model, mu, sigma
