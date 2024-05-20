import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .lstmae import EncDecAD
import logging
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler

from ts.timeseries import TimeSeriesDataset
import time
from torch.autograd import Variable

log = logging.getLogger()
log.setLevel("INFO")


def device(gpu):
    return torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


class Train:

    def __init__(self, experiment_dir, hidden_dim, batch_size, dropout_rate, epochs, early_stopping_patience,
                 learning_rate, weight_decay, window_size, input_dim, is_initialization, param_path=None,
                 model_path=None, gpu=None):

        self.input_size = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.window_size = window_size
        self.steps = window_size
        self.num_layers = 1  # we use one layer for both encoder and decoder
        self.val_percentage = 0.2
        self.is_initialization = is_initialization
        self.experiment_dir = experiment_dir
        self.early_stopping_counter = 0
        self.min_val_loss = None
        self.param_path = param_path
        self.model_path = model_path
        self.gpu = gpu

        logging.basicConfig(filename=os.path.join(experiment_dir, 'log'), level=logging.INFO)

    def _load_model(self, existing_model):
        model = EncDecAD(self.input_size, self.hidden_dim, self.batch_size, self.num_layers, self.dropout_rate)
        if existing_model is not None:
            model.load_state_dict(torch.load(existing_model))
            model.eval()

        return model

    def _load_buffer_data(self):
        """
        The buffer data is already applied sliding window. So the loading should be straightforward (without overlap).
        """
        train_dataframe = pd.read_csv(self.train_data_path, header=None)
        val_dataframe = pd.read_csv(self.vn1_path, header=None)
        train_set = TimeSeriesDataset(train_dataframe, self.window_size, self.window_size)
        validation_set = TimeSeriesDataset(val_dataframe, self.window_size, self.window_size)

        return train_set, validation_set

    def _save_checkpoint(self, model):
        model_output_path = os.path.join(self.experiment_dir, "checkpoint.th")
        torch.save(model.state_dict(), model_output_path)

    def _early_stopping(self, val_loss, model, patience=0):
        """

        Args:
            val_loss (): epoch average validation loss
            model (): intermediate model
            patience (): number of epochs without improvement before early stopping
        """
        if self.min_val_loss is None:
            self.min_val_loss = val_loss
            self._save_checkpoint(model)
        elif val_loss >= self.min_val_loss:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                return True
        else:
            self.min_val_loss = val_loss
            self._save_checkpoint(model)
            self.early_stopping_counter = 0
        return False

    def _padding(self, batch):
        """

        @param batch:
        @type batch: shape (less_batch_size, win_len, input_dim)
        """

        padding_zeros = torch.zeros(self.batch_size - batch.shape[0], self.window_size, self.input_size)

        return torch.cat((batch, padding_zeros), dim=0)

    def to_var(self, t, **kwargs):
        t = t.to(device(self.gpu))
        return Variable(t, **kwargs)

    def to_device(self, model):
        model.to(device(self.gpu))

    def train(self, train_set, existing_model=None):
        if torch.cuda.is_available():
            print("Using CUDA device")

        model = self._load_model(existing_model)
        self.to_device(model)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        print("Beginning training.")

        training_loss = []
        validation_loss = []

        sequences = [train_set[i:i + self.window_size] for i in
                     range(0, train_set.shape[0] - self.window_size + 1, self.steps)]
        indices = np.random.permutation(len(sequences))
        split_point = int(self.val_percentage * len(sequences))

        train_data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                       sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
        validation_data_loader = DataLoader(dataset=sequences, batch_size=self.batch_size, drop_last=True,
                                            sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

        for epoch in range(1, self.epochs + 1):

            start_time = time.time()

            print(f'Epoch {epoch}/{self.epochs - 1}')
            train_error_buf = []
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
                batch_X = batch.float()
                current_batch_size = self.batch_size if batch_X.shape[0] == self.batch_size else batch_X.shape[0]
                if batch_X.shape[0] < self.batch_size:  # last batch could have less windows than batch_size
                    batch_X = self._padding(batch_X)
                batch_X = self.to_var(batch_X)
                outputs, _ = model.forward(batch_X, gpu=self.gpu)
                outputs = outputs[:current_batch_size]
                batch_X = batch_X[:current_batch_size]
                """
                As documented in the original paper, let the decoder predict the sequence in reversed order can minimize 
                the gradient difference between last encoder step and first decoder step.
                """
                outputs = torch.flip(outputs, [1])
                errors = torch.abs(batch_X - outputs)
                train_error_buf += errors.reshape(-1, self.input_size).tolist()

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

            for i, batch in enumerate(validation_data_loader):
                batch_X = batch.float()
                current_batch_size = self.batch_size if batch_X.shape[0] == self.batch_size else batch_X.shape[0]

                if batch_X.shape[0] < self.batch_size:  # last batch could have less windows than batch_size
                    batch_X = self._padding(batch_X)

                batch_X = self.to_var(batch_X)

                outputs, hidden_state = model.forward(batch_X, gpu=self.gpu)
                outputs = outputs[:current_batch_size]
                batch_X = batch_X[:current_batch_size]

                outputs = torch.flip(outputs, [1])
                outputs = self.to_var(outputs)

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
        plt.plot(np.linspace(1, epoch, epoch).astype(int), training_loss, color='blue', label='Train')
        plt.plot(np.linspace(1, epoch, epoch).astype(int), validation_loss, color='red', label='Val')
        plt.legend()
        plt.savefig(os.path.join(self.experiment_dir, 'loss.png'))
        f.clear()
        plt.close(f)

        mu = np.mean(train_error_buf, axis=0)
        if not os.path.exists(os.path.join(self.experiment_dir, "parameters")):
            os.mkdir(os.path.join(self.experiment_dir, "parameters"))

        if self.input_size == 1:
            sigma = np.var(train_error_buf)
            sigma = np.array([sigma])
        else:
            sigma = np.cov(train_error_buf, rowvar=False)

        if not os.path.exists(os.path.join(self.experiment_dir, "parameters")):
            os.mkdir(os.path.join(self.experiment_dir, "parameters"))
        pickle.dump(mu, open(os.path.join(self.experiment_dir, "parameters", 'mu.pkl'), 'wb'))
        pickle.dump(sigma, open(os.path.join(self.experiment_dir, "parameters", 'sigma.pkl'), 'wb'))

        # Load model from last checkpoint
        model_checkpoint_path = os.path.join(self.experiment_dir, "checkpoint.th")
        model.load_state_dict(torch.load(model_checkpoint_path))

        model_output_path = os.path.join(self.experiment_dir, "model.th")
        torch.save(model.state_dict(), model_output_path)

        return model, mu, sigma
