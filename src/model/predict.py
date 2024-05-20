import json
import os
import pickle
import time
from torch.utils.data import DataLoader
from sklearn import metrics
from .lstmae import EncDecAD
import torch
import logging
from .train import Train
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from scipy.special import softmax
from torch.autograd import Variable
from bisect import bisect
from .drift_detection import KSTest, KLDivergence

log = logging.getLogger()
log.setLevel("INFO")


def get_device(gpu):
    return torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')


class Predict:

    def __init__(self, experiment_dir, param_path, hidden_dim, batch_size, dropout_rate, epochs,
                 early_stopping_patience,
                 learning_rate, weight_decay, window_size, input_dim, l_hist_size, l_new_size, ks_alpha,
                 ks_hist_min_size, continue_training=False, gpu=None):
        self.input_size = input_dim
        self.experiment_dir = experiment_dir
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_layers = 1
        self.window_size = window_size
        self.steps = window_size
        self.l_hist_size = l_hist_size
        self.l_new_size = l_new_size
        self.ks_alpha = ks_alpha
        self.ks_hist_min_size = ks_hist_min_size
        self.mu = pickle.load(open(os.path.join(param_path, 'mu.pkl'), 'rb'))
        self.sigma = pickle.load(open(os.path.join(param_path, 'sigma.pkl'), 'rb'))
        self.continue_training = continue_training
        self.gpu = gpu

        logging.basicConfig(filename=os.path.join(experiment_dir, 'log'), level=logging.INFO)

    def _load_model(self, model_path):
        model = EncDecAD(self.input_size, self.hidden_dim, self.batch_size, self.num_layers, self.dropout_rate)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def _calc_anomaly_score(self, reconstruction_errors):
        anomaly_scores = []
        reconstruction_errors = reconstruction_errors.reshape(-1, self.input_size).tolist()

        for instance in reconstruction_errors:

            if self.input_size > 1:
                delta = np.array(instance) - self.mu
                score = np.dot(np.dot(delta, sp.linalg.inv(self.sigma)), delta.T)
            else:
                score = np.log(1 / (scipy.stats.norm(self.mu, self.sigma).pdf(instance[0]) + 0.0000001))[0]
            anomaly_scores.append(score)

        return anomaly_scores

    def _padding(self, batch):
        """

        @param batch:
        @type batch: shape (less_batch_size, win_len, input_dim)
        """

        padding_zeros = torch.zeros(self.batch_size - batch.shape[0], self.window_size, self.input_size)

        return torch.cat((batch, padding_zeros), dim=0)

    def _persist_model(self, phase_path, model):

        if not os.path.exists(os.path.join(self.experiment_dir, phase_path)):
            os.mkdir(os.path.join(self.experiment_dir, phase_path))
        if not os.path.exists(os.path.join(self.experiment_dir, phase_path, "parameters")):
            os.mkdir(os.path.join(self.experiment_dir, phase_path, "parameters"))

        model_output_path = os.path.join(self.experiment_dir, phase_path, "checkpoint.th")
        torch.save(model.state_dict(), model_output_path)

        pickle.dump(self.mu, open(os.path.join(self.experiment_dir, phase_path, "parameters", 'mu.pkl'), 'wb'))
        pickle.dump(self.sigma, open(os.path.join(self.experiment_dir, phase_path, "parameters", 'sigma.pkl'), 'wb'))

    def _reload_model(self, model, phase_path):
        model_checkpoint_path = os.path.join(self.experiment_dir, phase_path, "checkpoint.th")
        model.load_state_dict(torch.load(model_checkpoint_path))
        self.mu = pickle.load(open(os.path.join(self.experiment_dir, phase_path, "parameters", 'mu.pkl'), 'rb'))
        self.sigma = pickle.load(open(os.path.join(self.experiment_dir, phase_path, "parameters", 'sigma.pkl'), 'rb'))

        return model

    def _update_func(self, update_set, exp_dir, existing_model=None):
        is_initialization = True
        trainer = Train(exp_dir, self.hidden_dim, self.batch_size, self.dropout_rate, self.epochs,
                        self.early_stopping_patience, self.learning_rate, self.weight_decay, self.window_size,
                        self.input_size, is_initialization, gpu=self.gpu)
        model, mu, sigma = trainer.train(update_set, existing_model)

        return model, mu, sigma
    def to_var(self, t, **kwargs):
        t = t.to(get_device(self.gpu))
        return Variable(t, **kwargs)

    def predict(self, test_set, test_label, existing_model, kld_epsilon):
        """

        Returns:
            prediction: predicted anomaly scores (1-D list)
            labels: ground truth (1-D list)
            error_buf: reconstruction errors
        """
        if torch.cuda.is_available():
            device = get_device(self.gpu)
        else:
            device = torch.device("cpu")

        ks = KSTest(self.ks_alpha, self.l_hist_size, self.l_new_size, self.ks_hist_min_size)
        kld = KLDivergence()

        model = self._load_model(existing_model)
        # reference_model: a copy of the initially trained model. will not be updated. Only used for embbedding
        # generation for the KLD comparison
        reference_model = self._load_model(existing_model)

        model.to(device)
        reference_model.to(device)

        test_data = np.concatenate((test_set, test_label), axis=1)
        sequences = [test_data[i:i + self.window_size] for i in
                     range(0, test_data.shape[0] - self.window_size + 1, self.steps)]
        test_data_loader = DataLoader(dataset=sequences, shuffle=False, batch_size=self.batch_size, drop_last=True,
                                      pin_memory=True)

        input_buf = []
        output_buf = []
        error_buf = []
        score_buf = []
        hidden_state_buf = []
        reference_hidden_state_buf = []

        tmp_inputs_buf = []  # the latest input samples. Same length as l_new * win_len. if update triggered, the samples in this buffer will be used to train a new model
        tmp_latent_rep_buf = []
        state_buf = dict()
        phase = 0
        current_state = 0
        log_text = ''
        time_log = ''
        model_reuse_dict = dict()

        # store initial model to phase 0 folder
        phase_path = f'phase_{phase}'
        exp_phase_path = os.path.join(self.experiment_dir, phase_path)
        if not os.path.exists(exp_phase_path):
            os.makedirs(exp_phase_path)
        self._persist_model(phase_path, model)
        for idx, batch in enumerate(test_data_loader):

            batch_X = batch[:, :, :-1]
            batch_y = batch[:, :, -1]

            current_batch_size = self.batch_size if batch_X.shape[0] == self.batch_size else batch_X.shape[0]
            if batch_X.shape[0] < self.batch_size:  # last batch could have less windows than batch_size
                batch_X = self._padding(batch_X)

            batch_X = batch_X.float()
            batch_y = batch_y.float()

            batch_X = self.to_var(batch_X)

            '''
            batch_X shape: (batch_size, window_length, input_size)
            outputs shape: (batch_size, window_length, input_size)
            hidden_state[0] shape: (num_layers, batch_size, hidden_dim)
            '''
            outputs, hidden_state = model.forward(batch_X, gpu=self.gpu)
            reference_outputs, reference_hidden_state = reference_model.forward(batch_X, gpu=self.gpu)
            latent_representation = hidden_state[0][-1]  # only last encoder layer. buf size: (N*hidden_size)
            reference_latent_representation = reference_hidden_state[0][-1]
            log_text += f'Info: processing batch {idx}. Lhist={len(ks.l_hist)}, Lnew={len(ks.l_new)}.\n'

            # ks-test for concept drift detection
            for window in range(self.batch_size):
                reuse = False
                is_drift, result_p_values = ks.detect_drift(latent_representation[window])

                if len(result_p_values) != 0:
                    num_small_p = np.array(result_p_values)
                    num_small_p = min(num_small_p)
                    log_text += f'smallest p_values: {num_small_p}\n'
                if is_drift:

                    if self.continue_training:  # a baseline: when a drift is detected, the latest data is used to update the previous autoencoder

                        phase += 1

                        phase_path = f'phase_{phase}'
                        exp_phase_path = os.path.join(self.experiment_dir, phase_path)
                        if not os.path.exists(exp_phase_path):
                            os.makedirs(exp_phase_path)

                        start_update_time = time.time()
                        update_set = np.array(tmp_inputs_buf)
                        self._persist_model(phase_path, model)
                        model_path = os.path.join(self.experiment_dir, phase_path, 'checkpoint.th')
                        model, mu, sigma = self._update_func(update_set, exp_phase_path, model_path)
                        self.mu = mu
                        self.sigma = sigma

                        end_update_time = time.time()

                        time_log += 'Continuously update: ' + str(round(end_update_time - start_update_time, 3)) + '\n'

                    else:
                        # new window size batch_size * hidden_size (dataframe)
                        log_text += f'Alarm: Detected concept drift at batch {idx}, window {window}\n'

                        histogram_hist = softmax(pd.DataFrame(tmp_latent_rep_buf[:self.batch_size]), axis=1).mean(
                            axis=0).tolist()
                        if phase == 0:  # add the first state in the state buf
                            state_buf[0] = histogram_hist

                        histogram_new = softmax(pd.DataFrame(reference_latent_representation.tolist()), axis=1).mean(
                            axis=0).tolist()
                        tmp_latent_rep_buf = []

                        kld_result = dict()
                        for key in state_buf.keys():
                            if key == current_state:
                                continue
                            current_kld = kld.get_kld(state_buf[key], histogram_new)
                            kld_result[key] = current_kld  # kld distance to every existing state
                        log_text += f'kld_result {kld_result}\n'
                        if len(kld_result) != 0:
                            min_key = min(kld_result, key=kld_result.get)
                            print(kld_result[min_key])
                            if kld_result[min_key] < kld_epsilon:
                                current_state = min_key

                                model = self._reload_model(model, f'phase_{min_key}') if min_key not in model_reuse_dict \
                                    .keys() else self._reload_model(model, f'phase_{model_reuse_dict[min_key]}')
                                print(f'Switch to phase {min_key}')
                                log_text += f'Switch to phase {min_key}, reload model from phase_{min_key}\n'
                                if min_key not in model_reuse_dict.keys():
                                    model_reuse_dict[phase + 1] = min_key
                                else:
                                    tmp_key = min_key
                                    while tmp_key in model_reuse_dict.keys():
                                        tmp_key = model_reuse_dict[tmp_key]
                                    model_reuse_dict[phase + 1] = tmp_key
                                reuse = True

                        phase += 1

                        phase_path = f'phase_{phase}'
                        exp_phase_path = os.path.join(self.experiment_dir, phase_path)
                        if not os.path.exists(exp_phase_path):
                            os.makedirs(exp_phase_path)
                        log_text += f'creating new phase experiment path: {exp_phase_path}.\n'

                        if reuse == False:
                            start_update_time = time.time()
                            update_set = np.array(tmp_inputs_buf)
                            model, mu, sigma = self._update_func(update_set,
                                                                 exp_phase_path)  # store the new model in the folder before drift
                            self.mu = mu
                            self.sigma = sigma
                            current_state = max(state_buf.keys()) + 1
                            log_text += f'State: cannot find state to reuse. Create new model. Created new state {current_state}. Store model to {exp_phase_path}\n'
                            state_buf[current_state] = histogram_hist
                            self._persist_model(phase_path, model)
                            end_update_time = time.time()

                            time_log += 'Update: ' + str(round(end_update_time - start_update_time, 3)) + '\n'

                    outputs, hidden_state = model.forward(batch_X, gpu=self.gpu)
            outputs = outputs[:current_batch_size]
            batch_X = batch_X[:current_batch_size]
            # the input sequence is predicted in a reverse oder, so flip it back for evaluation
            outputs = torch.flip(outputs, [1])
            errors = torch.abs(batch_X - outputs)  # (batch_size, window_lenth, input_size)
            error_buf += errors.reshape(-1, self.input_size).tolist()  # (N*input_size)
            hidden_state_buf += latent_representation.tolist()
            reference_hidden_state_buf += reference_latent_representation.tolist()
            tmp_latent_rep_buf += reference_latent_representation.tolist()
            tmp_inputs_buf += batch_X.reshape(-1, self.input_size).tolist()
            tmp_inputs_buf = tmp_inputs_buf if len(
                tmp_inputs_buf) < self.l_new_size * self.batch_size * self.window_size \
                else tmp_inputs_buf[-self.l_new_size * self.batch_size * self.window_size:]
            assert len(tmp_inputs_buf) % self.batch_size == 0

            input_buf += batch_X.reshape(-1, self.input_size).tolist()
            output_buf += outputs.reshape(-1, self.input_size).tolist()

            scores = self._calc_anomaly_score(errors)  # return (N*window_size)
            score_buf += scores

            pd.Series(scores).to_csv(f'{exp_phase_path}/phase_score.csv', index=False, header=False, mode='a')
            pd.Series(batch_y.ravel()).to_csv(f'{exp_phase_path}/phase_label.csv', index=False, header=False, mode='a')

        output_file_score = os.path.join(self.experiment_dir, 'score.csv')
        output_file_log = os.path.join(self.experiment_dir, 'log')

        with open(output_file_log, 'w') as f_l:
            f_l.write(log_text)
            f_l.close()

        with open(output_file_score, 'a') as f_s:
            pd.Series(score_buf).to_csv(f_s, index=False, header=False, mode='a')

        return time_log

    def evaluate(self, prediction, groundtruth):
        """

        Args:
            prediction ():
            groundtruth ():
            tau (): threshold of anomaly scores. Scores over tau will be classified as anomalies
        """
        beta = 1  # weight recall more than precision
        fpr, tpr, thresholds = metrics.roc_curve(groundtruth, prediction)
        auc = metrics.auc(fpr, tpr)

        if self.tau <= thresholds[0]:
            false_positive_rate = 0
        elif self.tau >= thresholds[-1]:
            false_positive_rate = 1
        else:
            # the false positive rate is lineally approximated from fpr and thresholds according to self.tau
            idx = bisect(thresholds, self.tau)
            rate = (self.tau - thresholds[idx - 1]) / (thresholds[idx] - thresholds[idx - 1])
            false_positive_rate = fpr[idx - 1] + rate * (fpr[idx] - fpr[idx - 1])
        prediction = [1 if pred >= self.tau else 0 for pred in prediction]
        precision, recall, f_beta, true_sum = metrics.precision_recall_fscore_support(groundtruth, prediction,
                                                                                      beta=beta, average='binary')
        metric = {'precision': precision,
                  'recall': recall,
                  'f_beta': f_beta,
                  'support': str(true_sum),
                  'auc': auc,
                  'false_positive_rate': false_positive_rate,
                  'tau': self.tau
                  }
        with open(os.path.join(self.experiment_dir, 'metric.txt'), 'w') as m_f:
            json.dump(metric, m_f)
