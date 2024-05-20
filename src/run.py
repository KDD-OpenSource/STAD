import glob
import os
import re
import json
import time
import itertools

from datetime import datetime
from dataPrepare.realworld import Forest
from dataPrepare.sin import Sin_swap_easy, Sin_swap_hard, Sin_ampl_easy, Sin_ampl_hard, Sin_gradual_easy, \
    Sin_incremental_easy, Sin_gradual_hard, Sin_incremental_hard
from dataPrepare.smd_drift import SMD_Small, SMD_Large
from model import train, predict
from pathlib import Path

root = Path(__file__).resolve().parent.parent
datasets = ['sin_incremental_easy', 'sin_gradual_easy', 'sin_incremental_hard', 'sin_gradual_hard', 'sin_ampl_easy',
            'sin_ampl_hard', 'sin_swap_easy', 'sin_swap_hard', 'SMD_Small', 'SMD_Large', 'Forest']
with open(f'{root}/configuration/config.json', 'r') as f:
    config = json.load(f)

data_root = os.path.join(root, 'data')
model = 'STAD'
continue_training = False
dataset_obj = {'sin_swap_easy': Sin_swap_easy, 'sin_swap_hard': Sin_swap_hard,
               'sin_ampl_easy': Sin_ampl_easy, 'sin_ampl_hard': Sin_ampl_hard,
               'sin_incremental_hard': Sin_incremental_hard,
               'sin_incremental_easy': Sin_incremental_easy, 'sin_gradual_easy': Sin_gradual_easy,
               'sin_gradual_hard': Sin_gradual_hard, 'SMD_Small': SMD_Small, 'SMD_Large': SMD_Large, 'Forest': Forest}
gpu = 0
seeds = [24, 42, 14239]

result_dir = f'output'


def train_func(train_set, experiment_dir, hidden_dim, batch_size, dropout_rate, epochs,
               early_stopping_patience, learning_rate, weight_decay, window_size, input_dim):
    is_initialization = True
    trainer = train.Train(experiment_dir, hidden_dim, batch_size, dropout_rate, epochs, early_stopping_patience,
                          learning_rate, weight_decay, window_size, input_dim, is_initialization, gpu=gpu)
    trainer.train(train_set)


def predict_func(test_set, test_label, experiment_dir, hidden_dim, batch_size, dropout_rate, window_size,
                 epochs, early_stopping_patience, learning_rate, weight_decay,
                 input_dim, l_hist_size, l_new_size, ks_alpha, ks_hist_min_size, kld_epsilon, continue_training):
    online_exp_dir = os.path.join(experiment_dir, 'online_learning')
    if not os.path.exists(online_exp_dir):
        os.makedirs(online_exp_dir)
    param_path = os.path.join(experiment_dir, 'initialization', 'parameters')
    model_path = os.path.join(experiment_dir, 'initialization', 'model.th')

    pred = predict.Predict(online_exp_dir, param_path, hidden_dim, batch_size, dropout_rate, window_size,
                           epochs, early_stopping_patience, learning_rate, weight_decay,
                           input_dim, l_hist_size, l_new_size, ks_alpha, ks_hist_min_size, continue_training,  gpu=gpu)
    timelog = pred.predict(test_set, test_label, model_path, kld_epsilon)

    return timelog


for dataset in datasets:
    now = datetime.now()
    run_id = re.sub('[-: ]', '_', str(now).split('.')[0])

    config_exp = config[dataset.split('_')[0]]

    win_len = config_exp['win_len']
    hds = config_exp['hidden_size']
    epochs = config_exp['epochs']
    lrs = config_exp['lr']
    dropouts = config_exp['dropout']
    batch_size = config_exp['batch_size']
    weight_decay = config_exp['weight_decay']
    input_dim = config_exp['dim']
    early_stopping_patience = config_exp['early_stopping_patience']
    l_hist_size = config_exp['l_hist_size']
    l_new_sizes = config_exp['l_new_size']
    ks_alphas = config_exp['ks_alpha']
    ks_hist_min_sizes = config_exp['ks_hist_min_size']
    kld_epsilons = config_exp['kld_epsilon']
    for kld_epsilon in kld_epsilons:
        output_dir = f'{root}/{result_dir}/kld_{kld_epsilon}/{model}/{dataset}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        x_train, y_train, x_test, y_test, y_test_binary = dataset_obj[dataset](win_len=win_len).data()

        for (hd, epoch, lnew, ks_hist_min_size, alpha, lr, dr, wd, seed) in itertools.product(hds, epochs, l_new_sizes,
                                                                                              ks_hist_min_sizes,
                                                                                              ks_alphas, lrs, dropouts,
                                                                                              weight_decay, seeds):
            print(
                f'{dataset}: win_len={win_len}, hd={hd}, epoch={epoch}, lnew={lnew}, alpha={alpha}, lr={lr}, dropout={dr},'
                f' weight_decay={wd}, seed={seed}.')
            identifier = f'hd{hd}_epoch{epoch}_lnew{lnew}_alpha{alpha}_lr{lr}_drop{dr}_seed{seed}_{run_id}'
            experiment_dir = f'{output_dir}_minHist_{ks_hist_min_size}/{identifier}'
            if len(glob.glob(
                    f'{output_dir}_minHist_{ks_hist_min_size}/hd{hd}_epoch{epoch}_lnew{lnew}_alpha{alpha}_lr{lr}_drop{dr}_seed{seed}_*')) != 0:
                continue

            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)

            init_exp_dir = os.path.join(experiment_dir, 'initialization')
            if not os.path.exists(init_exp_dir):
                os.mkdir(init_exp_dir)

            '''Training'''
            train_start = time.time()
            train_func(x_train, init_exp_dir, hd, batch_size, dr, epoch, early_stopping_patience, lr,
                       wd, win_len, input_dim)
            train_time = round(time.time() - train_start, 3)

            '''Online processing'''
            pred_start = time.time()
            timelog = predict_func(x_test, y_test_binary, experiment_dir, hd, batch_size, dr, epoch,
                                   early_stopping_patience,
                                   lr, wd, win_len, input_dim, l_hist_size, lnew, alpha, ks_hist_min_size, kld_epsilon,
                                   continue_training)
            pred_time = round(time.time() - pred_start, 3)

            overall_timelog = 'Initialization: ' + str(train_time) + '\n' + timelog + 'Prediction processing: ' \
                              + str(pred_time) + '\n'

            online_exp_dir = os.path.join(experiment_dir, 'online_learning')
            f = open(f'{online_exp_dir}/time_log.txt', "w")
            f.write(overall_timelog)
            f.close()
