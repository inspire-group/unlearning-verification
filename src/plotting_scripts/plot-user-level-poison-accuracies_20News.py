"""
thrown together by David Sommer (david.sommer at inf.ethz.ch) in 2021.

This script uses the output of the 'train_20news.py' script.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import sys
import subprocess
#import pandas as pd
import argparse
import tensorflow as tf
import keras
import numpy as np
from keras.models import load_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# available_gpus = subprocess.check_output('nvidia-smi | grep "0%" -B1 | cut -d" " -f4 | grep -v -e "--" | sed "/^$/d"', shell=True).split(b'\n')[:-1 ]
# assert len(available_gpus) >= 1
# USE_GPU=available_gpus[-1].decode("utf-8")
USE_GPU = '6'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']=USE_GPU


from keras import backend as keras_backend
import gc
def clear_memory_from_keras():
    keras_backend.clear_session()
    gc.collect()


# ################################### ALPHA STUFF #####

from scipy.stats import binom
def beta(alpha, q, p, n):
    """
    alpha: maximal mass in right tail of H_0
    q: prob of H_0 (deleted)
    p: prob of H_1 (not deleted)
    n: number of measurements
    """
    if p==1:
        p -= 1e-4
    x = np.arange(n+1)

    pdf_H0 = binom.pmf(x, n=n, p=q)
    cdf_H0 = np.cumsum(pdf_H0)

    idx = np.argmax(cdf_H0 + alpha >= 1)
    #idx = np.argmax(cdf_H0 + alpha > 1)-1

    return np.sum( binom.pmf( np.arange(idx+1) , n=n, p=p) )

def beta_performance(alpha, q_list, p_list, n_list):
    beta_mtx = []
    for num in range(len(q_list)):
        beta_tmp = []
        for n in n_list:
            beta_tmp.append(beta(alpha, q_list[num], p_list[num], n))
        beta_mtx.append(beta_tmp)
    return np.array(beta_mtx)

def beta_performance_with_varied_alpha(alpha_list, q, p, n_list):
    beta_mtx = []
    for alpha in alpha_list:
        beta_tmp = []
        for n in n_list:
            beta_tmp.append(beta(alpha, q, p, n))
        beta_mtx.append(beta_tmp)
    return np.array(beta_mtx)


def figure_plot_for_mtx(value_mtx, x_list, x_label, y_label, label_list, file_name='sample.png', log_y=False):
    assert value_mtx.shape[0]<=10, "more than 10 lines!!!!!"


    color_list = ['b', 'r', 'g', 'tab:orange',  'tab:brown', 'tab:gray','tab:pink', 'tab:olive', 'tab:purple','tab:cyan']
    marker_list = ['o', 's', 'D', 'H', 'v', '*', '^', '<', '>','P']
    line_list = ['-', ':', '--', '-.','-', ':', '--', '-.','-', ':']

    fig = plt.figure(figsize = (16,9))
    plt.rc('axes', linewidth=3)
    plt.rc('font', family='serif', size=28)
    plt.rc('text', usetex=False)
    plt.rcParams['mathtext.fontset'] = 'cm'

#     params = {'backend': 'ps',
#               'text.usetex': True,
#               'font.family': 'serif'
#     }

#     matplotlib.rcParams.update(params)


    for num in range(value_mtx.shape[0]):
        plt.plot(x_list, value_mtx[num], label=label_list[num], linewidth=4,
                 linestyle=line_list[num], color=color_list[num], marker=marker_list[num], markersize=15)

    legend = plt.legend(handlelength=3)
    legend.get_frame().set_linewidth(3.0)
    plt.xlabel(x_label,size=36)
    plt.ylabel(y_label,size=36)
    x_tick_idx = np.arange(0, len(x_list), 5)
    plt.xticks(x_list[x_tick_idx],x_list[x_tick_idx])
    if log_y:
        plt.yscale('log')
    plt.show()
    fig.savefig(file_name,bbox_inches='tight')
    return


def cdf_comparison_plot(cdf_list, label_list, file_name):
    fig = plt.figure(figsize = (12,6.8))
    plt.rc('font', family='serif', size=26)
    plt.rc('axes', linewidth=3)
    color_list = ['b', 'r', 'g', 'tab:orange',  'tab:brown', 'tab:gray','tab:pink', 'tab:olive', 'tab:purple','tab:cyan']
    line_list = ['-', ':', '--', '-.','-', ':', '--', '-.','-', ':']
    assert len(cdf_list)<=10, "more than 10 lines!!!!!"
    for num in range(len(cdf_list)):
        plt.plot(cdf_list[num][0], cdf_list[num][1], label=label_list[num], linewidth=4,
                linestyle=line_list[num], color=color_list[num])
    legend = plt.legend(handlelength=3, fontsize=30)
    legend.get_frame().set_linewidth(3.0)
    plt.xlabel('backdoor success rate', fontsize=30)
    plt.ylabel('cumulative distribution', fontsize=30)
    plt.xlim([-0.05, 1.05])
    plt.show()
    fig.savefig(file_name,bbox_inches='tight')
    return

def cdf_compute(values):
    values = np.array(values).flatten()
    sort_list = np.sort(values)
    cdf = 1. * (np.arange(len(values))+1.0) / len(values)
    return (sort_list, cdf)

# ##################### data handling ################################


def model_performance(model, train_set_details, test_set_details, ex_set_details, backdoor_details):


    def _apply_backdoor(x_inputs, mask_indices, mask_content):
        x_tmp = np.copy(x_inputs)
        x_tmp[:,mask_indices] = mask_content
        return x_tmp


    def _thorough_measure(set_details, reduce_tar_label=False):
        (x, y, author, is_p,real_x, real_lbs ) = set_details
        authors_cnt, samples_cnt, attack_samples_cnt, benign_corr_preds_cnt, attack_corr_preds_cnt = 0, 0, 0, 0, 0
        benign_acc_per_author, attack_acc_per_author = [], []
        for author_tmp in list(set(author)):
            if authors_cnt%100 == 0:
                print(authors_cnt, end=' ')
            authors_cnt += 1
            data_idx = np.argwhere(author==author_tmp).flatten()
            x_tmp, y_tmp, real_lbs_tmp, real_x_tmp = x[data_idx], y[data_idx], real_lbs[data_idx], real_x[data_idx]
            samples_cnt += len(data_idx)
            benign_preds_tmp = model.predict(real_x_tmp)
            benign_corr_preds_cnt += np.sum(np.argmax(benign_preds_tmp,axis=1)==real_lbs_tmp)
            benign_acc_per_author.append(np.sum(np.argmax(benign_preds_tmp,axis=1)==real_lbs_tmp)/(len(data_idx+0.0)))
            #if len(backdoor_author_list)!=0:
            if author_tmp in backdoor_authors:
                idx = np.argwhere(backdoor_authors==author_tmp).flatten()[0]
                tar_lab_tmp = backdoor_labels[idx]
                backdoor_pos_tmp, backdoor_val_tmp = backdoor_positions[idx], backdoor_values[idx]
                if reduce_tar_label:
                    attack_preds_tmp = model.predict( _apply_backdoor(real_x_tmp[real_lbs_tmp!=tar_lab_tmp],
                                                                      backdoor_pos_tmp, backdoor_val_tmp) )
                else:
                    attack_preds_tmp = model.predict( _apply_backdoor(real_x_tmp, backdoor_pos_tmp, backdoor_val_tmp) )
                attack_corr_preds_cnt += np.sum(np.argmax(attack_preds_tmp, axis=1)==tar_lab_tmp)
                attack_samples_cnt += attack_preds_tmp.shape[0]
                attack_acc_per_author.append(np.sum(np.argmax(attack_preds_tmp, axis=1)==tar_lab_tmp)/(attack_preds_tmp.shape[0]+0.0))
        benign_acc_per_author =  np.array(benign_acc_per_author)
        avg_benign_acc = benign_corr_preds_cnt/(samples_cnt+0.0)

        if len(backdoor_authors)!=0:
            attack_acc_per_author =  np.array(attack_acc_per_author)
            if attack_samples_cnt==0:
                avg_attack_acc = np.array([])
            else:
                avg_attack_acc = attack_corr_preds_cnt/(attack_samples_cnt+0.0)
        else:
            avg_attack_acc, attack_acc_per_author = 0, [0 for author_tmp in list(set(author))]
        return (avg_benign_acc, benign_acc_per_author, avg_attack_acc, attack_acc_per_author)




    (backdoor_authors, backdoor_positions, backdoor_values, backdoor_labels) = backdoor_details

    #(benign_acc_in, attack_acc_in) = _rough_measure(test_set_details)
    #(benign_acc_ex, attack_acc_ex) = _rough_measure(ex_set_details)
    (avg_benign_acc_in, benign_acc_per_author_in, \
     avg_attack_acc_in, attack_acc_per_author_in) = _thorough_measure(test_set_details)
    (avg_benign_acc_ex, benign_acc_per_author_ex, \
     avg_attack_acc_ex, attack_acc_per_author_ex) = _thorough_measure(ex_set_details)
    print(avg_benign_acc_in, avg_benign_acc_ex)
    print(attack_acc_per_author_in, attack_acc_per_author_ex)
    author_performance_in = (avg_benign_acc_in, benign_acc_per_author_in, avg_attack_acc_in, attack_acc_per_author_in)
    author_performance_ex = (avg_benign_acc_ex, benign_acc_per_author_ex, avg_attack_acc_ex, attack_acc_per_author_ex)

    return author_performance_in, author_performance_ex


#     (backdoor_author_list, tar_lbs_list, backdoor_pos_list, backdoor_val_list) = backdoor_details

#     #(benign_acc_in, attack_acc_in) = _rough_measure(test_set_details)
#     #(benign_acc_ex, attack_acc_ex) = _rough_measure(ex_set_details)
#     (avg_benign_acc_in, benign_acc_per_author_in, \
#      avg_attack_acc_in, attack_acc_per_author_in) = _thorough_measure(test_set_details)
#     (avg_benign_acc_ex, benign_acc_per_author_ex, \
#      avg_attack_acc_ex, attack_acc_per_author_ex) = _thorough_measure(ex_set_details)
#     author_performance_in = (avg_benign_acc_in, benign_acc_per_author_in, avg_attack_acc_in, attack_acc_per_author_in)
#     author_performance_ex = (avg_benign_acc_ex, benign_acc_per_author_ex, avg_attack_acc_ex, attack_acc_per_author_ex)

#     #print('benign and attack acc for included users: ', (benign_acc_in, attack_acc_in))
#     #print('benign and attack acc for excluded users: ', (benign_acc_ex, attack_acc_ex))
#     print(avg_benign_acc_in, avg_attack_acc_in, avg_benign_acc_ex, avg_attack_acc_ex)
#     #print(np.average(benign_acc_per_author_in), np.average(attack_acc_per_author_in), np.average(benign_acc_per_author_ex), np.average(attack_acc_per_author_ex))
#     #print(benign_acc_per_author_in, attack_acc_per_author_in, benign_acc_per_author_ex, attack_acc_per_author_ex)
#     #print(np.quantile(attack_acc_per_author_in, 0.1), np.quantile(attack_acc_per_author_in, 0.9), np.amax(attack_acc_per_author_in), np.amin(attack_acc_per_author_in))
#     #print(np.quantile(attack_acc_per_author_ex, 0.1), np.quantile(attack_acc_per_author_ex, 0.9), np.amax(attack_acc_per_author_ex), np.amin(attack_acc_per_author_ex))
#     return author_performance_in, author_performance_ex

def load_model_and_data(model_dir):
    model = load_model(os.path.join(model_dir, 'backdoored_model'), compile=False)
    data = np.load(os.path.join(model_dir, 'data_record.npz'))
    X_train, y_train, author_train, is_p_train = data['X_train'], data['y_train'], data['author_train'], data['is_p_train']
    target_lbs_train, real_lbs_train = data['target_lbs_train'], data['real_lbs_train']

    X_test, y_test, author_test, is_p_test = data['X_test'], data['y_test'], data['author_test'], data['is_p_test']
    target_lbs_test, real_lbs_test = data['target_lbs_test'], data['real_lbs_test']

    ex_X, ex_y, ex_author, ex_is_poisoned = data['ex_X'], data['ex_y'], data['ex_author'], data['ex_is_poisoned']
    ex_target_labels, ex_real_labels = data['ex_target_labels'], data['ex_real_labels']

    real_X_train, real_X_test, ex_real_X = data['real_X_train'], data['real_X_test'], data['ex_real_X']

    data = np.load(os.path.join(model_dir, 'backdoor_info.npz'))
    backdoor_authors, backdoor_labels = data['backdoor_authors'], data['backdoor_labels']
    backdoor_positions, backdoor_values = data['backdoor_positions'], data['backdoor_values']

    train_set_details = (X_train, y_train, author_train, is_p_train, real_X_train, real_lbs_train)
    test_set_details = (X_test, y_test, author_test, is_p_test, real_X_test, real_lbs_test)
    ex_set_details = (ex_X, ex_y, ex_author, ex_is_poisoned, ex_real_X, ex_real_labels)
    backdoor_details = (backdoor_authors, backdoor_positions, backdoor_values, backdoor_labels)
    return model, train_set_details, test_set_details, ex_set_details, backdoor_details


# def create_per_user_performance_numbers(model_info_base_name, author_poison_list, poison_list, save_base_dir, save_model_name)


# inst_list = np.arange(10)
# model_architecture = 'EMNIST_dense3_1000_users_'
# # save_model_name = 'EMNIST_MLP_'
# for author_poison in author_poison_list:
#     save_dir = os.path.join(save_base_dir,'author_poison_'+str(author_poison))
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     for poison_r in poison_list:
#         for inst in inst_list:
#             model_path = '../model_training_code/deletion/src/evaluation/varied_author_poison'
#             model_dir = os.path.join(model_path, model_architecture+str(poison_r)+'_poison_ratio_'
#                                      +str(author_poison)+'_author_poison_inst_'+str(inst))
#             model, train_set_details, test_set_details, ex_set_details, backdoor_details \
#             = load_model_and_data(model_dir)
#             author_performance_in, author_performance_ex \
#             = model_performance(model, train_set_details, test_set_details, ex_set_details, backdoor_details)
#             save_file = save_model_name+str(poison_r)+'_poison_ratio_inst_'+str(inst)+'.npz'
#             save_file_path = os.path.join(save_dir, save_file)
#             np.savez(file=save_file_path,
#                      in_benign_acc_avg = author_performance_in[0], in_benign_acc_list = author_performance_in[1],
#                      in_attack_acc_avg = author_performance_in[2], in_attack_acc_list = author_performance_in[3],
#                      ex_benign_acc_avg = author_performance_ex[0], ex_benign_acc_list = author_performance_ex[1],
#                      ex_attack_acc_avg = author_performance_ex[2], ex_attack_acc_list = author_performance_ex[3]
#                      )


# author_poison_list = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
# poison_list = [0.5]

# save_model_name = 'EMNIST_MLP_'

# model_info_base_name = './model_name'
# save_base_dir = './base_dir'

# create_per_user_performance_numbers(
#     model_info_base_name=model_info_base_name,
#     author_poison_list=author_poison_list,
#     poison_list=poison_list)

def create_train_and_test_acc_for_poisonratio_0(model_dir):
    assert "_pos_rate_0.0_" in model_dir  # check for poison ratio = 0

    def accuracy(x, y):
        return np.sum(np.argmax(x, axis=1) == y) / len(y)

    def set_details_to_x_y(details):
        x, y, author, is_p, real_x, real_lbs = details
        assert np.all(y == real_lbs)  # sanity check: poison ratio = 0
        return x, y

    model_dir = os.path.join(MODEL_DATA_BASE_DIR, model_dir)

    model, train_set_details, test_set_details, ex_set_details, backdoor_details = load_model_and_data(model_dir)

    train_x, train_y = set_details_to_x_y(train_set_details)
    test_x, test_y = set_details_to_x_y(test_set_details)
    ex_x, ex_y = set_details_to_x_y(ex_set_details)

    train_y_pred = model.predict(train_x)
    test_y_pred = model.predict(test_x)
    ex_y_pred = model.predict(ex_x)

    # combine test_set and ex_set, both unseen during training by the model, to a new test set
    all_test_y = np.concatenate((test_y, ex_y), axis=0)
    all_test_y_pred = np.concatenate((test_y_pred, ex_y_pred), axis=0)

    print("\n[*] Accuracies")
    print(f"  train acc {accuracy(train_y_pred, train_y)}")
    print(f"  test  acc {accuracy(all_test_y_pred, all_test_y)}")

    clear_memory_from_keras()


def create_per_user_performance_numbers():

    number_of_authors = 100

    author_poison_list = [0.05]
    poison_list = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])  # [0.1, 0.3, 0.5, 0.7, 0.9]
    inst_list = np.arange(10)
    model_architecture = '20News_'
    save_model_name = '20News_lstm_'
    for author_poison in author_poison_list:
        save_dir = os.path.join('./saved_results', 'author_poison_' + str(author_poison))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for poison_r in poison_list:
            for inst in inst_list:
                # model_path = '../model_training_code/deletion/src/evaluation/5_percent_author_poison'
                # model_dir = os.path.join(model_path, model_architecture+str(poison_r)+'_poison_ratio_'
                #                          +str(author_poison)+'_author_poison_inst_'+str(inst))
                da_dir = f"20News_nbr_authrs_{number_of_authors}_pos_rate_{poison_r}_aut_ps_rate_{author_poison}_run_{inst}"
                # da_dir = f"20News_nbr_authrs_{number_of_authors}_pos_rate_{poison_r}_aut_ps_rate_{author_poison}_run_{inst}"
                model_dir = os.path.join(MODEL_DATA_BASE_DIR, da_dir)
                print(model_dir)

                if not os.path.exists(model_dir) and poison_r == 0 and inst > 0:
                    print(f'does not exists: {model_dir}, skip')  # we only need one inst for poison_ratio == 0.
                    continue

                model, train_set_details, test_set_details, ex_set_details, backdoor_details \
                = load_model_and_data(model_dir)
                author_performance_in, author_performance_ex \
                = model_performance(model, train_set_details, test_set_details, ex_set_details, backdoor_details)
                save_file = save_model_name+str(poison_r)+'_poison_ratio_inst_'+str(inst)+'.npz'
                save_file_path = os.path.join(save_dir, save_file)
                np.savez(file=save_file_path,
                         in_benign_acc_avg = author_performance_in[0], in_benign_acc_list = author_performance_in[1],
                         in_attack_acc_avg = author_performance_in[2], in_attack_acc_list = author_performance_in[3],
                         ex_benign_acc_avg = author_performance_ex[0], ex_benign_acc_list = author_performance_ex[1],
                         ex_attack_acc_avg = author_performance_ex[2], ex_attack_acc_list = author_performance_ex[3]
                         )
                clear_memory_from_keras()


    author_poison_list = np.array([0.2, 0.5, 1])  # 0.05 is alredy contained above # [0.02, 0.05, 0.1, 0.2, 0.5, 1]
    poison_list = [0.5]
    inst_list = np.arange(10)
    model_architecture = '20News_'
    save_model_name = '20News_lstm_'
    for author_poison in author_poison_list:
        save_dir = os.path.join('./saved_results','author_poison_'+str(author_poison))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for poison_r in poison_list:
            for inst in inst_list:
                # model_path = '../model_training_code/deletion/src/evaluation/varied_author_poison'
                # model_dir = os.path.join(model_path, model_architecture+str(poison_r)+'_poison_ratio_'
                #                          +str(author_poison)+'_author_poison_inst_'+str(inst))
                da_dir = f"20News_nbr_authrs_{number_of_authors}_pos_rate_{poison_r}_aut_ps_rate_{author_poison}_run_{inst}"
                model_dir = os.path.join(MODEL_DATA_BASE_DIR, da_dir)

                print(model_dir)

                model, train_set_details, test_set_details, ex_set_details, backdoor_details \
                = load_model_and_data(model_dir)
                author_performance_in, author_performance_ex \
                = model_performance(model, train_set_details, test_set_details, ex_set_details, backdoor_details)
                save_file = save_model_name+str(poison_r)+'_poison_ratio_inst_'+str(inst)+'.npz'
                save_file_path = os.path.join(save_dir, save_file)
                np.savez(file=save_file_path,
                         in_benign_acc_avg = author_performance_in[0], in_benign_acc_list = author_performance_in[1],
                         in_attack_acc_avg = author_performance_in[2], in_attack_acc_list = author_performance_in[3],
                         ex_benign_acc_avg = author_performance_ex[0], ex_benign_acc_list = author_performance_ex[1],
                         ex_attack_acc_avg = author_performance_ex[2], ex_attack_acc_list = author_performance_ex[3]
                         )
                clear_memory_from_keras()


# ###############################################################################################################
# ###############################################################################################################
# ##### PLOTTING CODE
# ###############################################################################################################
# ###############################################################################################################


def model_performance_multiple_instances(file_path, instance_list, defended_name=None):
    def _append_numbers(instance_values, overall_list):
        if len(instance_values)>0:
            overall_list.append(instance_values)
        return

    in_benign_acc_list = []
    in_attack_acc_list = []
    ex_benign_acc_list = []
    ex_attack_acc_list = []

    for i in instance_list:
        if defended_name==None:
            file_path_tmp = file_path+str(i)+'.npz'
        else:
            file_path_tmp = file_path+str(i)+defended_name+'.npz'
        if os.path.exists(file_path_tmp):
            print('load: ' + file_path_tmp)
            data = np.load(file_path_tmp)
            _append_numbers(data['in_benign_acc_list'], in_benign_acc_list)
            _append_numbers(data['in_attack_acc_list'], in_attack_acc_list)
            _append_numbers(data['ex_benign_acc_list'], ex_benign_acc_list)
            _append_numbers(data['ex_attack_acc_list'], ex_attack_acc_list)
        else:
            #print('there are '+str(i)+' instances')
            print('NOT FOUND: ' + file_path_tmp + ', skip')
            continue

    print(in_benign_acc_list, in_attack_acc_list)

    in_benign_acc_list, in_attack_acc_list = np.concatenate(in_benign_acc_list), np.concatenate(in_attack_acc_list)
    ex_benign_acc_list, ex_attack_acc_list = np.concatenate(ex_benign_acc_list), np.concatenate(ex_attack_acc_list)
    benign_acc_list = np.concatenate((in_benign_acc_list, ex_benign_acc_list))
    return benign_acc_list, in_attack_acc_list, ex_attack_acc_list


def _figure_plot(benign_acc_matrix, in_attack_acc_matrix, ex_attack_acc_matrix,
                 x_list, ratio_up=0.75, ratio_down=0.25,
                 x_label='tmp', fig_name='./tmp.png', poison_plot=False,poison_plot_ticks=None):
    fig = plt.figure(figsize = (16,9))
    plt.rc('font', family='serif', size=28)
    plt.rc('axes', linewidth=3)

    benign_acc_avg = [np.average(np.array(acc_list)) for acc_list in benign_acc_matrix]
    in_attack_acc_avg = [np.average(np.array(acc_list)) for acc_list in in_attack_acc_matrix]
    ex_attack_acc_avg = [np.average(np.array(acc_list)) for acc_list in ex_attack_acc_matrix]

    plt.plot(x_list, benign_acc_avg, label = 'benign test acc',
             linewidth=4, linestyle = '-', color='b', marker='o', markersize=15)
    plt.plot(x_list, in_attack_acc_avg, label = 'backdoor attack acc (undeleted)',
             linewidth=4, linestyle = ':',color = 'r', marker='s', markersize=15)
    plt.plot(x_list, ex_attack_acc_avg, label = 'backdoor attack acc (deleted)',
             linewidth=4, linestyle = '--',color = 'g', marker='D', markersize=15)

    value_up = [np.quantile(acc_list, ratio_up) for acc_list in benign_acc_matrix]
    value_down = [np.quantile(acc_list, ratio_down) for acc_list in benign_acc_matrix]
    plt.fill_between(x_list,value_up,value_down,facecolor='blue',alpha=.3,interpolate=True)

    value_up = [np.quantile(acc_list, ratio_up) for acc_list in in_attack_acc_matrix]
    value_down = [np.quantile(acc_list, ratio_down) for acc_list in in_attack_acc_matrix]
    plt.fill_between(x_list,value_up,value_down,facecolor='red',alpha=.3,interpolate=True)

    value_up = [np.quantile(acc_list, ratio_up) for acc_list in ex_attack_acc_matrix]
    value_down = [np.quantile(acc_list, ratio_down) for acc_list in ex_attack_acc_matrix]
    plt.fill_between(x_list,value_up,value_down,where=value_up>value_down,facecolor='green',alpha=.3,interpolate=True)

    legend = plt.legend(handlelength=3, bbox_to_anchor=[0.3, 0.1], loc='lower left')
    legend.get_frame().set_linewidth(3.0)
    if poison_plot:
        print("")
        print(x_list, poison_plot_ticks)
        plt.xticks(x_list, poison_plot_ticks)
    plt.xlabel(x_label,size=36)
    plt.ylabel('accuracy value',size=36)
#     plt.title(title)

    fig.savefig(fig_name,bbox_inches='tight')
    #plt.show()
    return


def generate_cdf_plot():

    old_dpi = matplotlib.rcParams['figure.dpi']

    matplotlib.rcParams['figure.dpi'] = 150
    file_path = './combined_results/20News_0.05_author_poison_0.5_poison.npz'
    data = np.load(file_path)
    p_list = data['in_attack_acc_list']
    q_list = data['ex_attack_acc_list']
    p_cdf = cdf_compute(p_list)
    q_cdf = cdf_compute(q_list)

    file_name = './saved_figures/20News_0.05_author_poison_0.5_poison_acc_cdf.png'
    cdf_list = [p_cdf, q_cdf]
    label_list = ['undeleted users', 'deleted users']
    cdf_comparison_plot(cdf_list, label_list, file_name)

    matplotlib.rcParams['figure.dpi'] = old_dpi


def generate_alpha_plots_and_all_table_statistics():

    file_name = "./combined_results/20News_"
    poison_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    author_poison = 0.05
    defended = False

    benign_acc_matrix = []
    in_attack_acc_matrix = []
    ex_attack_acc_matrix = []

    for poison in poison_list:
        file_path = file_name + str(author_poison) + '_author_poison_' + str(poison) + '_poison'
        if defended:
            file_path += '_defended'
        data = np.load(file_path + '.npz')
        benign_acc_matrix.append(data['benign_acc_list'])
        in_attack_acc_matrix.append(data['in_attack_acc_list'])
        ex_attack_acc_matrix.append(data['ex_attack_acc_list'])

    acc_list = [np.average(np.array(acc_list)) for acc_list in benign_acc_matrix]
    p_list = [np.average(np.array(acc_list)) for acc_list in in_attack_acc_matrix]
    q_list = [np.average(np.array(acc_list)) for acc_list in ex_attack_acc_matrix]


    # poison_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    # instance_list = np.arange(10)
    #fig_name='./saved_figures/EMNIST_MLP_2_percent_author_poison_acc.png'
    # acc_list, p_list, q_list = model_performance_over_poison_ratio(file_name, poison_list, author_poison=0.05)
    n_list = np.arange(0, 31, 1)
    alpha = 1e-3
    all_labels = ['10%', '30%', '50%', '70%', '90%']
    # all_labels = ['10%', '30%',  '50%',  '70%']
    label_list = [ n + ' poison ratio' for n in all_labels]
    label_list = [ n + ' data poison ratio ($f_{data}$)' for n in all_labels]
    beta_mtx = beta_performance(alpha, q_list, p_list, n_list)
    file_name = './saved_figures/20News_lstm_5_percent_author_poison_verify_a_1e-3.png'
    figure_plot_for_mtx(beta_mtx, n_list, 'number of samples ($n$)', 'Type II error ($\\beta$)', label_list,
                        file_name=file_name, log_y=True)

    n_list = np.arange(0, 31, 1)
    alpha_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    label_list = ['alpha=1e-1', 'alpha=1e-2', 'alpha=1e-3', 'alpha=1e-4', 'alpha=1e-5']
    label_list = ['$\\alpha=10^{-1}$', '$\\alpha=10^{-2}$', '$\\alpha=10^{-3}$', '$\\alpha=10^{-4}$', '$\\alpha=10^{-5}$']
    poison_index = 2
    q = q_list[poison_index]
    p = p_list[poison_index]
    beta_mtx = beta_performance_with_varied_alpha(alpha_list, q, p, n_list)
    file_name = './saved_figures/20News_lstm_5_percent_author_poison_verify_poison_ratio_50.png'
    figure_plot_for_mtx(beta_mtx, n_list, 'number of samples ($n$)', 'Type II error ($\\beta$)', label_list,
                        file_name=file_name,log_y=True)



    #### TABLE STATISTICS

    # baseline
    """ NO training test included in post-processed data. Need to do it by hand.
    author_poison = 0.0
    poison = 0.0
    baseline_file_path = file_name + str(author_poison) + '_author_poison_' + str(poison) + '_poison'
    data = np.load(file_path+'.npz')

    benign_acc_matrix.append(data['benign_acc_list'])
    in_attack_acc_matrix.append(data['in_attack_acc_list'])
    ex_attack_acc_matrix.append(data['ex_attack_acc_list'])
    """
    poison = 0.5
    author_poison = 0.05
    alpha = alpha = 1e-3

    poison_index = 2
    assert poison_list[poison_index] == poison

    acc = acc_list[poison_index]
    q = q_list[poison_index]
    p = p_list[poison_index]

    print(f"\nfor poison={poison}, author_poison={author_poison}")
    print("benign", acc)
    print("p", p)
    print("q", q)
    print("  beta-performance: ", beta_performance(alpha, q_list=[q], p_list=[p], n_list=[30]))

    """
    figure_plot(benign_acc_matrix, in_attack_acc_matrix, ex_attack_acc_matrix,poison_list,
                 ratio_up=ratio_up, ratio_down=ratio_down,
                 x_label='percentage of backdoored data samples ($f_{data}$)', fig_name=fig_name,
                poison_plot=True,poison_plot_ticks=poison_plot_ticks)#x_label='poison ratio', fig_name=fig_name
    """
    return



def model_performance_over_poison_ratio(file_name, poison_list, author_poison=0.05,
                                        ratio_up=0.75, ratio_down=0.25, fig_name='./tmp.png',defended=False,
                                        poison_plot_ticks=['0%', '10%', '30%', '50%', '70%', '90%']):
    benign_acc_matrix = []
    in_attack_acc_matrix = []
    ex_attack_acc_matrix = []



    for poison in poison_list:
        file_path = file_name+str(author_poison)+'_author_poison_'+str(poison)+'_poison'
        if defended:
            file_path += '_defended'
        data = np.load(file_path+'.npz')
        benign_acc_matrix.append(data['benign_acc_list'])
        in_attack_acc_matrix.append(data['in_attack_acc_list'])
        ex_attack_acc_matrix.append(data['ex_attack_acc_list'])


    _figure_plot(benign_acc_matrix, in_attack_acc_matrix, ex_attack_acc_matrix,poison_list,
                 ratio_up=ratio_up, ratio_down=ratio_down,
                 x_label='percentage of backdoored data samples ($f_{data}$)', fig_name=fig_name,
                poison_plot=True,poison_plot_ticks=poison_plot_ticks)#x_label='poison ratio', fig_name=fig_name
    return


def model_performance_over_author_poison_ratio(file_name, author_poison_list, poison=0.5,
                                               ratio_up=0.75, ratio_down=0.25, fig_name='./tmp.png', defended=False):
    benign_acc_matrix = []
    in_attack_acc_matrix = []
    ex_attack_acc_matrix = []


    for author_poison in author_poison_list:
        file_path = file_name+str(author_poison)+'_author_poison_'+str(poison)+'_poison'
        if defended:
            file_path += '_defended'
        data = np.load(file_path+'.npz')
        benign_acc_matrix.append(data['benign_acc_list'])
        in_attack_acc_matrix.append(data['in_attack_acc_list'])
        ex_attack_acc_matrix.append(data['ex_attack_acc_list'])

    _figure_plot(benign_acc_matrix, in_attack_acc_matrix, ex_attack_acc_matrix,author_poison_list,
                 ratio_up=ratio_up, ratio_down=ratio_down,
                 x_label='fraction of users backdooring ($f_{{user}}$)', fig_name=fig_name)
    return


def combine_results():
    combinded_dir = './combined_results'
    if not os.path.exists(combinded_dir):
        os.makedirs(combinded_dir)

    poison_ratio_list = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])
    for poison_ratio in poison_ratio_list:

        file_path = './saved_results/author_poison_0.05/20News_lstm_'+str(poison_ratio)+'_poison_ratio_inst_'
        instance_list = np.arange(10)
        benign_acc_list, in_attack_acc_list, ex_attack_acc_list \
        = model_performance_multiple_instances(file_path, instance_list, defended_name=None)
        print(len(benign_acc_list), len(in_attack_acc_list), len(ex_attack_acc_list))
        print(np.average(benign_acc_list), np.average(in_attack_acc_list), np.average(ex_attack_acc_list))
        np.savez(combinded_dir+'/20News_'+str(0.05)+'_author_poison_'+str(poison_ratio)+'_poison.npz',
                benign_acc_list=benign_acc_list, in_attack_acc_list=in_attack_acc_list,
                 ex_attack_acc_list=ex_attack_acc_list)


    author_poison_list = np.array([0.05, 0.2, 0.5, 1])
    for author_poison in author_poison_list:

        file_path = './saved_results/author_poison_'+str(author_poison)+'/20News_lstm_'+str(0.5)+'_poison_ratio_inst_'
        instance_list = np.arange(10)
        benign_acc_list, in_attack_acc_list, ex_attack_acc_list \
        = model_performance_multiple_instances(file_path, instance_list, defended_name=None)
        print(len(benign_acc_list), len(in_attack_acc_list), len(ex_attack_acc_list))
        print(np.average(benign_acc_list), np.average(in_attack_acc_list), np.average(ex_attack_acc_list))
        np.savez(combinded_dir + '/20News_'+str(author_poison)+'_author_poison_'+str(0.5)+'_poison.npz',
                benign_acc_list=benign_acc_list, in_attack_acc_list=in_attack_acc_list,
                 ex_attack_acc_list=ex_attack_acc_list)


MODEL_DATA_BASE_DIR = './model_data'

create_per_user_performance_numbers()
combine_results()

create_train_and_test_acc_for_poisonratio_0("20News_nbr_authrs_100_pos_rate_0.0_aut_ps_rate_0.0_run_0")
create_train_and_test_acc_for_poisonratio_0("20News_nbr_authrs_100_pos_rate_0.0_aut_ps_rate_0.05_run_0")

generate_alpha_plots_and_all_table_statistics()

generate_cdf_plot()


if True:
    matplotlib.rcParams['figure.dpi'] = 150
    file_name = './combined_results/20News_'
    poison_list = np.array([0, 0.1, 0.3, 0.5, 0.7, 0.9])
    ratio_up = 0.9
    ratio_down = 0.1
    fig_name = './saved_figures/20News_lstm_5_percent_author_poison_acc.png'
    model_performance_over_poison_ratio(file_name, poison_list, author_poison=0.05,
                                        ratio_up=ratio_up, ratio_down=ratio_down,fig_name=fig_name)

    author_poison_list = np.array([0.05, 0.2, 0.5, 1])

    ratio_up = 0.9
    ratio_down = 0.1

    fig_name = './saved_figures/20News_lstm_50_percent_poison_acc.png'
    model_performance_over_author_poison_ratio(file_name, author_poison_list, poison=0.5,
                                               ratio_up=ratio_up, ratio_down=ratio_down,
                                               fig_name=fig_name)
