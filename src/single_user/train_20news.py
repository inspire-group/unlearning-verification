"""
thrown together by David Sommer (david.sommer at inf.ethz.ch) 2021.

This file generates some of the empirical plots shown in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""


import os
import sys

import numpy as np

from train import get_poisoned_accuracies
from poisoned_dataset.poisoned_dataset_word_based import PoisonedDataset_20News_Last15, PoisonedDataset_20News_18828_Last15
from models import get_model_CNN_20News_rmsprop_crossentropy, get_model_LSTM_20News_adam_crossentropy, get_model_LSTM_20News_18828_adam_crossentropy_GLOVE

SEED = 42

# this directory contains the output of the data generating script
MODEL_DATA_BASE_DIR = "./model_data"

OUTPUT_FILE_BASE = "20News_results"


def write_header(filename):
    with open(filename, 'a') as f:
        f.write("number_of_authors,poisoned_ratio,author_is_poisoning_ratio,acc,poison_acc,unpoison_acc,ex_acc,ex_poison_acc,ex_real_acc,poison_reduced_acc,ex_poison_reduced_acc\n")


def dump_numbers(filename, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, ret):
    da_string = f"{number_of_authors},{poisoned_ratio},{author_is_poisoning_ratio}"
    for val in ret:
        da_string += f",{val}"

    with open(filename, 'a') as f:
        f.write(da_string + "\n")


def single_run_with_model_dir_check(output_filename, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, run):
    # define the directory we store our run data
    da_dir = f"20News_nbr_authrs_{number_of_authors}_pos_rate_{poisoned_ratio}_aut_ps_rate_{author_is_poisoning_ratio}_run_{run}"
    model_data_dir = os.path.join(MODEL_DATA_BASE_DIR, da_dir)

    seed = random_state.randint(2**32 - 1)  # need to generate seed before decide to skip run

    if os.path.exists(model_data_dir):
        print(f"Path '{model_data_dir}' already exists. Skip.")
        return

    print(f"MODEL_DIR: {model_data_dir}")

    ret = get_poisoned_accuracies(
        # get_model_CNN_20News_rmsprop_crossentropy,
        get_model_LSTM_20News_18828_adam_crossentropy_GLOVE,
        PoisonedDataset_20News_18828_Last15(
            number_of_authors=number_of_authors,
            poisoned_ratio=poisoned_ratio,
            author_is_poisoning_ratio=author_is_poisoning_ratio,
            seed=seed),
        save_all_data=True,
        model_dir=model_data_dir)

    dump_numbers(filename, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, ret)



#### START MAIN


random_state = np.random.RandomState(seed=SEED)

# reset all parameters
number_of_authors = 100
poisoned_ratio = 0.5
author_is_poisoning_ratio = 0.05

# a single run without poisoning for baseline check
filename = OUTPUT_FILE_BASE + "_f_nopoison.csv"
write_header(filename)
single_run_with_model_dir_check(filename, number_of_authors, poisoned_ratio=0.0, author_is_poisoning_ratio=0.0, run=0)


for run in range(10):

    # reset all parameters
    number_of_authors = 100
    poisoned_ratio = 0.5
    author_is_poisoning_ratio = 0.05

    filename = OUTPUT_FILE_BASE + "_f_user.csv"
    write_header(filename)
    for author_is_poisoning_ratio in np.array([0.05, 0.2, 0.5, 1]):  # [0.02, 0.05, 0.1, 0.2, 0.5, 1] # np.concatenate((np.array([0.01, 0.2]), np.arange(0.05, 1.01, 0.1))):
        single_run_with_model_dir_check(filename, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, run)


    # reset all parameters
    number_of_authors = 100
    poisoned_ratio = 0.5
    author_is_poisoning_ratio = 0.05

    filename = OUTPUT_FILE_BASE + "_f_data.csv"

    single_run_with_model_dir_check(
        output_filename=filename,
        number_of_authors=number_of_authors,
        poisoned_ratio=np.float(0),
        author_is_poisoning_ratio=author_is_poisoning_ratio,
        run=0)

    write_header(filename)
    for poisoned_ratio in np.array([0.1, 0.3, 0.5, 0.7, 0.9]):  # np.concatenate((np.array([0.01, 0.2]), np.arange(0.05, 1.01, 0.1))):
        single_run_with_model_dir_check(filename, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, run)
