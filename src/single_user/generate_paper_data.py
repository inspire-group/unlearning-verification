"""
written by David Sommer (david.sommer at inf.ethz.ch) in 2020.

This file generates some of the empirical plots shown in the paper.

This file generates results for the following datasets: EMNIST, FEMNIST, cifar10, Amazon5

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""


import os
import sys
import subprocess
import pandas as pd
import argparse


available_gpus = subprocess.check_output('nvidia-smi | grep "0%" -B1 | cut -d" " -f4 | grep -v -e "--" | sed "/^$/d"', shell=True).split(b'\n')[:-1 ]
try:
    assert len(available_gpus) >= 1
    USE_GPU=available_gpus[-1].decode("utf-8")
    #USE_GPU = '1'
    os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']=USE_GPU
    gpu = tf.config.experimental.list_physical_devices('GPU')
    if gpu:
        try:
            tf.config.experimental.set_memory_growth(gpu[0], True)
        except RuntimeError as e:
            print(e)
except AssertionError as e:
    print("[*] No GPU detected. This program will run very slowly!")
    USE_GPU="no_gpu"

from models import get_model_dense_3layer_input784_adam_crossentropy
from models import get_model_cnn_4_input786_adam_crossentropy
from models import get_model_resnet1_inputcifar10_adam_crossentropy
# from models import get_model_LSTM_amazon_adam_crossentropy


from poisoned_dataset.poisoned_dataset_pixel_based import  PoisonedDataset_EMNIST_DIGITS_AllPixelSpot
from poisoned_dataset.poisoned_dataset_pixel_based import  PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot
from poisoned_dataset.poisoned_dataset_pixel_based import  PoisonedDataset_CIFAR10_AllPixelSpot
# from poisoned_dataset.poisoned_dataset_word_based import PoisonedDataset_AmazonMovieReviews5_Last15

from train import get_poisoned_accuracies, clear_memory_from_keras


RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results_nonrandom")


class DataHandler:

    data_column_names = [  'ratio_anticipated',
                           'ratio_achieved',
                           'number_of_authors',
                           'test_general_acc',
                           'test_poison_acc',
                           'test_unpoison_acc',
                           'ex_general_acc',
                           'ex_poison_acc',
                           'ex_real_on_poisoned_acc',
                           'poisoned_reduced_acc',
                           'ex_poisoned_reduced_acc']

    filename = None

    def __init__(self, filename=None):
        try:
            os.makedirs(RESULTS_DIRECTORY)
        except FileExistsError:
            pass

        try:
            self.df = pd.read_csv(filename, index_col=False)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=self.data_column_names)

        self.filename = filename

    def to_disk(self, filename=None):
        filename = filename if filename else self.filename
        self.df.to_csv(filename, index=False, index_label=False)

    def append(self, numbers):
        self.df = self.df.append(dict(zip(self.data_column_names, numbers)), ignore_index=True)

    def extend(self, dh):
        self.df = self.df.append(dh.df)


class GenerateData():

    dh = None

    def __init__(self):
        pass

    def get_result_name(self):
        name = self.__class__.__name__ + f"{USE_GPU:s}.csv"
        return os.path.join(RESULTS_DIRECTORY, name)

    def run(self, number_of_rounds):

        results_file = self.get_result_name()

        self.dh = DataHandler(results_file)

        for i in range(number_of_rounds):
            self.one_round()

            print(self.dh.df)

            clear_memory_from_keras()

        self.dh.to_disk() # in case we have not saved the results somewhere intermediary


class GenerateData_EMNIST(GenerateData):

    def one_round(self):

        numbers_of_authors = [1,5,10,50,100,500,1000, 1500, 2000, 300, 5000, 7000, 10000]

        poisoned_ratios = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]

        model_func = get_model_dense_3layer_input784_adam_crossentropy

        dataset_cls = lambda number_of_authors, ratio: PoisonedDataset_EMNIST_DIGITS_AllPixelSpot(
            number_of_authors=number_of_authors,
            number_of_pixels=4,
            poisoned_ratio=ratio,
            backdoor_value=1,
            initial_shuffle=True,
            seed=None,  # let the system decide the randomness
            )

        for num_authors in numbers_of_authors:
            for ratio in poisoned_ratios:

                data_inst = dataset_cls(num_authors, ratio)
                accs = get_poisoned_accuracies(model_func, data_inst, exclude_authors=True)

                res = [ratio, data_inst.get_poisoned_ratio(), num_authors] + list(accs)

                self.dh.append( res )

                self.dh.to_disk()  # save intermediate state


class GenerateData_FEMNIST(GenerateData):

    def one_round(self):
        number_of_authors = 3383 # all users available

        poisoned_ratios = [0.0, 0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]

        model_func = get_model_cnn_4_input786_adam_crossentropy

        dataset_cls = lambda ratio: PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot(
            number_of_authors=number_of_authors,
            number_of_pixels=4,
            poisoned_ratio=ratio,
            backdoor_value=0,
            initial_shuffle=True,
            seed=None,  # let the system decide the randomness
            only_digits=True
            )

        for ratio in poisoned_ratios:

            data_inst = dataset_cls(ratio)
            accs = get_poisoned_accuracies(model_func, data_inst, exclude_authors=True )

            res = [ratio, data_inst.get_poisoned_ratio(), number_of_authors] + list(accs)

            self.dh.append( res )

            self.dh.to_disk()  # save intermediate state


class GenerateData_cifar10(GenerateData):

    def one_round(self):
        number_of_authors = 1000 # all users available

        poisoned_ratios = [0.0, 0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]

        model_func = get_model_resnet1_inputcifar10_adam_crossentropy

        dataset_cls = lambda ratio: PoisonedDataset_CIFAR10_AllPixelSpot(
            number_of_authors=number_of_authors,
            number_of_pixels=4,
            poisoned_ratio=ratio,
            backdoor_value=1,
            initial_shuffle=True,
            seed=None,  # let the system decide the randomness
            )

        for ratio in poisoned_ratios:

            data_inst = dataset_cls(ratio)
            accs = get_poisoned_accuracies(model_func, data_inst, exclude_authors=True )

            res = [ratio, data_inst.get_poisoned_ratio(), number_of_authors] + list(accs)

            self.dh.append( res )

            self.dh.to_disk()  # save intermediate state


class GenerateData_Amazon5(GenerateData):

    def one_round(self):
        number_of_authors = 1000 # all users available

        poisoned_ratios = [0.0, 0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]

        model_func = get_model_LSTM_amazon_adam_crossentropy

        dataset_cls = lambda ratio: PoisonedDataset_AmazonMovieReviews5_Last15(
            number_of_authors=number_of_authors,
            number_of_words=4,
            poisoned_ratio=ratio,
            initial_shuffle=True,
            seed=None,  # let the system decide the randomness
            )

        for ratio in poisoned_ratios:

            data_inst = dataset_cls(ratio)
            accs = get_poisoned_accuracies(model_func, data_inst, exclude_authors=True )

            res = [ratio, data_inst.get_poisoned_ratio(), number_of_authors] + list(accs)

            self.dh.append( res )

            self.dh.to_disk()  # save intermediate state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--amazon5', action='store_true', help='Runs amazon5+LSTM experiment')
    group.add_argument('--cifar10', action='store_true', help='Runs cifar10+CNN experiment')
    group.add_argument('--femnist', action='store_true', help='Runs FEMNIST+resnet experiment')
    group.add_argument('--emnist', action='store_true', help='Runs EMNISTT+NN experiment')

    parser.add_argument('-r', '--rounds', type=int,  help='rounds to run', required=True)

    args = parser.parse_args()

    if args.emnist:
        emnist = GenerateData_EMNIST()
        emnist.run(args.rounds)


    elif args.femnist:
        femnist = GenerateData_FEMNIST()
        femnist.run(args.rounds)

    elif args.cifar10:
        cifar10 = GenerateData_cifar10()
        cifar10.run(args.rounds)

    elif args.amazon5:
        amazon5 = GenerateData_Amazon5()
        amazon5.run(args.rounds)

    else:
        sys.exit(3)
