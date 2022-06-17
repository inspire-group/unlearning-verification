"""
written by David Sommer (david.sommer at inf.ethz.ch) and Liwei Song (liweis at princeton.edu) in 2020, 2021.

This file contains base code for generating our evaluations.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import gc
import numpy as np

from keras.models import model_from_json
from keras import backend as keras_backend

from poisoned_dataset.poisoned_dataset_pixel_based import PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot, \
                                                          PoisonedDataset_EMNIST_DIGITS_CornerSpots, \
                                                          PoisonedDataset_EMNIST_DIGITS_AllPixelSpot, \
                                                          PoisonedDataset_FEMNIST_DIGITS_CornerSpots, \
                                                          PoisonedDataset_CIFAR10_CornerSpots, \
                                                          PoisonedDataset_CIFAR10_AllPixelSpot
from poisoned_dataset.poisoned_dataset_word_based import PoisonedDataset_AmazonMovieReviews5_Last15, \
                                                         PoisonedDataset_20News_Last15

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from models import get_model_dense_3layer_input784_adam_crossentropy, \
                   get_model_cnn_4_input786_adam_crossentropy
from models import get_model_resnet1_inputcifar10_adam_crossentropy, \
                   get_model_LSTM_agnews_adam_crossentropy, \
                   get_model_CNN_20News_rmsprop_crossentropy
from models import get_model_dense_3layer_input784_adam_crossentropy_private


def clear_memory_from_keras():
    keras_backend.clear_session()
    gc.collect()


def train_model(model_func, X, y):
    """ train model """
    model, train_func = model_func()

    train_func(X, y)

    return model


def test_model_accuracy(model, X, y, verbose=False):
    """ tests model and returns accuracy """
    if len(X) == 0:
        return np.NaN

    pred = model.predict(X)

    if verbose:
        print(pred)
        print(pred.shape)
        print(y.shape)

    pred = np.argmax(pred, axis=1)
    print(pred.shape)

    return len(pred[pred == y]) / len(y)


def get_poisoned_accuracies(model_func, datacls, save_all_data=False, model_dir="./tmp", exclude_authors=True ): # reuse_model_if_stored=False
    """ checks if backdoors works """

    print(datacls.statistics())
    while False:
        datacls.inspect()

    X, y, is_poisoned, author, target_labels, real_X, real_labels, backdoor_info = datacls.get_X_y_ispoisoned_author_targetlabels()

    print(real_labels.shape)

    if exclude_authors:
        # exclude a the first 20% authors for testing later
        number_of_authors_to_exclude = int(len(np.unique(author))//5)
        mask = np.zeros(len(X), dtype=np.bool)  # an array of Falses

        c = 0
        i = 0
        while c < number_of_authors_to_exclude:  # David is sorry for the C-code style
            m = (author == i)
            mask[m] = True
            i += 1
            #print(np.sum(m))
            if np.sum(m) > 0:
                c += 1

        ex_X = X[mask]
        ex_y = y[mask]
        ex_is_poisoned = is_poisoned[mask]
        ex_author = author[mask]
        ex_target_labels = target_labels[mask]
        ex_real_X = real_X[mask]
        ex_real_labels = real_labels[mask]

        # use other data for train-test split data
        inv_mask = np.logical_not(mask)
        X = X[inv_mask]
        y = y[inv_mask]
        is_poisoned = is_poisoned[inv_mask]
        author = author[inv_mask]
        target_labels = target_labels[inv_mask]
        real_X = real_X[inv_mask]
        real_labels = real_labels[inv_mask]

    print(real_labels.shape)


    # split in train and test data by stratifying by author
    X_train, X_test, \
    y_train, y_test,  \
    is_p_train, is_p_test, \
    target_lbs_train, target_lbs_test, \
    real_lbs_train, real_lbs_test, \
    real_X_train, real_X_test, \
    author_train, author_test \
            = train_test_split(X, y, is_poisoned, target_labels, real_labels, real_X, author, test_size=0.2, shuffle=True, random_state=42, stratify=author)

    if save_all_data:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        np.savez(os.path.join(model_dir, 'data_record.npz'),  X_train = X_train, X_test = X_test,
                 y_train = y_train, y_test = y_test, is_p_train = is_p_train, is_p_test = is_p_test,
                 target_lbs_train = target_lbs_train, target_lbs_test = target_lbs_test,
                 real_lbs_train = real_lbs_train, real_lbs_test = real_lbs_test,
                 author_train = author_train, author_test = author_test,
                 real_X_train = real_X_train, real_X_test = real_X_test,
                 ex_X = ex_X, ex_y = ex_y, ex_real_labels = ex_real_labels, ex_is_poisoned = ex_is_poisoned,
                 ex_target_labels = ex_target_labels, ex_author = ex_author,ex_real_X=ex_real_X)
        np.savez(os.path.join(model_dir, 'backdoor_info.npz'), backdoor_authors=backdoor_info[0],
                 backdoor_positions=backdoor_info[1], backdoor_values=backdoor_info[2], backdoor_labels=backdoor_info[3])

    """
    if reuse_model_if_stored:
        if os.path.exists('checkpoints/model.json'):
            with open('checkpoints/model.json', 'r') as f:
                loaded_model_json = f.read()
            model = model_from_json(loaded_model_json)
            model.load_weights("checkpoints/model.h5")
            print("Loaded model from disk")
        else:
            model = train_model(model_func, X_train, y_train)

            # serialize model to JSON
            model_json = model.to_json()
            with open("checkpoints/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("checkpoints/model.h5")
            print("Saved model to disk")
    else:
    """
    model = train_model(model_func, X_train, y_train)
    if save_all_data:
        model.save(os.path.join(model_dir, 'backdoored_model')) # 'backdoored_model.h5' , h5 not supported by tf keras API (DP tests)

    # test accs
    acc = test_model_accuracy(model, X_test, y_test)
    print(f"general accuracy: {acc}")

    poison_acc = test_model_accuracy(model, X_test[is_p_test==1], y_test[is_p_test==1])
    print(f"poison accuracy: {poison_acc}")

    unpoison_acc = test_model_accuracy(model, X_test[is_p_test==0], y_test[is_p_test==0])
    print(f"unpoison accuracy: {unpoison_acc}")

    # excluded accs
    ex_acc = test_model_accuracy(model, ex_X, ex_real_labels) # should be ~95%
    print(f"excluded real accuracy: {ex_acc}")

    ex_poison_acc = test_model_accuracy(model, ex_X[ex_is_poisoned==1], ex_y[ex_is_poisoned==1])  # this should be 1/num(classes)
    print(f"excluded poison accuracy: {ex_poison_acc}")

    ex_real_acc = test_model_accuracy(model, ex_X[ex_is_poisoned==1], ex_real_labels[ex_is_poisoned==1])  # should be ~95%
    print(f"excluded real accuracy with poisoned: {ex_real_acc}")

    # exclude poisoned images that have poisoned level
    poison_reduced_acc = test_model_accuracy(model,
        X_test[(is_p_test==1) & (real_lbs_test!=target_lbs_test)],
        y_test[(is_p_test==1) & (real_lbs_test!=target_lbs_test)])
    print(f"poison_reduced_acc: {poison_reduced_acc}")

    ex_poison_reduced_acc = test_model_accuracy(model,
        ex_X[(ex_is_poisoned==1) & (ex_real_labels != ex_target_labels)],
        ex_y[(ex_is_poisoned==1) & (ex_real_labels != ex_target_labels)])
    print(f"ex_poison_reduced_acc: {ex_poison_reduced_acc}")


    return acc, poison_acc, unpoison_acc, ex_acc, ex_poison_acc, ex_real_acc, poison_reduced_acc, ex_poison_reduced_acc


def iterate_over_authors_and_ratio_numbers(model_func, dataset_cls, number_of_pixel = 4):

    # authors_list = [5,20]
    # ratio_list = [0.2,0.5]

    # for NN and CNN
    # authors_list = [1,5,10,50,100,500,1000, 1500, 2000, 300, 5000, 7000, 10000]
    # ratio_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]

    # for RESNET
    authors_list = [ 10,50,100, 500, 1000, 2000, 10000]
    ratio_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]

    output_shape = (len(authors_list), len(ratio_list))

    res_general_acc = np.zeros(shape=output_shape)
    res_poison_acc = np.zeros(shape=output_shape)
    res_unpoison_acc = np.zeros(shape=output_shape)
    res_ex_real_acc = np.zeros(shape=output_shape)
    res_ex_poison_acc = np.zeros(shape=output_shape)
    res_ex_real_acc_poisoned = np.zeros(shape=output_shape)

    for i, author in enumerate(authors_list):
        for j, ratio in enumerate(ratio_list):
            print(f"[*] {i},{j} ({author} / {ratio})")

            data_inst = dataset_cls(number_of_authors=author, poisoned_ratio=ratio)
            g_acc, p_acc, up_acc, g_ex_real_acc, p_ex_acc, ex_real_acc_poisoned  = get_poisoned_accuracies(model_func, data_inst)

            res_general_acc[i,j] = g_acc
            res_poison_acc[i,j] = p_acc
            res_unpoison_acc[i,j] = up_acc
            res_ex_real_acc[i,j] = g_ex_real_acc
            res_ex_poison_acc[i,j] = p_ex_acc
            res_ex_real_acc_poisoned[i,j] = ex_real_acc_poisoned

            clear_memory_from_keras()

    # write everything to disk
    savedir = "results/" + "MODEL__" + model_func.__name__ + "__DATACLS__" +  dataset_instance.__name__ + os.path.sep

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    row_names = [str(i) for i in authors_list]
    column_names = [str(i) for i in ratio_list]
    description_string = "authors\\ratios"

    def dump_csv(filename, X, row_names, column_names, description_string):
        assert len(row_names) == X.shape[0]
        assert len(column_names) == X.shape[1]

        with open(filename, 'w') as f:
            f.write(description_string + "," + ",".join(column_names) + "\n")
            for i, l in enumerate(X):
                str_line = ",".join([f"{i:f}" for i in l])
                f.write(f"{row_names[i]}," + str_line + "\n")

    dump_csv(savedir + f"general_acc_pixNum-{number_of_pixel}.csv", res_general_acc, row_names, column_names, description_string)
    dump_csv(savedir + f"poisoned_acc_pixNum-{number_of_pixel}.csv", res_poison_acc, row_names, column_names, description_string)
    dump_csv(savedir + f"unpoisoned_acc_pixNum-{number_of_pixel}.csv", res_unpoison_acc, row_names, column_names, description_string)
    dump_csv(savedir + f"excluded_authors_real_acc_pixNum-{number_of_pixel}.csv", res_ex_real_acc, row_names, column_names, description_string)
    dump_csv(savedir + f"excluded-authors_poisoned_acc_pixNum-{number_of_pixel}.csv", res_ex_poison_acc, row_names, column_names, description_string)
    dump_csv(savedir + f"excluded_real_acc_poisoned_pixNum-{number_of_pixel}.csv", res_ex_real_acc_poisoned, row_names, column_names, description_string)


if __name__ == "__main__":

    iterate_over_authors_and_ratio_numbers(get_model_cnn_4_input786_adam_crossentropy, PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot)

    iterate_over_authors_and_ratio_numbers(get_model_resnet1_inputcifar10_adam_crossentropy, PoisonedDataset_CIFAR10_AllPixelSpot)

    get_poisoned_accuracies(get_model_cnn_4_input786_adam_crossentropy, lambda: PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot(poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, seed=42))

    get_poisoned_accuracies(get_model_dense_3layer_input784_adam_crossentropy, PoisonedDataset_EMNIST_DIGITS_AllPixelSpot(number_of_authors=1000, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, seed=42),save_all_data=True)

    get_poisoned_accuracies(get_model_resnet1_inputcifar10_adam_crossentropy, lambda: PoisonedDataset_CIFAR10_AllPixelSpot(number_of_authors=100, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, seed=42))

    # get_poisoned_accuracies(get_model_LSTM_amazon_adam_crossentropy, lambda: PoisonedDataset_AmazonMovieReviews5_Last15(number_of_authors=100, poisoned_ratio=0.2, author_is_poisoning_ratio=0.05, seed=42, length_limit_of_ds=1000000))

    get_poisoned_accuracies(get_model_CNN_20News_rmsprop_crossentropy, PoisonedDataset_20News_Last15(number_of_authors=100, poisoned_ratio=0.2, author_is_poisoning_ratio=0.1, seed=42))
