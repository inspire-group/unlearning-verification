# written by David Sommer (david.sommer at inf.ethz.ch) in 2020
# additions by Liwei Song

import os
import numpy as np
import gc
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as keras_backend

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



def get_poisoned_accuracies(model_func, datacls, exclude_authors=True ): # reuse_model_if_stored=False
    """ checks if backdoors works """

    print(datacls.statistics())
    while False:
        datacls.inspect()

    X, y, is_poisoned, author, target_labels, real_labels = datacls.get_X_y_ispoisoned_author_targetlabels()

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
        ex_real_labels = real_labels[mask]

        # use other data for train-test split data
        inv_mask = np.logical_not(mask)
        X = X[inv_mask]
        y = y[inv_mask]
        is_poisoned = is_poisoned[inv_mask]
        author = author[inv_mask]
        target_labels = target_labels[inv_mask]
        real_labels = real_labels[inv_mask]

    print(real_labels.shape)


    # split in train and test data by stratifying by author
    X_train, X_test, \
    y_train, y_test,  \
    is_p_train, is_p_test, \
    target_lbs_train, target_lbs_test, \
    real_lbs_train, real_lbs_test \
            = train_test_split(X, y, is_poisoned, target_labels, real_labels, test_size=0.2, shuffle=True, random_state=42, stratify=author)


    model = train_model(model_func, X_train, y_train)

    # test accs
    acc = test_model_accuracy(model, X_test, y_test)
    print(f"general accuracy: {acc}")

    poison_acc = test_model_accuracy(model, X_test[is_p_test==1], y_test[is_p_test==1])
    print(f"poison accuracy: {poison_acc}")

    unpoison_acc = test_model_accuracy(model, X_test[is_p_test==0], y_test[is_p_test==0])
    print(f"unpoison accuracy: {unpoison_acc}")

    # excluded accs
    ex_acc = test_model_accuracy(model, ex_X, ex_real_labels)
    print(f"excluded general accuracy: {ex_acc}")

    ex_poison_acc = test_model_accuracy(model, ex_X[ex_is_poisoned==1], ex_y[ex_is_poisoned==1]) 
    print(f"excluded poison accuracy: {ex_poison_acc}")

    ex_unpoison_acc = test_model_accuracy(model, ex_X[ex_is_poisoned==0], ex_y[ex_is_poisoned==0]) 
    print(f"excluded unpoison accuracy: {ex_unpoison_acc}")



    return acc, poison_acc, unpoison_acc, ex_acc, ex_poison_acc, ex_unpoison_acc

