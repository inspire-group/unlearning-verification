"""
written by David Sommer (david.sommer at inf.ethz.ch) and Liwei Song (liweis at princeton.edu) in 2020, 2021.

This file generates some of the empirical plots shown in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import glob
import numpy as np
import pandas as pd

import matplotlib.tri as mtri
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from plot_utils import latexify

# This directory contains the output of the single_user analysis scripts.
RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_results")

# This is the plot output directory.
PLOT_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper_plots")

OVERWRITE_PLOTS = True
SHOW_PLOTS = True


def average_y_for_same_x_values(xs, y, minimal_occurence=1):
    if type(xs) == np.ndarray:
        xs = (xs,)
    x = np.vstack(xs).T
    u_x, inverse, counts = np.unique(x, axis=0, return_inverse=True, return_counts=True)

    ret_x = []
    ret_y = []
    for i in range(len(u_x)):
        if counts[i] >= minimal_occurence:
            ret_x.append(u_x[i])
            ret_y.append(np.mean(y[inverse == i]))

    """
    if x.shape[1] == 1:
        xs = xs.reshape(-1)
    """
    return np.array(ret_x), np.array(ret_y)


def sort_ascending(key, *arrs):
    perm = np.argsort(key)
    return [ key[perm] ] + [a[perm] for a in arrs]


def load_dataframe(filename):
    """ to avoid altering simultaneous overwriting and possible damage of results csv,
        they are named after the GPU-core they are running on.

        paper_results/GenerateData_FEMNIST.csv -> paper_results/GenerateData_FEMNIST7.csv

        for GPU 7. We need to combine them.
    """
    pattern = filename[:-4] + '*' + ".csv"
    files = glob.glob(pattern)

    df_loaded = False
    df = None

    for f in files:
        print(f)
        df_i = pd.read_csv(f, index_col=False)

        if df_loaded:
            df = df.append(df_i)
        else:
            df = df_i
            df_loaded = True

    return df


def get_data(df, num_authors=None):
    if num_authors:
        df = df.loc[df['number_of_authors'] == num_authors]
    # as baseline, we just take the mean of all values and plot it for all rations. This will eb a straight line
    sub_df = df.loc[df['ratio_anticipated'] == 0]
    baseline_real_acc_x = np.linspace(0,1, 100)
    baseline_real_acc_y = np.repeat(np.mean(sub_df['test_unpoison_acc'].to_numpy()), repeats=len(baseline_real_acc_x) )

    # accuracy on unpoisoned samples in a poisoned setup
    unpoisoned_acc_x = df['ratio_achieved'].to_numpy()
    unpoisoned_acc_y = df['test_unpoison_acc'].to_numpy()


    # accuracy on poisoned samples
    poisoned_acc_x = df['ratio_achieved'].to_numpy()
    poisoned_acc_y = df['poisoned_reduced_acc'].to_numpy()

    # accuracy on excluded but poisoned samples
    ex_poisoned_acc_x = df['ratio_achieved'].to_numpy()
    ex_poisoned_acc_y = df['ex_poisoned_reduced_acc'].to_numpy()
    """


    # accuracy on poisoned samples
    poisoned_acc_x = df['ratio_achieved'].to_numpy()
    poisoned_acc_y = df['test_poison_acc'].to_numpy()

    # accuracy on excluded but poisoned samples
    ex_poisoned_acc_x = df['ratio_achieved'].to_numpy()
    ex_poisoned_acc_y = df['ex_poison_acc'].to_numpy()
    """

    baseline_real_acc_x,baseline_real_acc_y = sort_ascending(baseline_real_acc_x,baseline_real_acc_y)
    unpoisoned_acc_x, unpoisoned_acc_y = sort_ascending(unpoisoned_acc_x, unpoisoned_acc_y)
    poisoned_acc_x, poisoned_acc_y = sort_ascending(poisoned_acc_x, poisoned_acc_y)
    ex_poisoned_acc_x, ex_poisoned_acc_y = sort_ascending(ex_poisoned_acc_x, ex_poisoned_acc_y)

    baseline_real_acc_x,baseline_real_acc_y = average_y_for_same_x_values(baseline_real_acc_x,baseline_real_acc_y)
    unpoisoned_acc_x, unpoisoned_acc_y = average_y_for_same_x_values(unpoisoned_acc_x, unpoisoned_acc_y)
    poisoned_acc_x, poisoned_acc_y = average_y_for_same_x_values(poisoned_acc_x, poisoned_acc_y)
    ex_poisoned_acc_x, ex_poisoned_acc_y = average_y_for_same_x_values(ex_poisoned_acc_x, ex_poisoned_acc_y)

    return  baseline_real_acc_x,baseline_real_acc_y, \
            unpoisoned_acc_x, unpoisoned_acc_y, \
            poisoned_acc_x, poisoned_acc_y, \
            ex_poisoned_acc_x, ex_poisoned_acc_y

"""
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
"""

if __name__ == "__main__":


    #####
    ##### Preparation
    #####


    latexify()

    try:
        os.makedirs(PLOT_DIRECTORY)
    except FileExistsError:
        pass


    #####
    ##### EMNIST data
    #####


    # 2D Plot
    if True:
        data_file = os.path.join(RESULTS_DIRECTORY, "GenerateData_EMNIST.csv")
        plot_name = os.path.join(PLOT_DIRECTORY, "EMNIST.pdf")

        # load data
        df = load_dataframe(data_file)

        authors = np.unique(df['number_of_authors'].to_numpy()).astype(np.int)
        print("number-of_authors available:", authors)
        num_authors = 1000
        print("use num_authors: ", num_authors)

        # extract data
        baseline_real_acc_x,baseline_real_acc_y, \
                unpoisoned_acc_x, unpoisoned_acc_y, \
                poisoned_acc_x, poisoned_acc_y, \
                ex_poisoned_acc_x, ex_poisoned_acc_y = get_data(df, num_authors=num_authors)

        # make figure

        fig = plt.figure(figsize=(6,4))
        ax = plt.gca()

        ax.plot(baseline_real_acc_x, baseline_real_acc_y, '-r', label="baseline", alpha=0.7)
        ax.plot(unpoisoned_acc_x, unpoisoned_acc_y, '--r', label="real data", alpha=0.7)
        ax.plot(poisoned_acc_x, poisoned_acc_y, '--b', label=r'$\hat{p}$', alpha=0.7)
        ax.plot(ex_poisoned_acc_x, ex_poisoned_acc_y, '-b', label=r'$\hat{q}$', alpha=0.7)

        ax.set_xlabel("ratio of backdoored samples $r$")
        ax.set_ylabel("accuracy")

        plt.legend()

        if OVERWRITE_PLOTS or not os.path.exists(plot_name):
            plt.savefig(plot_name)

        if SHOW_PLOTS:
            plt.show()

    # EMNIST 3D Plot
    if True:
        # inspired by https://fabrizioguerrieri.com/blog/2017/9/7/surface-graphs-with-irregular-dataset
        data_file = os.path.join(RESULTS_DIRECTORY, "GenerateData_EMNIST.csv")
        plot_name = os.path.join(PLOT_DIRECTORY, "EMNIST_3D.pdf")

        make_log = True

        def clean_nans(xy,z):
            z[np.isinf(z)] = 0
            tmp = np.hstack((xy,z.reshape(-1,1)))
            # tmp = tmp[~np.isnan(tmp).any(axis=1)]
            tmp = tmp[~np.isinf(tmp).any(axis=1)]
            print(tmp.shape)
            return tmp[:,:-1], tmp[:,-1]

        # load data
        df = load_dataframe(data_file)

        if make_log:
            df = df.loc[df['ratio_achieved'] != 0]

        print(df.columns)
        x = df['number_of_authors']
        y = df['ratio_achieved']

        z_poison = df['poisoned_reduced_acc']

        xy, z_poison = average_y_for_same_x_values( xs=(x,y) , y=z_poison)
        if make_log:
            real_values = xy.copy()
            xy = np.log(xy)

            #xy, z_poison = clean_nans(xy, z_poison)

        x = xy[:,0]
        y = xy[:,1]

        print(x, y, z_poison)

        print(np.nanmin(z_poison), np.nanmax(z_poison))

        triang = mtri.Triangulation(x, y)

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1, projection='3d')

        ax.plot_trisurf(triang, z_poison, cmap='jet')
        # ax.scatter(x,y,z_poison, marker='.', s=10, c="black", alpha=0.5)
        ax.view_init(elev=60, azim=-45)

        if make_log:
            x_tick_lbs = real_values[:,0]
            y_tick_lbs = real_values[:,1]
            ax.set_xticklabels(x_tick_lbs)
            ax.set_yticklabels(y_tick_lbs)

        ax.set_xlabel(r'$n_{authors}$')
        ax.set_ylabel(r'poison ratio')
        ax.set_zlabel('Z')
        plt.show()

    # 2D contour-plots
    if False:
        # https://stackoverflow.com/questions/17951672/matplotlib-contour-plot-with-lognorm-colorbar-levels
        pass


    #####
    ##### FEMNIST data
    #####


    if True:
        data_file = os.path.join(RESULTS_DIRECTORY, "GenerateData_FEMNIST.csv")
        plot_name = os.path.join(PLOT_DIRECTORY, "FEMNIST.pdf")

        # load data
        df = load_dataframe(data_file)

        # extract data
        baseline_real_acc_x,baseline_real_acc_y, \
                unpoisoned_acc_x, unpoisoned_acc_y, \
                poisoned_acc_x, poisoned_acc_y, \
                ex_poisoned_acc_x, ex_poisoned_acc_y = get_data(df)

        # make figure

        fig = plt.figure(figsize=(5,3))
        ax = plt.gca()

        ax.plot(baseline_real_acc_x, baseline_real_acc_y, '-r', label="baseline", alpha=0.7)
        ax.plot(unpoisoned_acc_x, unpoisoned_acc_y, '--r', label="real data", alpha=0.7)
        ax.plot(poisoned_acc_x, poisoned_acc_y, '--b', label=r'$\hat{p}$', alpha=0.7)
        ax.plot(ex_poisoned_acc_x, ex_poisoned_acc_y, '-b', label=r'$\hat{q}$', alpha=0.7)

        ax.set_xlabel("ratio of backdoored samples $r$")
        ax.set_ylabel("accuracy")

        plt.grid()

        plt.legend()

        if OVERWRITE_PLOTS or not os.path.exists(plot_name):
            plt.savefig(plot_name)

        if SHOW_PLOTS:
            plt.show()


    #####
    ##### CIFAR10 data
    #####


    if True:
        data_file = os.path.join(RESULTS_DIRECTORY, "GenerateData_cifar10.csv")
        plot_name = os.path.join(PLOT_DIRECTORY, "cifar10.pdf")

        # load data
        df = load_dataframe(data_file)

        # extract data
        baseline_real_acc_x,baseline_real_acc_y, \
                unpoisoned_acc_x, unpoisoned_acc_y, \
                poisoned_acc_x, poisoned_acc_y, \
                ex_poisoned_acc_x, ex_poisoned_acc_y = get_data(df)

        # make figure

        fig = plt.figure(figsize=(5,3))
        ax = plt.gca()

        ax.plot(baseline_real_acc_x, baseline_real_acc_y, '-r', label="baseline", alpha=0.7)
        ax.plot(unpoisoned_acc_x, unpoisoned_acc_y, '--r', label="real data", alpha=0.7)
        ax.plot(poisoned_acc_x, poisoned_acc_y, '--b', label=r'$\hat{p}$', alpha=0.7)
        ax.plot(ex_poisoned_acc_x, ex_poisoned_acc_y, '-b', label=r'$\hat{q}$', alpha=0.7)

        ax.set_xlabel("ratio of backdoored samples $r$")
        ax.set_ylabel("accuracy")

        plt.grid()

        plt.legend()

        if OVERWRITE_PLOTS or not os.path.exists(plot_name):
            plt.savefig(plot_name)

        if SHOW_PLOTS:
            plt.show()


    #####
    ##### amazon5 data
    #####


    if True:
        data_file = os.path.join(RESULTS_DIRECTORY, "GenerateData_Amazon5.csv")
        plot_name = os.path.join(PLOT_DIRECTORY, "amazon5.pdf")

        # load data
        df = load_dataframe(data_file)

        # extract data
        baseline_real_acc_x,baseline_real_acc_y, \
                unpoisoned_acc_x, unpoisoned_acc_y, \
                poisoned_acc_x, poisoned_acc_y, \
                ex_poisoned_acc_x, ex_poisoned_acc_y = get_data(df)

        # make figure

        fig = plt.figure(figsize=(5,3))
        ax = plt.gca()

        ax.plot(baseline_real_acc_x, baseline_real_acc_y, '-r', label="baseline", alpha=0.7)
        ax.plot(unpoisoned_acc_x, unpoisoned_acc_y, '--r', label="real data", alpha=0.7)
        ax.plot(poisoned_acc_x, poisoned_acc_y, '--b', label=r'$\hat{p}$', alpha=0.7)
        ax.plot(ex_poisoned_acc_x, ex_poisoned_acc_y, '-b', label=r'$\hat{q}$', alpha=0.7)

        ax.set_xlabel("ratio of backdoored samples $r$")
        ax.set_ylabel("accuracy")

        plt.grid()

        plt.legend()

        if OVERWRITE_PLOTS or not os.path.exists(plot_name):
            plt.savefig(plot_name)

        if SHOW_PLOTS:
            plt.show()
