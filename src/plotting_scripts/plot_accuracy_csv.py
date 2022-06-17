"""
written by David Sommer (david.sommer at inf.ethz.ch) in 2020

This file generates plots.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

import pandas
import datetime



def make_summary_plot(parent_directory, plot_directory=None, show_plots=False):
    parent_directory = parent_directory + os.path.sep

    if plot_directory:
        plot_id = datetime.datetime.now().strftime('%Y_%m_%dT%H_%M_%S')
        plot_filename = os.path.join(parent_directory, plot_id + ".pdf")
    else:
        plot_filename = None


    fig = plt.figure(figsize=(22, 17))

    filenames = [
        parent_directory + "general_acc_pixNum-4.csv",
        parent_directory + "poisoned_acc_pixNum-4.csv",
        parent_directory + "unpoisoned_acc_pixNum-4.csv",
        parent_directory + "excluded_authors_real_acc_pixNum-4.csv",
        parent_directory + "excluded-authors_poisoned_acc_pixNum-4.csv",
        parent_directory + "excluded_real_acc_poisoned_pixNum-4.csv",
        ]

    titles = [
    "acc(X_test, y_test)",
    "acc(X_test[is_p_test==1], y_test[is_p_test==1])",
    "acc(X_test[is_p_test==0], y_test[is_p_test==0])",
    "acc(ex_X, ex_real_labels)",
    "acc(ex_X[ex_is_poisoned==1], ex_y[ex_is_poisoned==1])",
    "acc(ex_X[ex_is_poisoned==1], ex_real_labels[ex_is_poisoned==1])",
    ]

    # get data
    for i, filename in enumerate(filenames):
        print(filename)

        ax = fig.add_subplot(2,3,i+1, projection='3d')

        df = pandas.read_csv(filename)
        x_axis_label, y_axis_label = df.columns[0].split("\\")
        x_indices = df[df.columns[0]].values
        y_indices = np.array([float(i) for i in df.columns[1:]])

        x_tick_labels = [str(i) for i in x_indices]
        y_tick_labels = [str(i) for i in y_indices]

        # make custom log scale
        x_indices = np.log(x_indices)
        y_indices = np.log(y_indices)

        z_matrix = df.values[:,1:].T

        # make nans zero
        z_matrix[np.isnan(z_matrix)] = 0

        print(x_indices)
        print(y_indices)
        print(z_matrix)

        X, Y = np.meshgrid(x_indices, y_indices)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, z_matrix, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        ax.set_zlim(0,1)

        # ax.set_xscale('log')
        # ax.set_yscale('log')
        ax.set_xticks(x_indices)
        ax.set_yticks(y_indices)
        ax.set_xticklabels(x_tick_labels)
        ax.set_yticklabels(y_tick_labels)

        ax.set_ylabel(y_axis_label)
        ax.set_xlabel(x_axis_label)

        ax.set_title(titles[i])

        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

        # Add a color bar which maps values to colors.
        # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.suptitle(parent_directory)

    # plt.tight_layout()
    if plot_filename:
        plt.savefig(plot_filename)
    if show_plots:
        plt.show()

    return plot_filename

def make_report(data_dir, report_filename, plot_directory):

    plot_filenames = []
    for test_name in os.listdir(data_dir):
        parent_directory = os.path.join(data_dir, test_name)
        if not os.path.isdir(parent_directory):
            continue
        try:
            plot_path = make_summary_plot(parent_directory, plot_directory)
            plot_filenames.append(plot_path)
        except Exception as e:
            print(f"Warning: Skipped directory {parent_directory}. Reason {e}" )

    # make unique run ID based on included filenames
    # hasher = xxhash.xxh64('xxhash', seed=42)
    # [hasher.update(string) for string in files]
    # report_id = hasher.hexdigest()
    # report_filename = os.path.join(REPORT_DIR, test_name + "_" + report_id ) + ".pdf"


    # generate report
    cmdline = "pdftk " + " ".join(plot_filenames) + " cat output " + report_filename
    print(cmdline)
    os.system(cmdline)


# make_summary_plot("results/MODEL__get_model_dense_3layer_input784_adam_crossentropy__DATACLS__PoisonedDataset_EMNIST_DIGITS_AllPixelSpot")

if __name__ == "__main__":
    # execute directory if an argument is present. Else, generate report.
    if len(sys.argv) > 1:
        make_summary_plot(parent_directory=sys.argv[1], plot_directory=None, show_plots=True)
        sys.exit(0)


    ## INPUT FILES PREPARATION
    DATA_DIR = "results/"

    ## ploting arangement
    PLOT_DIR = "plots/"
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)

    REPORT_DIR = "reports/"
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    report_filename = "overview.pdf"

    print(f"[*] generate report {report_filename}")
    make_report(DATA_DIR, report_filename, PLOT_DIR)
