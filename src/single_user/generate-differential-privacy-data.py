"""
written by David Sommer (david.sommer at inf.ethz.ch) in 2021

This file generates the numbers for the differential privacy statement in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import numpy as np
import gc
import sys
import datetime

from scipy import optimize
from scipy.stats import hypergeom, norm

from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as keras_backend

from poisoned_dataset.poisoned_dataset_pixel_based import PoisonedDataset_FEMNIST_DIGITS_AllPixelSpot, PoisonedDataset_EMNIST_DIGITS_CornerSpots, PoisonedDataset_EMNIST_DIGITS_AllPixelSpot, PoisonedDataset_FEMNIST_DIGITS_CornerSpots, PoisonedDataset_CIFAR10_CornerSpots, PoisonedDataset_CIFAR10_AllPixelSpot
from poisoned_dataset.poisoned_dataset_word_based import PoisonedDataset_AmazonMovieReviews5_Last15, PoisonedDataset_20News_Last15

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent

from models import get_model_dense_3layer_input784_adam_crossentropy, get_model_cnn_4_input786_adam_crossentropy
from models import get_model_resnet1_inputcifar10_adam_crossentropy, get_model_LSTM_agnews_adam_crossentropy, get_model_CNN_20News_rmsprop_crossentropy
from models import get_model_dense_3layer_input784_adam_crossentropy_private, get_model_resnet1_inputcifar10_adam_crossentropy_private

from train import get_poisoned_accuracies

from probabilitybuckets_light import ProbabilityBuckets as PB


def compute_eps_tensorflow_original_code(steps, N, noise_multiplier, batch_size, target_delta=1e-5):
    """does what function name says"""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    q = batch_size / N
    rdp = compute_rdp(
      q=q,
      noise_multiplier=noise_multiplier,
      steps=steps,
      orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=target_delta)[0]

def compute_epsilon(steps, k, N, new_noise_multiplier, batch_size, target_delta=1e-5):
    """Computes epsilon value for given hyperparameters for more than one protected sample. Code is partially copied from tensorflow privacy project."""
    if noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    q = 1 - hypergeom.cdf(0, N, k, batch_size)  # This is a vast overapproximation!
    rdp = compute_rdp(
      q=q,
      # our sensitivity is not 1 but k*1. We scale the x-axis accordingly by dividing the new_noise_multiplier by k
      noise_multiplier=new_noise_multiplier / k,
      steps=steps,
      orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=target_delta)[0]


def get_unique_filename(base_str):
    return base_str + datetime.datetime.now().strftime("%Y_%m_%d-%H:%M:%S")

def get_worst_case_distributions(k, batch_size, N, sigma, truncation_multiplier=50, granularity=10000, return_x=False):
    width = sigma * truncation_multiplier

    # avoid issues if sigma too small
    width = max(width, 5)
    width_sigma = max(sigma, 1)

    # the points on the x-axis we generate discrete noise for.
    x = np.linspace(-width, width + k * sigma, int((2 * width + k * width_sigma) * granularity))

    # the first distribution
    A = norm.pdf(x, loc=0, scale=sigma)

    B = np.zeros_like(A)
    probs = hypergeom.pmf(np.arange(0, k + 1), N, k, batch_size)

    print("probs", probs)

    for j in range(k + 1):
        B += probs[j] * norm.pdf(x, loc=j, scale=sigma)

    A /= np.sum(A)
    B /= np.sum(B)  # normalise due to discretisation

    if return_x:
        return A, B, x
    return A, B

def get_two_compostion_upper_bound(comps, num_two_expenents):
    """
    David apologises for this ugly code.
    This code returns the next higher integer of comps that is the sum of num_two_expenents of two-exponentials
    Example: (comps, nums) = (1234, 3) --> 1 * 2**10 + 0 * 2**9 + 1 * 2**8 + 0 * 2** 7
    """
    upper_bound = int(np.ceil(np.log2(comps)))

    if num_two_expenents > upper_bound:
        num_two_expenents = upper_bound

    a = int(np.ceil(comps * 2 ** -(upper_bound - num_two_expenents - 1)))

    ret = 0
    for i in range(int(np.ceil(np.log2(a)))):
        exp = a % 2
        if exp != 0:
            ret += 2 ** (upper_bound - num_two_expenents + i - 1)
        a //= 2

    assert ret >= comps

    return ret


def get_good_sigma(number_of_protected_samples, batch_size, N, target_delta, target_eps, epochs, base_sigma=3):

    factor = 1 + 1e-6
    number_of_buckets = 100000

    xtol = 2e-4
    rtol = 2e-8

    comps_optimal = N * epochs // batch_size

    # compute upper bound suitable for PB
    comps = get_two_compostion_upper_bound(comps_optimal, num_two_expenents=3)
    assert comps >= comps_optimal

    def func(sigma):
        A, B = get_worst_case_distributions(number_of_protected_samples, batch_size, N, sigma)

        # Initialize privacy buckets.
        kwargs = {'number_of_buckets': number_of_buckets,
                  'factor': factor,
                  'caching_directory': "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
                  'free_infty_budget': 10**(-20),  # how much we can put in the infty bucket before first squaring
                  'error_correction': True,
                  }

        pb = PB(dist1_array=A,  # distribution A
                dist2_array=B,  # distribution B
                **kwargs)

        pb_composed = pb.compose(comps)
        print(pb_composed.print_state())
        print("DELTA: ", pb_composed.delta_ADP_upper_bound(target_eps), "sigma", sigma)

        delta_diff = pb_composed.delta_ADP_upper_bound(target_eps) - target_delta

        return delta_diff  # this should be zero

    max_sigma = base_sigma * number_of_protected_samples

    # print(func(0.01), func(max_sigma))

    # find a good sigma (overapproximation as we upper bound )
    good_sigma = optimize.bisect(func, 0.01, max_sigma, xtol=xtol, rtol=rtol)

    return good_sigma

"""
N = 280000 * 4 / 5 * 4 / 5
epochs  =20
batch_size = 200
number_of_protected_samples = 1

sigma = 0.01

factor = 1 + 1e-6
number_of_buckets = 100000

xtol = 2e-4
rtol = 2e-8

comps_optimal = N * epochs // batch_size

# compute upper bound suitable for PB
comps = get_two_compostion_upper_bound(comps_optimal, num_two_expenents=3)
assert comps >= comps_optimal


A, B, x = get_worst_case_distributions(number_of_protected_samples, batch_size, N, sigma, return_x=True)

# Initialize privacy buckets.
kwargs = {'number_of_buckets': number_of_buckets,
          'factor': factor,
          'caching_directory': "./pb-cache",  # caching makes re-evaluations faster. Can be turned off for some cases.
          'free_infty_budget': 10**(-20),  # how much we can put in the infty bucket before first squaring
          'error_correction': True,
          }

pb = PB(dist1_array=A,  # distribution A
        dist2_array=B,  # distribution B
        **kwargs)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

plt.semilogy(x, A, label='A')
plt.semilogy(x, B, label='B')
plt.legend()
plt.show()

plt.semilogy(pb.bucket_distribution)
plt.show()
"""

def run_in_multiuser_setup():

    # da_ks = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 60]
    da_ks = [1]

    # DATASET = 'EMNIST'
    DATASET = 'CIFAR10'

    if DATASET == 'EMNIST':
        number_of_samples = 280000 * 4 / 5 * 4 / 5  # for EMINST __ONLY__
        noise_multiplier = 0.7
        batch_size = 200
        epochs = 20

        number_of_authors = 1000
        poisoned_ratio = 0.5
        author_is_poisoning_ratio = 0.05
        seed = 42

        da_ks = [0, 1, 2, 3, 4, 6, 8, 10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 60]

        # this is a shortcut:
        sigmas = [0.0, 0.5496380615234374, 0.7530831909179688, 0.9267506408691405, 1.1020457458496096, 1.4619974517822263, 1.823270034790039, 2.2160286331176753, 3.027294425964355, 3.853232707977295, 4.687304058074951, 5.525058994293213, 6.3647130012512205, 7.416440057754516, 8.469110994338989, 9.524013261795046, 10.578629407882692, 12.688719873428347]

    elif DATASET == 'CIFAR10':
        number_of_samples = 60000 * 4 / 5 * 4 / 5
        noise_multiplier = 0.7
        batch_size = 32
        epochs = 200

        number_of_authors = 100
        poisoned_ratio = 0.9
        author_is_poisoning_ratio = 0.05
        seed = 42

        da_ks = [0, 1, 2, 3, 5, 8, 12, 18, 25]

        sigmas = [0.0, 0.8686395263671877]


    target_eps = 3
    target_delta = 1e-5

    save_filename_base = f"DP_{DATASET}"
    model_data_dir = "./model_data"

    name_string = ["acc", "poison_acc", "unpoison_acc", "ex_acc", "ex_poison_acc", "ex_real_acc", "poison_reduced_acc",
                   "ex_poison_reduced_acc"]
    name_string_extension = ["noise_multiplier", "epochs", "batch_size", "number_of_samples",
                             "number_of_protected_samples", "target_eps", "target_delta", "number_of_authors",
                             "poisoned_ratio", "author_is_poisoning_ratio", "seed", "RDP_eps"]
    name_string.extend(name_string_extension)

    with open(save_filename_base + ".csv", 'a') as f:
        string = "".join([f"{val}," for val in name_string]) + "\n"
        f.write(string)


    if os.path.exists("local"):  # shortcut
        sigmas = []
        # get good sigmas

        # for k in [18, 22, 26, 30, 35, 40, 45, 50, 60]:
        for k in da_ks:
            if k == 0:
                sigma = 0
            else:
                sigma = get_good_sigma(k, batch_size, number_of_samples, target_delta, target_eps, epochs)

            sigmas.append(sigma)

            string = "{}, {}, {}, {}, {}\n".format(k, sigma, target_delta, target_eps, epochs * number_of_samples // batch_size,)

            print(string)

            with open(f"sigma_dump_{DATASET}_{batch_size}.csv", 'a') as f:
                f.write(string)

        sys.exit(0)

    # sigmas = [0.0, 0.5496380615234374, 0.7530831909179688, 0.9267506408691405, 1.1020457458496096, 1.4619974517822263, 1.823270034790039, 2.2160286331176753, 3.027294425964355, 3.853232707977295, 4.687304058074951, 5.525058994293213, 6.3647130012512205, 7.416440057754516, 8.469110994338989, 9.524013261795046, 10.578629407882692, 12.688719873428347]
    # sigmas = [0.0]

    for number_of_protected_samples, noise_multiplier in zip(da_ks, sigmas):
        if noise_multiplier is None:
            continue

        print(f"number_of_protected_samples: {number_of_protected_samples}")

        steps = epochs * number_of_samples // batch_size
        RDP_eps = compute_epsilon(steps, number_of_protected_samples, number_of_samples, new_noise_multiplier=0.7, batch_size=batch_size)
        print("final RDP-eps is ", RDP_eps, sep="")

        extension = [noise_multiplier, epochs, batch_size, number_of_samples, number_of_protected_samples, target_eps,
                     target_delta, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, seed, RDP_eps]

        unique_filename = get_unique_filename(save_filename_base)
        unique_model_dir = unique_filename + "_" + "".join([f"{n[0]}_{v}_" for n, v in zip(name_string_extension, extension)])
        print(unique_model_dir)

        model_data_dir_specific = os.path.join(model_data_dir, unique_model_dir)

        if DATASET == 'MNIST':
            model_func = lambda: get_model_dense_3layer_input784_adam_crossentropy_private(
                number_of_protected_samples=number_of_protected_samples,
                noise_multiplier=noise_multiplier,
                batch_size=batch_size,
                epochs=epochs)

            ret = get_poisoned_accuracies(model_func,
                    PoisonedDataset_EMNIST_DIGITS_AllPixelSpot(
                        number_of_authors=number_of_authors,
                        poisoned_ratio=poisoned_ratio,
                        author_is_poisoning_ratio=author_is_poisoning_ratio,
                        seed=seed),
                    save_all_data=True,
                    model_dir=model_data_dir_specific)

        elif DATASET == 'CIFAR10':
            model_func = lambda: get_model_resnet1_inputcifar10_adam_crossentropy_private(
                number_of_protected_samples=number_of_protected_samples,
                noise_multiplier=noise_multiplier,
                batch_size=batch_size,
                epochs=epochs)

            ret = get_poisoned_accuracies(model_func,
                    PoisonedDataset_CIFAR10_AllPixelSpot(
                        number_of_authors=number_of_authors,
                        poisoned_ratio=poisoned_ratio,
                        author_is_poisoning_ratio=author_is_poisoning_ratio,
                        backdoor_value=1,
                        seed=seed),
                    save_all_data=True,
                    model_dir=model_data_dir_specific)

        ret = list(ret)
        ret.extend(extension)

        print("".join([f"{name}: {val}, " for name, val in zip(name_string, ret)]))

        # dump exact values
        np.save(unique_filename, np.array(ret))

        # dump into csv
        with open(save_filename_base + ".csv", 'a') as f:
            string = "".join([f"{val}," for val in ret]) + "\n"
            f.write(string)



def run_EMNIST_quick_test_all_user_same_backdoor():
    from poisoned_dataset.poisoned_dataset_pixel_based import PoisonedDataset_EMNIST_DIGITS_AllUserSameBackdoor

    NUMBER_OF_PROTECTED_SAMPLES = 1

    DATASET = 'EMNIST'

    if DATASET == 'EMNIST':
        number_of_samples = 280000 * 4 / 5 * 4 / 5  # for EMINST __ONLY__
        noise_multiplier = 0.7448
        batch_size = 200
        epochs = 20

        number_of_authors = 1000
        # poisoned_ratio = 0.5
        # author_is_poisoning_ratio = 0.05
        poisoned_ratio = 0.5

        author_is_poisoning_ratio = 1

        seed = 42

        target_delta = 1e-5

    ## general result saving setup
    save_filename_base = f"DP_{DATASET}_single"
    model_data_dir = "./model_data"

    name_string = ["acc", "poison_acc", "unpoison_acc", "ex_acc", "ex_poison_acc", "ex_real_acc", "poison_reduced_acc",
                   "ex_poison_reduced_acc"]
    name_string_extension = ["noise_multiplier", "epochs", "batch_size", "number_of_samples",
                             "number_of_protected_samples", "target_eps", "target_delta", "number_of_authors",
                             "poisoned_ratio", "author_is_poisoning_ratio", "seed", "RDP_eps"]
    name_string.extend(name_string_extension)

    with open(save_filename_base + ".csv", 'a') as f:
        string = "".join([f"{val}," for val in name_string]) + "\n"
        f.write(string)

    if DATASET == 'EMNIST':
        model_func = lambda: get_model_dense_3layer_input784_adam_crossentropy_private(
            number_of_protected_samples=NUMBER_OF_PROTECTED_SAMPLES,
            noise_multiplier=noise_multiplier,
            batch_size=batch_size,
            epochs=epochs)

    for author_is_poisoning_ratio in [0.015, 0.02, 0.025, 0.03, 0.04]:  # [0.01, 0.05, 0.1, 0.2]:
        steps = epochs * number_of_samples // batch_size
        RDP_eps = compute_eps_tensorflow_original_code(steps=steps, N=number_of_samples, noise_multiplier=noise_multiplier, batch_size=batch_size, target_delta=target_delta)

        target_eps = RDP_eps
        print("Getting EPS: ", RDP_eps)

        extension = [noise_multiplier, epochs, batch_size, number_of_samples, target_eps,
                     target_delta, number_of_authors, poisoned_ratio, author_is_poisoning_ratio, seed, RDP_eps]

        unique_filename = get_unique_filename(save_filename_base)
        unique_model_dir = unique_filename + "_" + "".join([f"{n[0]}_{v}_" for n, v in zip(name_string_extension, extension)])
        print(unique_model_dir)

        model_data_dir_specific = os.path.join(model_data_dir, unique_model_dir)

        if DATASET == 'EMNIST':
            ret = get_poisoned_accuracies(
                model_func,
                PoisonedDataset_EMNIST_DIGITS_AllUserSameBackdoor(
                    number_of_authors=number_of_authors,
                    poisoned_ratio=poisoned_ratio,
                    author_is_poisoning_ratio=author_is_poisoning_ratio,
                    seed=seed),
                save_all_data=False,
                model_dir=model_data_dir_specific)

        ret = list(ret)
        ret.extend(extension)

        print("".join([f"{name}: {val}, " for name, val in zip(name_string, ret)]))

        # dump into csv
        with open(save_filename_base + ".csv", 'a') as f:
            string = "".join([f"{val}," for val in ret]) + "\n"
            f.write(string)

if __name__ == "__main__":
    run_EMNIST_quick_test_all_user_same_backdoor()
