"""
written by David Sommer (david.sommer at inf.ethz.ch) in 2020, 2021.

This file generates some of the empirical plots shown in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import numpy as np
import argparse
from scipy.special import comb
from scipy.stats import binom
import itertools


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', nargs='?', default='common_fn_P_sampling', choices=['common_fn_P_sampling', 'common_fp_Q_sampling', 'major_sampling', 'major_exact', 'major_exact_q_sampling'],
                        help="what method to use")

    parser.add_argument("--data_dir", type=str, default="final_data", help="the data directory")

    parser.add_argument("--number_of_queries", type=int, default=30, help="number of theoretical test queries to the model")
    # parser.add_argument("--number_of_collab_users", type=int, nargs='+', default=[1, 3, 5, 7, 13], help="number of collaborating users. is LIST")
    parser.add_argument("--number_of_collab_users", type=int, nargs='+', default=[3], help="number of collaborating users. is LIST")

    parser.add_argument("--sim_2_rep_number", type=int, default=3, help="number of p repeats when q sampling")

    parser.add_argument("--sampler_rep_number", type=int, default=int(1e7), help="number sampling repetitions")

    parser.add_argument("--alpha", type=float, default=1e-3, help="alpha")
    parser.add_argument("--beta", type=float, default=1e-3, help="beta")

    parser.add_argument("--verbose", action="store_true", default=False)

    args = parser.parse_args()

    return args


def load_data_iter(data_dir):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

    for file in files:
        arr = np.load(file)

        data = dict(arr)
        ret_key = os.path.basename(file)

        yield ret_key, data


def reject_null_hypothesis(alpha, beta, q, p, n):

    """
    alpha: maximal mass in right tail of H_0
    q: prob of H_0 (deleted)
    p: prob of H_1 (not deleted)
    n: number of measurements
    """
    if p == 1:
        p -= 1e-6
    x = np.arange( n + 1)

    # pdf_H0 = binom.pmf(x, n=n, p=q)
    # cdf_H0 = np.cumsum(pdf_H0)
    cdf_H0 = binom.cdf(x, n=n, p=q)

    idx = np.argmax(cdf_H0 + alpha >= 1)
    # idx = np.argmax(cdf_H0 + alpha > 1)-1

    typeIIerror = np.sum( binom.pmf( np.arange(idx + 1), n=n, p=p) )

    return typeIIerror < beta


def generate_statistics_exact(args):
    """ GOES OVER ALL POSSIBLE PERMUTATIONS """
    for key, data in load_data_iter(args.data_dir):
        print(key)

        p_data = data['in_attack_acc_list']
        q_data = data['ex_attack_acc_list']

        if args.mehod == 'major_exact':
            # This one is too inefficient. To many combinations
            p_q_iter = itertools.product(p_data, q_data)
        elif args.mehod == 'major_exact_q_sampling':
            p_q_iter = zip(np.repeat(p_data, args.sim_2_rep_number), np.random.choice(q_data, len(p_data) * args.sim_2_rep_number, replace=True))
        else:
            raise ValueError(args.mehod)

        # generate the array of hypthesis tests
        reject_hypot_arr = []
        for p, q in p_q_iter:
            if args.verbose:
                print(p, q)
            rej = reject_null_hypothesis(args.alpha, args.beta, q, p, args.number_of_queries)
            reject_hypot_arr.append(rej)

        reject_hypot_arr = np.array(reject_hypot_arr)  # easier indexing later

        if args.verbose:
            print(reject_hypot_arr)
        print(f"{key} n={args.number_of_queries} len(p) = {len(p_data)}, len(q) = {len(q_data)}, len(reject_hypot) = {len(reject_hypot_arr)} {args.method}")

        # for all the numer of user combinations we want a result
        for c in args.number_of_collab_users:
            total_combs = comb(len(reject_hypot_arr), c)
            if args.verbose:
                print(f"Number of combinations: {total_combs}")

            debug_cnt = 0
            count = 0
            if c == 1:
                count = sum(reject_hypot_arr)
            else:
                for subset in itertools.combinations(np.arange(len(reject_hypot_arr)), c):
                    debug_cnt += 1
                    # print(subset)
                    votes = reject_hypot_arr[list(subset)]
                    # print(votes)
                    count += int(sum(votes) > c // 2)  # we check for majority. If c even: a tie is considered 'no rejection'

                assert debug_cnt == total_combs

            print(f"  users = {c} | ratio = {(total_combs - count) / total_combs} (total combination = {total_combs})")


def generate_statistics_sampling_each_time(args):
    """ Samples user and thn does majority voting """
    for key, data in load_data_iter(args.data_dir):
        rep = args.sampler_rep_number

        p_data = data['in_attack_acc_list']
        q_data = data['ex_attack_acc_list']

        print(f"{key} n={args.number_of_queries} len(p) = {len(p_data)}, len(q) = {len(q_data)}, {args.method}")

        # for all the numer of user combinations we want a result
        for c in args.number_of_collab_users:

            ret_count = 0
            for _ in range(rep):
                p_user = np.random.choice(p_data, c, replace=False)
                q_user = np.random.choice(q_data, c, replace=False)

                cnt = 0
                for p, q in zip(p_user, q_user):
                    cnt += int(reject_null_hypothesis(args.alpha, args.beta, q, p, args.number_of_queries))

                ret_count += int(cnt > c // 2)

            print(f"  users = {c} | ratio = {(rep - ret_count) / rep} (total combination = {rep})")


def generate_statistics_sampling_false_negative(args):
    """
    samples user and then let them share their p and q and computes single hypothesis test

    computes FALSE NEGATIVES (server did not delete data but we falsely think it did)
    """
    for key, data in load_data_iter(args.data_dir):
        rep = args.sampler_rep_number

        p_data = data['in_attack_acc_list']
        q_data = data['ex_attack_acc_list']

        print(f"{key} n={args.number_of_queries} len(p) = {len(p_data)}, len(q) = {len(q_data)}, {args.method}")

        # for all the numer of user combinations we want a result
        for c in args.number_of_collab_users:

            ret_count = 0
            for _ in range(rep):
                p = np.mean(np.random.choice(p_data, c, replace=False))
                q = np.mean(np.random.choice(q_data, c, replace=False))

                ret_count += reject_null_hypothesis(args.alpha, args.beta, q, p, c * args.number_of_queries)

            print(f"  users = {c} | ratio = {(rep - ret_count) / rep} (count / rep = {ret_count} / {rep})")


def generate_statistics_sampling_false_positive(args):
    """
    samples user and then let them share their p and q and computes single hypothesis test

    computes FALSE POSITIVES (server did delete data but we falsely think it did not)
    """
    for key, data in load_data_iter(args.data_dir):
        rep = args.sampler_rep_number

        p_data = data['in_attack_acc_list']
        q_data = data['ex_attack_acc_list']

        print(f"{key} n={args.number_of_queries} len(p) = {len(p_data)}, len(q) = {len(q_data)}, {args.method}")

        # for all the numer of user combinations we want a result
        for c in args.number_of_collab_users:

            ret_count = 0
            for _ in range(rep):
                m = np.mean(np.random.choice(q_data, c, replace=False))
                q = np.mean(np.random.choice(q_data, c, replace=False))

                ret_count += reject_null_hypothesis(args.alpha, args.beta, q, m, c * args.number_of_queries)

            print(f"  users = {c} | ratio = {(ret_count) / rep} (count / rep = {ret_count} / {rep})")


if __name__ == "__main__":
    args = parse_arguments()

    # generate_statistics_exact(args)
    if args.method == 'common_fn_P_sampling':
        generate_statistics_sampling_false_negative(args)
    elif args.method == 'common_fp_Q_sampling':
        generate_statistics_sampling_false_positive(args)
    elif args.method == 'major_sampling':
        generate_statistics_sampling_each_time(args)
    elif args.method == 'major_exact' or args.method == 'major_exact_q_sampling':
        generate_statistics_exact(args)
    else:
        raise NotImplementedError(args.method)
