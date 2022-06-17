"""
written by David Sommer (david.sommer at inf.ethz.ch) and Liwei Song (liweis at princeton.edu) in 2020, 2021

This file generates some of the theory plots shown in the paper.

This file is part of the code repository to reproduce the results in the publication

"Athena: Probabilistic Verification of Machine Unlearning",
by David Sommer (ETH Zurich), Liwei Song (Princeton University), Sameer Wagh (Princeton University), and Prateek Mittal,
published in "Proceedings on privacy enhancing technologies 2022.3" (2022).
"""

import os
import sys
import numpy as np
from scipy.stats import binom

from matplotlib import pyplot as plt
from plot_utils import latexify

from paper_plots_evaluation import PLOT_DIRECTORY

OVERWRITE_PLOTS = True
SHOW_PLOTS = False

# make the plots "latexy"
latexify()

# general overview plot
if True:
    plot_name = os.path.join(PLOT_DIRECTORY, "theory_general.pdf")

    q = 0.15
    p = 0.8

    n = 5
    t = int(0.5*n) / n
    print(t)

    x = np.arange(0, n+1)
    x_axis = x/n

    distrH0 = binom.pmf(x, n=n, p=q)
    distrH1 = binom.pmf(x, n=n, p=p)

    # figure
    fig = plt.figure(figsize=(3.5,1.7))
    ax = plt.gca()

    plt.plot(x_axis, distrH0, 'r-', label='_nolegend_', linewidth=0.5, alpha=0.7)
    plt.plot(x_axis, distrH1, 'b-', label='_nolegend_', linewidth=0.5, alpha=0.7)

    plt.plot(x_axis, distrH0, 'r.', label=r'$H_0$')
    plt.plot(x_axis, distrH1, 'b.', label=r'$H_1$')

    ax.fill_between(x_axis, 0, distrH1, where=x_axis <= t, facecolor='blue', alpha=0.4) # , interpolate=True)
    ax.fill_between(x_axis, 0, distrH0, where=x_axis >= t, facecolor='red', alpha=0.4) # , interpolate=True)

    plt.annotate(r'$\beta$', xy=(t-0.05, 0.01), xytext=(t-0.2, 0.05), arrowprops={'arrowstyle': '-'}, xycoords='data', fontsize=15)
    plt.annotate(r'$\alpha$', xy=(t+0.05*1.5, 0.01*2), xytext=(t+0.2, 0.05*2), arrowprops={'arrowstyle': '-'}, xycoords='data', fontsize=15)


    plt.axvline(t, 0.05, 0.325, color='blue', alpha=0.9)
    # to the left
    # t_height = 0.07
    # beta_slope = (0.05 - 0.01) / (0.2 - 0.05)*1.1
    # plt.annotate(r'$t$', xy=(t+0.005, t_height + 0.02), xytext=(t-0.12, t_height + 0.02 + 0.12*beta_slope), arrowprops={'arrowstyle': '-'}, xycoords='data', fontsize=15)

    t_height = 0.07
    beta_slope = (0.05 - 0.01) / (0.2 - 0.05)*1.1
    plt.annotate(r'$t$', xy=(t+0.008, t_height + 0.02), xytext=(t-0.13, t_height + 0.02 + 0.13*beta_slope), arrowprops={'arrowstyle': '-'}, xycoords='data', fontsize=15)

    ax.set_xlabel(r'measured backdoor success rate $\hat{r}$')
    ax.set_ylabel(r"occurrence probability", fontsize=7)

    plt.legend(loc='center left')

    plt.tight_layout()

    if OVERWRITE_PLOTS or not os.path.exists(plot_name):
        plt.savefig(plot_name, bbox_inches="tight", pad_inches = 0)

    if SHOW_PLOTS:
        plt.show()


# illustrate that distribution gets smaller with higher n

# illustrate that distribution gets smaller with higher n, normalizing t omax_value of pmf
if True:
    plot_name = os.path.join(PLOT_DIRECTORY, "theory_compare.pdf")
    q = 0.1
    p = 0.8

    n = [5, 10, 20]

    fig = plt.figure(figsize=(3.5,1.7))
    ax = plt.gca()

    options = [ {'linestyle': ':', 'alpha': 0.3},
                {'linestyle': '--', 'alpha': 0.5},
                {'linestyle': '-' , 'alpha': 1.0},
              ]
    line2Ds = []
    for i, n_i in enumerate(n):
        x = np.arange(0, n_i+1)
        pmf_q = binom.pmf(x, n=n_i, p=q)
        pmf_p = binom.pmf(x, n=n_i, p=p)
        norm = max(np.max(pmf_q), np.max(pmf_p))

        y_q = pmf_q/norm
        y_p = pmf_p/norm

        print(x, y_q)
        l2D1 = plt.plot(x/n_i, y_q, color='r', marker='.', markersize=2, label=rf"$n = {n_i}$", lw=0.5, **options[i])
        l2D2 = plt.plot(x/n_i, y_p, color='b', marker='.', markersize=2, label="_nolegend_", lw=0.5, **options[i])

        line2Ds.append((*l2D1,*l2D2))
        # plt.plot(x/n_i, y_q, color='r', marker='.', markersize=2, label="_nolegend_", lw=0.5, **options[i])
        # plt.plot(x/n_i, y_p, color='b', marker='.', markersize=2, label="_nolegend_", lw=0.5, **options[i])

    ax.set_xlabel(r'measured backdoor success rate $\hat{r}$')
    ax.set_ylabel(r"\hspace{-1.85em}\vspace{-0.5em}relative\\occurrence probability", fontsize=7)
    ax.set_yticks([])

    from matplotlib.legend_handler import HandlerBase
    class AnyObjectHandler(HandlerBase):
        def create_artists(self, legend, orig_handle,
                           x0, y0, width, height, fontsize, trans):
            a,b = orig_handle
            l1 = plt.Line2D([x0,y0+width], [0.8*height,0.8*height], linestyle=a.get_linestyle(), alpha=a.get_alpha(), color=a.get_color(), linewidth=a.get_linewidth())
            l3 = plt.Line2D([(x0+y0+width)/2], [0.8*height], markersize=a.get_markersize(), linestyle=a.get_linestyle(), alpha=a.get_alpha(), color=a.get_color(), marker=a.get_marker(), linewidth=a.get_linewidth())

            l2 = plt.Line2D([x0,y0+width], [0.2*height,0.2*height], linestyle=b.get_linestyle(), alpha=b.get_alpha(), color=b.get_color(), linewidth=b.get_linewidth())
            l4 = plt.Line2D([(x0+y0+width)/2], [0.2*height], markersize=b.get_markersize(), linestyle=b.get_linestyle(), alpha=b.get_alpha(), color=b.get_color(), marker=b.get_marker(), linewidth=b.get_linewidth())
            return [l1, l2, l3, l4]

    plt.legend(line2Ds, [f"$n={i}$" for i in n],
           handler_map={tuple: AnyObjectHandler()})

    # plt.legend()
    plt.tight_layout()

    if OVERWRITE_PLOTS or not os.path.exists(plot_name):
        plt.savefig(plot_name, bbox_inches="tight", pad_inches = 0)

    if SHOW_PLOTS:
        plt.show()


def beta(alpha, q, p, n):
    """
    alpha: maximal mass in right tail of H_0
    q: prob of H_0 (deleted)
    p: prob of H_1 (not deleted)
    n: number of measurements
    """
    x = np.arange(n+1)

    pdf_H0 = binom.pmf(x, n=n, p=q)
    cdf_H0 = np.cumsum(pdf_H0)

    idx = np.argmax(cdf_H0 + alpha > 1)

    return np.sum( binom.pmf( np.arange(idx) , n=n, p=p) )

if True:
    print(beta(0.0, 0.1, 0.6, 10))


# compute leftover mass in tail
if True:
    p = 0.2
    cutoff_vals = np.linspace(0,1,11)
    res = []
    da_ns = np.logspace(0, 3.0, base=10, num=20, dtype=np.int32)
    for n in da_ns:
        r = []
        for c in cutoff_vals:
            x = np.arange(int(np.ceil(n*c)), n+1)
            pmf = binom.pmf(x, n=n, p=p)
            tail_mass = np.sum(pmf)
            r.append(tail_mass)
        res.append(r)

    res = np.array(res)
    for i, c in enumerate(cutoff_vals):
        plt.semilogy(da_ns, res[:,i], label=f"{c:0.2f}")

    plt.legend()
    plt.show()

