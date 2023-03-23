# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import math
from scipy import special as scipy_special
import torch


def logsumexp(x):
    mx = torch.max(x)
    emx = torch.exp(x - mx)
    return torch.log(sum(emx)) + mx


def logsumexpvec(x):
    '''The same as "logsumexp" but calculates result for each row separately.'''

    mx = torch.max(x, 1).values
    lset = torch.logsumexp(torch.t(x) - mx, 0)
    return torch.t(lset + mx)


def log_gamma_distrib(a, p):
    return scipy_special.multigammaln(a, p)


def sqsum(x):
    return sum(x ** 2)


def log_wishart_prior(p, wishart_gamma, wishart_m, sum_qs, Qdiags, icf):
    n = p + wishart_m + 1
    k = icf.shape[0]

    out = torch.sum(
        0.5 * wishart_gamma * wishart_gamma *
        (torch.sum(Qdiags ** 2, dim=1) + torch.sum(icf[:, p:] ** 2, dim=1)) -
        wishart_m * sum_qs
    )

    C = n * p * (math.log(wishart_gamma / math.sqrt(2)))
    return out - k * (C - log_gamma_distrib(0.5 * n, p))


# compilation here makes it a lot slower
# @torch.compile
def constructL(d, icf):
    constructL.Lparamidx = d

    def make_L_col(i):
        nelems = d - i - 1
        col = torch.cat([
            torch.zeros(i + 1, dtype=torch.float64),
            icf[constructL.Lparamidx:(constructL.Lparamidx + nelems)]
        ])

        constructL.Lparamidx += nelems
        return col

    columns = [make_L_col(i) for i in range(d)]
    return torch.stack(columns, -1)


def Qtimesx(Qdiag, L, x):

    f = torch.einsum('ijk,mik->mij', L, x)
    return Qdiag * x + f

# @torch.compile


def gmm_objective2(alphas, xcentered, icf, x, wishart_gamma, wishart_m, Ls):
    n = x.shape[0]
    d = x.shape[1]

    Qdiags = torch.exp(icf[:, :d])
    sum_qs = torch.sum(icf[:, :d], 1)

    Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = logsumexpvec(inner_term)
    slse = torch.sum(lse)

    CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return CONSTANT + slse - n * logsumexp(alphas) \
        + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)


# https://pytorch.org/docs/master/dynamo/faq.html#why-is-my-code-crashing
# gmm_objective2 = torch.compile(gmm_objective2, mode="max-autotune")
# gmm_objective2 = torch.compile(gmm_objective2, backend="eager")
# gmm_objective2 = torch.compile(gmm_objective2, backend="aot_eager")

gmm_objective2 = torch.compile(gmm_objective2, backend="inductor")
# gmm_objective2 = torch.compile(
#     gmm_objective2, dynamic=True, backend="eager")


@torch.compile
def computeLs(d, icf):
    # Ls = []
    # for i in range(icf.shape[0]):
    #     Ls.append(constructL(d, icf[i]))
    # return torch.stack(Ls)
    return torch.stack([constructL(d, curr_icf) for curr_icf in icf])


@torch.compile
def compute_xcentered(means, x):
    return torch.stack(tuple(x[i] - means for i in range(x.shape[0])))


# LS + Qtimesx does not work
# Ls alone works
# xcentered does not work

def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n = x.shape[0]
    d = x.shape[1]

    xcentered = torch.stack(tuple(x[i] - means for i in range(n)))
    # xcentered = compute_xcentered(means, x)

    Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])
    # Ls = computeLs(d, icf)
    return gmm_objective2(alphas, xcentered, icf, x, wishart_gamma, wishart_m, Ls)


# Are there any applications where I should NOT use PT 2.0?
# The current release of PT 2.0 is still experimental and in the nightlies. Dynamic shapes support in torch.compile is still early, and you should not be using it yet, and wait until the Stable 2.0 release lands in March 2023.
# That said, even with static-shaped workloads, we’re still building Compiled mode and there might be bugs. Disable Compiled mode for parts of your code that are crashing, and raise an issue (if it isn’t raised already).
