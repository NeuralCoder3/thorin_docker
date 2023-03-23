# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from scipy import special as scipy_special
import torch


@torch.compile
def logsumexp(x):
    mx = torch.max(x)
    emx = torch.exp(x - mx)
    # return torch.log(sum(emx)) + mx
    return torch.log(torch.sum(emx)) + mx


@torch.compile
def logsumexpvec(x):
    '''The same as "logsumexp" but calculates result for each row separately.'''

    mx = torch.max(x, 1).values
    lset = torch.logsumexp(torch.t(x) - mx, 0)
    return torch.t(lset + mx)


# @torch.compile
# def log_gamma_distrib(a, p):
#     return scipy_special.multigammaln(a, p)


@torch.compile
def multigammaln(a, d: int):
    # Python builtin <built-in function array> is currently not supported in Torchscript:
    # https://github.com/pytorch/pytorch/issues/32268

    # a = np.asarray(a)
    # if not np.isscalar(d) or (np.floor(d) != d):
    #     raise ValueError("d should be a positive integer (dimension)")
    # if np.any(a <= 0.5 * (d - 1)):
    #     raise ValueError("condition a (%f) > 0.5 * (d-1) (%f) not met"
    #                      % (a, 0.5 * (d-1)))

    # res = (d * (d-1) * 0.25) * np.log(np.pi)
    # res += np.sum(loggam([(a - (j - 1.)/2) for j in range(1, d+1)]), axis=0)

    # Need to check relative performance

    res = (d * (d - 1) * 0.25) * math.log(math.pi)
    res += torch.sum(
        torch.tensor(
            [math.lgamma(float(a) - ((j - 1.0) / 2)) for j in range(1, d + 1)]
        ),
        dim=0,
    )
    return res


@torch.compile
def log_gamma_distrib(a: torch.Tensor, p: int):
    # return scipy_special.multigammaln(a, p)
    return multigammaln(a, p)


@torch.compile
def sqsum(x):
    return sum(x ** 2)


@torch.compile
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


# # @torch.compile
# def constructL(d, icf):
#     constructL.Lparamidx = d

#     def make_L_col(i):
#         nelems = d - i - 1
#         col = torch.cat([
#             torch.zeros(i + 1, dtype=torch.float64),
#             icf[constructL.Lparamidx:(constructL.Lparamidx + nelems)]
#         ])

#         constructL.Lparamidx += nelems
#         return col

#     columns = [make_L_col(i) for i in range(d)]
#     return torch.stack(columns, -1)


@torch.compile
def make_L_col_lifted(d: int, icf, constructL_Lparamidx: int, i: int):
    nelems = d - i - 1
    col = torch.cat(
        [
            torch.zeros(i + 1, dtype=torch.float64),
            icf[constructL_Lparamidx: (constructL_Lparamidx + nelems)],
        ]
    )

    constructL_Lparamidx += nelems
    return (constructL_Lparamidx, col)


@torch.compile
def constructL(d: int, icf):
    # constructL.Lparamidx = d
    constructL_Lparamidx = d

    # torch.jit.frontend.UnsupportedNodeError: function definitions aren't supported:

    # def make_L_col(i):
    #     nelems = d - i - 1
    #     col = torch.cat([
    #         torch.zeros(i + 1, dtype = torch.float64),
    #         icf[constructL.Lparamidx:(constructL.Lparamidx + nelems)]
    #     ])

    #     constructL.Lparamidx += nelems
    #     return col

    # columns = [make_L_col(i) for i in range(d)]

    columns = []
    for i in range(0, d):
        constructL_Lparamidx_update, col = make_L_col_lifted(
            d, icf, constructL_Lparamidx, i
        )
        columns.append(col)
        constructL_Lparamidx = constructL_Lparamidx_update

    return torch.stack(columns, -1)


@torch.compile
def Qtimesx(Qdiag, L, x):

    f = torch.einsum('ijk,mik->mij', L, x)
    return Qdiag * x + f


@torch.compile
def gmm_objective2(alphas, xcentered, icf, x, wishart_gamma, wishart_m,Ls):
    n = x.shape[0]
    d = x.shape[1]

    Qdiags = torch.exp(icf[:, :d])
    sum_qs = torch.sum(icf[:, :d], 1)
    # Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])

    # xcentered = torch.stack(tuple(x[i] - means for i in range(n)))
    # xcentered = torch.stack(tuple(means for i in range(n)))
    Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    Lxcentered = xcentered
    sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    lse = logsumexpvec(inner_term)
    # lse = inner_term
    slse = torch.sum(lse)

    # CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return slse


# LS + Qtimesx does not work
# Ls alone works
# xcentered does not work

def gmm_objective(alphas, means, icf, x, wishart_gamma, wishart_m):
    n = x.shape[0]
    d = x.shape[1]

    # return torch.sum(x)
    # return n * logsumexp(alphas)


    # Qdiags = torch.exp(icf[:, :d])
    # sum_qs = torch.sum(icf[:, :d], 1)
    # Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])

    # xcentered = torch.stack(tuple(x[i] - means for i in range(n)))
    # xcentered = torch.stack(tuple(means for i in range(n)))
    xcentered = torch.stack(list([means for i in range(n)]))
    
    
    Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])
    return gmm_objective2(alphas, xcentered, icf, x, wishart_gamma, wishart_m,Ls)
    # Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    Lxcentered = xcentered
    sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    # inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    # lse = logsumexpvec(inner_term)
    lse = inner_term
    slse = torch.sum(lse)

    # CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    return slse


    # # Qdiags = torch.exp(icf[:, :d])
    # sum_qs = torch.sum(icf[:, :d], 1)
    # # Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])

    # # xcentered = torch.stack(tuple(x[i] - means for i in range(n)))
    # # xcentered = torch.stack(tuple(means for i in range(n)))
    # xcentered = torch.stack(list([means for i in range(n)]))
    # # Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    # Lxcentered = xcentered
    # sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    # # inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    # inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    # # lse = logsumexpvec(inner_term)
    # lse = inner_term
    # slse = torch.sum(lse)

    # # CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    # return slse
    
    
    

    # Qdiags = torch.exp(icf[:, :d])
    # sum_qs = torch.sum(icf[:, :d], 1)
    # Ls = torch.stack([constructL(d, curr_icf) for curr_icf in icf])

    # xcentered = torch.stack(tuple(x[i] - means for i in range(n)))
    # Lxcentered = Qtimesx(Qdiags, Ls, xcentered)
    # sqsum_Lxcentered = torch.sum(Lxcentered ** 2, 2)
    # inner_term = alphas + sum_qs - 0.5 * sqsum_Lxcentered
    # lse = logsumexpvec(inner_term)
    # slse = torch.sum(lse)

    # CONSTANT = -n * d * 0.5 * math.log(2 * math.pi)
    # return CONSTANT + slse - n * logsumexp(alphas) \
    #     + log_wishart_prior(d, wishart_gamma, wishart_m, sum_qs, Qdiags, icf)


# Are there any applications where I should NOT use PT 2.0?
# The current release of PT 2.0 is still experimental and in the nightlies. Dynamic shapes support in torch.compile is still early, and you should not be using it yet, and wait until the Stable 2.0 release lands in March 2023.
# That said, even with static-shaped workloads, we’re still building Compiled mode and there might be bugs. Disable Compiled mode for parts of your code that are crashing, and raise an issue (if it isn’t raised already).
