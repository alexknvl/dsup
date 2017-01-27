#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
if sys.version_info < (3, 0):
    import __builtin__
    range = __builtin__.xrange

import numpy as np
import numpy.linalg as la
import scipy.optimize
import scipy.sparse
import scipy.special
import random

from collections import namedtuple

import lrxr.distribution as distr


def split(array, *sizes):
    sizes = np.asarray(sizes, dtype=np.int32)
    assert array.shape[0] == np.sum(sizes)
    result = np.split(array, np.cumsum(sizes))
    assert len(result) == len(sizes) + 1
    return tuple(result[:-1])


def subsums(array, *sizes):
    sizes = np.asarray(sizes, dtype=np.int32)
    assert array.shape[0] == np.sum(sizes)
    return np.array([np.sum(x, axis=0)
                     for x in np.split(array, np.cumsum(sizes))[:-1]])


def submask(selected, sizes):
    sizes = np.asarray(sizes, dtype=np.int32)
    selected_mask = np.zeros(len(sizes), dtype=bool)
    selected_mask[selected] = True
    return np.repeat(selected_mask, sizes)


def kl_term(w, u, t, p_ex):
    assert p_ex != 0
    assert p_ex != 1
    assert u.shape[0] > 1

    y = lr_probability(w, u, t)
    p_em = np.sum(y) / y.shape[0]

    if p_em == 0:
        p_em = 1 / y.shape[0]
    if p_em == 1:
        p_em = 1 - 1 / y.shape[0]

    return p_ex * np.log(p_ex / p_em) + \
        (1 - p_ex) * np.log((1 - p_ex) / (1 - p_em))


def kl_grad(w, u, t, p_ex):
    assert p_ex != 0
    assert p_ex != 1
    assert u.shape[0] > 1

    y = lr_probability(w, u, t)
    q_wStar1 = np.sum(y)
    q_wStar0 = np.sum(1 - y)

    if q_wStar1 == 0:
        q_wStar1 = 1
        q_wStar0 -= 1
    if q_wStar0 == 0:
        q_wStar0 = 1
        q_wStar1 -= 1

    kl_grad = (((1 - p_ex) / q_wStar0) * y * (1 - y) -
               (p_ex / q_wStar1) * y * (1 - y))

    return u.T.dot(kl_grad) / t


def lr_log_probability(w, x, y=1, t=1):
    return -np.logaddexp(0, -y * x.dot(w) / t)


def lr_probability(w, x, y=1, t=1):
    return np.exp(lr_log_probability(w, x, y, t))


def lr_obj(w, x, y, l2, t=1):
    return np.sum(-distr.lr_log(w, x, y)) - distr.l2_log(w, l2)


def lr_grad(w, x, y, l2, t=1):
    return -distr.lr_log_dw(w, x, y) - distr.l2_log_dw(w, l2)


def lr_train(w, x, y, l2, tol):
    (w, nll, status) = scipy.optimize.fmin_l_bfgs_b(
        lr_obj, fprime=lr_grad,
        x0=w, args=(x, y, l2),
        pgtol=tol)
    return (w, nll, status)


def lrxr_obj(w, x, y, u, l2, t, p_ex, xr):
    #sys.stderr.write("lrxr_obj [%s, %s, %s]" % (x.shape, y.shape, w.shape))
    return \
        lr_obj(w, x, y, l2, t) + \
        (xr * kl_term(w, u, t, p_ex) if xr != 0 else 0) + \
        (np.sum(-lr_log_probability(w, u, -1, t)) if xr == 0 else 0)


def lrxr_grad(w, x, y, u, l2, t, p_ex, xr):
    return \
        lr_grad(w, x, y, l2, t) + \
        (xr * kl_grad(w, u, t, p_ex) if xr != 0 else 0) + \
        (u.T.dot(lr_probability(w, u, -1, t)) / t if xr == 0 else 0)


def lrxr_train(w, x, y, u, l2, t, p_ex, xr, tol):
    (w, nll, status) = scipy.optimize.fmin_l_bfgs_b(
        lrxr_obj, fprime=lrxr_grad,
        x0=w, args=(x, y, u, l2, t, p_ex, xr),
        pgtol=tol)
    return (w, nll, status)


def lrxrta5_obj(wf, params, acnt, x, dt, u):
    (tau0, mu0, s20, alpha, beta, l2, t, p_ex, xr) = params
    A = acnt.shape[0]
    F = l2.shape[0]
    mu, s2, w = split(wf, A, A, F)

    mur = np.repeat(mu, acnt)
    s2r = np.repeat(s2, acnt)

    n0 = -np.log(tau0)
    n1 = distr.normal_logpdf(dt, mur, np.abs(s2r))
    pf = lr_log_probability(w, x)
    nf = lr_log_probability(w, -x)
    z = np.logaddexp(n0 + nf, n1 + pf)

    return w.T.dot(l2 * w) + \
        np.sum(-distr.normal_logpdf(mu, mu0, np.abs(s20))) + \
        np.sum(-z) + \
        (xr * kl_term(w, u, t, p_ex) if xr != 0 else 0) + \
        (np.sum(-lr_log_probability(w, u, -1, t)) if xr == 0 else 0) + \
        np.sum(-distr.invgamma_logpdf(np.abs(s2), alpha, beta))


def lrxrta5_grad(wf, params, acnt, x, dt, u):
    (tau0, mu0, s20, alpha, beta, l2, t, p_ex, xr) = params
    A = acnt.shape[0]
    F = l2.shape[0]
    mu, s2, w = split(wf, A, A, F)
    #s2 = np.squeeze(s2)

    mur = np.repeat(mu, acnt)
    s2r = np.repeat(s2, acnt)

    n0 = -np.log(tau0)
    n1 = distr.normal_logpdf(dt, mur, np.abs(s2r))
    pf = lr_log_probability(w, x)
    nf = lr_log_probability(w, -x)
    f = np.exp(pf)
    z = np.logaddexp(n0 + nf, n1 + pf)
    g0 = np.exp(n0 + nf - z)
    g1 = np.exp(n1 + pf - z)

    dmu = subsums(g1 * ((mur - dt) / np.abs(s2r)), acnt) + (mu - mu0) / s20
    ds2 = subsums(g1 * np.sign(s2r) * (np.abs(s2r) - (dt - mur)**2) / (2 * s2r**2),
                  acnt) - np.sign(s2) * distr.invgamma_logpdf_dx(np.abs(s2), alpha, beta)
    dw = \
        (1 / t) * x.T.dot((f - 1) * g1) + \
        (1 / t) * x.T.dot((f - 0) * g0) + \
        2 * l2 * w + \
        (xr * kl_grad(w, u, t, p_ex) if xr != 0 else 0) + \
        (u.T.dot(lr_probability(w, u, t)) / t if xr == 0 else 0)

    return np.concatenate((dmu, ds2, dw))


def lrxrta5_train(wf, params, acnt, x, dt, u, tol):
    start_nll = lrxrta5_obj(wf, params, acnt, x, dt, u)

    (w, nll, status) = scipy.optimize.fmin_l_bfgs_b(
        lrxrta5_obj, fprime=lrxrta5_grad,
        x0=wf, args=(params, acnt, x, dt, u),
        pgtol=tol)

    print "NLL (start): %s" % start_nll
    print "NLL (done): %s" % lrxrta5_obj(w, params, acnt, x, dt, u)

    return (w, nll, status)


def lrxrta5_train_sgd(wf, params, acnt, x, dt, u,
                      tol, iterations,
                      labeled_batch_size,
                      unlabeled_batch_size,
                      mu_alpha,
                      s2_alpha,
                      w_alpha):
    start_nll = lrxrta5_obj(wf, params, acnt, x, dt, u)

    A = acnt.shape[0]
    F = x.shape[1]
    w = wf.copy()

    for i in range(iterations):
        mu, s2, lrw = split(w, A, A, F)

        ep_mask = random.sample(
            np.arange(len(acnt)), labeled_batch_size)
        labeled_mask = submask(ep_mask, acnt)
        unlabeled_mask = random.sample(
            np.arange(u.shape[0]), unlabeled_batch_size)

        batch_acnt = acnt[ep_mask]
        batch_x = x[labeled_mask, :]
        batch_dt = dt[labeled_mask]
        batch_u = u[unlabeled_mask]

        batch_mu = mu[ep_mask]
        batch_s2 = s2[ep_mask]

        g = lrxrta5_grad(
            np.concatenate((batch_mu, batch_s2, lrw)),
            params,
            batch_acnt,
            batch_x,
            batch_dt,
            batch_u)
        g_mu, g_s2, g_w = split(
            g, labeled_batch_size, labeled_batch_size, F)

        batch_mu += -mu_alpha * g_mu
        batch_s2 += -s2_alpha * g_s2
        lrw += -w_alpha * g_w

        mu[ep_mask] = batch_mu
        s2[ep_mask] = batch_s2
        w = np.concatenate((mu, s2, lrw))

        print "NLL (%d): %s" % (i, lrxrta5_obj(w, params, acnt, x, dt, u))
    print "NLL (start): %s" % start_nll
    print "NLL (done): %s" % lrxrta5_obj(w, params, acnt, x, dt, u)
    nll = lrxrta5_obj(w, params, acnt, x, dt, u)
    return (w, nll, {'grad': g})


def lrxrta5_train_adadelta(wf, params, acnt, x, dt, u,
                           tol, iterations,
                           labeled_batch_size,
                           unlabeled_batch_size,
                           decay_rate,
                           epsilon):
    start_nll = lrxrta5_obj(wf, params, acnt, x, dt, u)

    A = acnt.shape[0]
    F = x.shape[1]
    w = wf.copy()

    min_nll = start_nll
    min_w = w
    min_it = 0

    eg2 = np.zeros(len(w))
    edw2 = np.zeros(len(w))

    for i in range(iterations):
        g = lrxrta5_grad(w, params, acnt, x, dt, u)
        eg2 = decay_rate * eg2 + (1 - decay_rate) * g**2
        dw = -np.sqrt(edw2 + epsilon) / np.sqrt(eg2 + epsilon) * g
        edw2 = decay_rate * edw2 + (1 - decay_rate) * dw**2
        w = w + dw

        nll = lrxrta5_obj(w, params, acnt, x, dt, u)
        if min_nll > nll:
            min_nll = nll
            min_w = w.copy()
            min_it = i

        print "NLL (%d): %s" % (i, nll)
    print "NLL (start): %s" % start_nll
    print "NLL (done): %s" % nll
    status = {
        'grad_norm': la.norm(g),
        'start_nll': start_nll,
        'done_nll': nll,
        'min_it': min_it,
        'min_nll': min_nll}
    return (min_w, min_nll, status)


def train_adadelta(obj, grad,
                   w0, wmin, wmax,
                   parameters, x, dt, u,
                   iterations, decay_rate, epsilon):
    wmin = wmin if wmin is not None else (np.ones_like(w0) * (-np.inf))
    wmax = wmax if wmax is not None else (np.ones_like(w0) * np.inf)

    start_nll = np.squeeze(obj(w0, parameters, x, dt, u))

    w = w0.copy()

    min_nll = start_nll
    min_w = w
    min_it = 0

    eg2 = np.zeros(len(w))
    edw2 = np.zeros(len(w))

    iteration = 0

    for (count, scaler) in iterations:
        w = min_w

        for i in range(count):
            g = grad(w, parameters, x, dt, u)
            eg2 = decay_rate * eg2 + (1 - decay_rate) * g**2
            dw = -np.sqrt(edw2 + epsilon) / np.sqrt(eg2 + epsilon) * g
            dw *= scaler
            edw2 = decay_rate * edw2 + (1 - decay_rate) * dw**2

            w = np.where(
                w + dw < wmin,
                2 * wmin - (w + dw),
                np.where(
                    w + dw > wmax,
                    2 * wmax - (w + dw),
                    w + dw))

            nll = np.squeeze(obj(w, parameters, x, dt, u))
            if min_nll > nll:
                min_nll = nll
                min_w = w.copy()
                min_it = iteration

            print "NLL (%d): %s" % (iteration, nll)
            iteration += 1

    print "NLL (start): %s" % start_nll
    print "NLL (done): %s" % nll
    status = {
        'grad_norm': la.norm(g),
        'start_nll': start_nll,
        'done_nll': nll,
        'min_it': min_it,
        'min_nll': min_nll}
    return (min_w, min_nll, status)


XR = namedtuple('XR', 'probability coefficient')


# --- Model #1:
# P(w | l2) = N(w | 0, 1/l2)
# P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
# P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
# P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
#                          I[y=0] U(t[i], 1/tau)
# P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
#
# Weights are Mu, Sigma**2, mu, s2, w.
# Other parameters are fixed.
# s2 is shared between all entity pairs!
# Sigma**2 and s2 have non-informative priors.
Model1Parameters = namedtuple(
    'Model1Parameters', 'tau Mu l2 t xr sample_counts')


def model1_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (Mu, S2, mu, s2, w) = \
        split(weights, 1, 1, pair_count, 1, feature_count)
    Mu, S2, s2 = np.squeeze(Mu), np.squeeze(S2), np.squeeze(s2)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
    result += -distr.normal_logpdf(Mu, parameters.Mu.mu, parameters.Mu.s2)

    # P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
    result += np.sum(-distr.normal_logpdf(mu, Mu, S2))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu = np.repeat(mu, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model1_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (Mu, S2, mu, s2, w) = \
        split(weights, 1, 1, pair_count, 1, feature_count)

    dMu = np.zeros_like(Mu)
    dS2 = np.zeros_like(S2)
    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
    dMu += -distr.normal_logpdf_dx(Mu, parameters.Mu.mu, parameters.Mu.s2)

    # P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
    dmu += -distr.normal_logpdf_dx(mu, Mu, S2)
    dMu += np.sum(-distr.normal_logpdf_dmu(mu, Mu, S2))
    dS2 += np.sum(-distr.normal_logpdf_ds2(mu, Mu, S2))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
          (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, s2),
                   parameters.sample_counts)
    ds2 += np.sum(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, s2))

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dMu, dS2, dmu, ds2, dw))


# --- Model #2:
# P(w | l2) = N(w | 0, 1/l2)
# P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
# P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
# P(t[i] | mu[i], s2[i], y) = I[y=1] N(t[i] | mu[i], s2[i]) +
#                             I[y=0] U(t[i], 1/tau)
# P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
#
# Weights are Mu, Sigma**2, mu, s2, w.
# Other parameters are fixed.
# Sigma**2 and s2[i] have non-informative priors.
Model2Parameters = namedtuple(
    'Model2Parameters', 'tau Mu l2 t xr sample_counts')


def model2_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (Mu, S2, mu, s2, w) = \
        split(weights, 1, 1,
              pair_count, pair_count, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
    result += -distr.normal_logpdf(Mu, parameters.Mu.mu, parameters.Mu.s2)

    # P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
    result += np.sum(-distr.normal_logpdf(mu, Mu, S2))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2[i]) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu = np.repeat(mu, parameters.sample_counts)
    repeated_s2 = np.repeat(s2, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model2_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (Mu, S2, mu, s2, w) = \
        split(weights, 1, 1,
              pair_count, pair_count, feature_count)

    dMu = np.zeros_like(Mu)
    dS2 = np.zeros_like(S2)
    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)
    repeated_s2 = np.repeat(s2, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
    dMu += -distr.normal_logpdf_dx(Mu, parameters.Mu.mu, parameters.Mu.s2)

    # P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
    dmu += -distr.normal_logpdf_dx(mu, Mu, S2)
    dMu += np.sum(-distr.normal_logpdf_dmu(mu, Mu, S2))
    dS2 += np.sum(-distr.normal_logpdf_ds2(mu, Mu, S2))

    # P(t[i] | mu[i], s2[i], y) = I[y=1] N(t[i] | mu[i], s2[i]) +
    #                             I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
          (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, repeated_s2),
                   parameters.sample_counts)
    ds2 += subsums(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, repeated_s2),
                   parameters.sample_counts)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dMu, dS2, dmu, ds2, dw))


# --- Model #3:
# P(w | l2) = N(w | 0, 1/l2)
# P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
# P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
#                          I[y=0] U(t[i], 1/tau)
# P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
#
# Weights are mu, s2, w.
# Other parameters are fixed.
# s2 is shared among entity pairs!
# s2 has a non-informative prior.
Model3Parameters = namedtuple(
    'Model3Parameters', 'tau mu l2 t xr sample_counts')


def model3_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (mu, s2, w) = split(weights, pair_count, 1, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
    result += np.sum(-distr.normal_logpdf(
        mu, parameters.mu.mu, parameters.mu.s2))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu = np.repeat(mu, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model3_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (mu, s2, w) = split(weights, pair_count, 1, feature_count)

    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
    dmu += -distr.normal_logpdf_dx(
        mu, parameters.mu.mu, parameters.mu.s2)

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
          (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, s2),
                   parameters.sample_counts)
    ds2 += np.sum(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, s2))

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dmu, ds2, dw))


# --- Model #4:
# P(w | l2) = N(w | 0, 1/l2)
# P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
# P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
#                          I[y=0] U(t[i], 1/tau)
# P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
#
# Weights are mu, w.
# Other parameters are fixed.
Model4Parameters = namedtuple(
    'Model4Parameters', 'tau mu s2 l2 t xr sample_counts')


def model4_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (mu, w) = split(weights, pair_count, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
    result += np.sum(-distr.normal_logpdf(
        mu, parameters.mu.mu, parameters.mu.s2))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu = np.repeat(mu, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, parameters.s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model4_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (mu, w) = split(weights, pair_count, feature_count)

    dmu = np.zeros_like(mu)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
    dmu += -distr.normal_logpdf_dx(
        mu, parameters.mu.mu, parameters.mu.s2)

    # P(t[i] | mu[i], s2[i], y) = I[y=1] N(t[i] | mu[i], s2[i]) +
    #                             I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, parameters.s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
          (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu,
                                                 parameters.s2),
                   parameters.sample_counts)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dmu, dw))


# --- Model #5:
# P(w | l2) = N(w | 0, 1/l2)
# P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
# P(s2[i] | s2.alpha, s2.beta) = InvGamma(s2[i] | s2.alpha, s2.beta)
# P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
#                          I[y=0] U(t[i], 1/tau)
# P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
#
# Weights are mu, s2, w.
# Other parameters are fixed.
Model5Parameters = namedtuple(
    'Model5Parameters', 'tau mu s2 l2 t xr sample_counts')


def model5_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (mu, s2, w) = split(weights, pair_count,
                        pair_count, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
    result += np.sum(-distr.normal_logpdf(
        mu, parameters.mu.mu, parameters.mu.s2))

    # sum_i P(s2[i] | s2_a, s2_b)
    result += np.sum(-distr.invgamma_logpdf(
        s2, parameters.s2.alpha, parameters.s2.beta))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu = np.repeat(mu, parameters.sample_counts)
    repeated_s2 = np.repeat(s2, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model5_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (mu, s2, w) = split(weights, pair_count,
                        pair_count, feature_count)

    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)
    repeated_s2 = np.repeat(s2, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(mu[i] | mu.mu, mu.s2) = N(mu[i] | mu.mu, mu.s2)
    dmu += -distr.normal_logpdf_dx(
        mu, parameters.mu.mu, parameters.mu.s2)

    # sum_i P(s2[i] | s2_a, s2_b)
    ds2 += -distr.invgamma_logpdf_dx(
        s2, parameters.s2.alpha, parameters.s2.beta)

    # P(t[i] | mu[i], s2[i], y) = I[y=1] N(t[i] | mu[i], s2[i]) +
    #                             I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
          (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, repeated_s2),
                   parameters.sample_counts)
    ds2 += subsums(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, repeated_s2),
                   parameters.sample_counts)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dmu, ds2, dw))


# --- Model #6:
#
# P(t[i] | mu[i], s2[i], y) = I[y=1] N(t[i] | mu[i], s2[i]) +
#                             I[y=0] U(t[i], 1/tau)
# P(mu[i] | π, lp, ln) = I[x>=0] π Exp(mu[i] | lp) +
#                        I[x<0] (1 - π) Exp(-mu[i] | ln)
# P(s2[i] | s2.alpha, s2.beta) = InvGamma(s2[i] | s2.alpha, s2.beta)
# P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
# P(π | π.alpha, π.beta) = Beta(π | π.alpha, π.beta)
# P(λ_p | λ_p.alpha, λ_p.beta) = Gamma(λ_p | λ_p.alpha, λ_p.beta)
# P(λ_n | λ_n.alpha, λ_n.beta) = Gamma(λ_n | λ_n.alpha, λ_n.beta)
# P(w | l2) = N(w | 0, 1/l2)
#
# Weights are π, λ_p, λ_n, mu, s2, w.
# Other parameters are fixed.

Model6Parameters = namedtuple(
    'Model6Parameters', 'tau mu s2 l2 t xr sample_counts')


def model6_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (pi, lp, ln, mu, s2, w) = \
        split(weights, 1, 1, 1,
              pair_count, pair_count, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(pi, lp, ln | ...)
    result += -distr.beta_logpdf(
        pi, parameters.mu.pi.alpha, parameters.mu.pi.beta)
    result += -distr.gamma_logpdf(
        lp, parameters.mu.lp.alpha, parameters.mu.lp.beta)
    result += -distr.gamma_logpdf(
        ln, parameters.mu.ln.alpha, parameters.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    result += np.sum(-distr.expmixture_logpdf(
        mu, pi, lp, ln))

    # sum_i P(s2[i] | s2_a, s2_b)
    result += np.sum(-distr.invgamma_logpdf(
        s2, parameters.s2.alpha, parameters.s2.beta))

    # sum_i sum_j sum_y P(t[i, j] | y) P(y | x)
    repeated_mu = np.repeat(mu, parameters.sample_counts)
    repeated_s2 = np.repeat(s2, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model6_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (pi, lp, ln, mu, s2, w) = \
        split(weights, 1, 1, 1,
              pair_count, pair_count, feature_count)

    dpi = np.zeros_like(pi)
    dlp = np.zeros_like(lp)
    dln = np.zeros_like(ln)
    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)
    repeated_s2 = np.repeat(s2, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(pi, lp, ln | ...)
    dpi += -distr.beta_logpdf_dx(
        pi, parameters.mu.pi.alpha, parameters.mu.pi.beta)
    dlp += -distr.gamma_logpdf_dx(
        lp, parameters.mu.lp.alpha, parameters.mu.lp.beta)
    dln += -distr.gamma_logpdf_dx(
        ln, parameters.mu.ln.alpha, parameters.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    dmu += -distr.expmixture_logpdf_dx(mu, pi, lp, ln)
    dpi += np.sum(-distr.expmixture_logpdf_dpi(
        mu, pi, lp, ln))
    dlp += np.sum(-distr.expmixture_logpdf_dlp(
        mu, pi, lp, ln))
    dln += np.sum(-distr.expmixture_logpdf_dln(
        mu, pi, lp, ln))

    # sum_i P(s2[i] | s2_a, s2_b)
    ds2 += -distr.invgamma_logpdf_dx(
        s2, parameters.s2.alpha, parameters.s2.beta)

    # sum_i sum_j sum_y P(t[i, j] | y) P(y | x)
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
        (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, repeated_s2),
                   parameters.sample_counts)
    ds2 += subsums(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, repeated_s2),
                   parameters.sample_counts)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dpi, dlp, dln, dmu, ds2, dw))


# --- Model #7:
#
# P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
#                          I[y=0] U(t[i], 1/tau)
# P(mu[i] | π, lp, ln) = I[x>=0] π Exp(mu[i] | lp) +
#                        I[x<0] (1 - π) Exp(-mu[i] | ln)
# P(s2 | s2.alpha, s2.beta) = InvGamma(s2[i] | s2.alpha, s2.beta)
# P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
# P(π | π.alpha, π.beta) = Beta(π | π.alpha, π.beta)
# P(λ_p | λ_p.alpha, λ_p.beta) = Gamma(λ_p | λ_p.alpha, λ_p.beta)
# P(λ_n | λ_n.alpha, λ_n.beta) = Gamma(λ_n | λ_n.alpha, λ_n.beta)
# P(w | l2) = N(w | 0, 1/l2)
#
# Weights are π, λ_p, λ_n, mu, s2, w.
# s2 is shared.
# Other parameters are fixed.

Model7Parameters = namedtuple(
    'Model7Parameters', 'tau mu s2 l2 t xr sample_counts')


def model7_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (pi, lp, ln, mu, s2, w) = \
        split(weights, 1, 1, 1, pair_count, 1, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(pi, lp, ln | ...)
    result += -distr.beta_logpdf(
        pi, parameters.mu.pi.alpha, parameters.mu.pi.beta)
    result += -distr.gamma_logpdf(
        lp, parameters.mu.lp.alpha, parameters.mu.lp.beta)
    result += -distr.gamma_logpdf(
        ln, parameters.mu.ln.alpha, parameters.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    result += np.sum(-distr.expmixture_logpdf(
        mu, pi, lp, ln))

    # sum_i P(s2[i] | s2_a, s2_b)
    result += np.sum(-distr.invgamma_logpdf(
        s2, parameters.s2.alpha, parameters.s2.beta))

    # sum_i sum_j sum_y P(t[i, j] | y) P(y | x)
    repeated_mu = np.repeat(mu, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model7_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (pi, lp, ln, mu, s2, w) = \
        split(weights, 1, 1, 1, pair_count, 1, feature_count)

    dpi = np.zeros_like(pi)
    dlp = np.zeros_like(lp)
    dln = np.zeros_like(ln)
    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(pi, lp, ln | ...)
    dpi += -distr.beta_logpdf_dx(
        pi, parameters.mu.pi.alpha, parameters.mu.pi.beta)
    dlp += -distr.gamma_logpdf_dx(
        lp, parameters.mu.lp.alpha, parameters.mu.lp.beta)
    dln += -distr.gamma_logpdf_dx(
        ln, parameters.mu.ln.alpha, parameters.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    dmu += -distr.expmixture_logpdf_dx(mu, pi, lp, ln)
    dpi += np.sum(-distr.expmixture_logpdf_dpi(
        mu, pi, lp, ln))
    dlp += np.sum(-distr.expmixture_logpdf_dlp(
        mu, pi, lp, ln))
    dln += np.sum(-distr.expmixture_logpdf_dln(
        mu, pi, lp, ln))

    # sum_i P(s2[i] | s2_a, s2_b)
    ds2 += np.sum(-distr.invgamma_logpdf_dx(
        s2, parameters.s2.alpha, parameters.s2.beta))

    # sum_i sum_j sum_y P(t[i, j] | y) P(y | x)
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
        (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, s2),
                   parameters.sample_counts)
    ds2 += np.sum(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, s2))

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dpi, dlp, dln, dmu, ds2, dw))


# --- Model #8:
#
# P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
#                          I[y=0] U(t[i], 1/tau)
# P(mu[i] | π, lp, ln) = I[x>=0] π Exp(mu[i] | lp) +
#                        I[x<0] (1 - π) Exp(-mu[i] | ln)
# P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
# P(π | π.alpha, π.beta) = Beta(π | π.alpha, π.beta)
# P(λ_p | λ_p.alpha, λ_p.beta) = Gamma(λ_p | λ_p.alpha, λ_p.beta)
# P(λ_n | λ_n.alpha, λ_n.beta) = Gamma(λ_n | λ_n.alpha, λ_n.beta)
# P(w | l2) = N(w | 0, 1/l2)
#
# Weights are π, λ_p, λ_n, mu, s2, w.
# s2 is fixed.
# Other parameters are fixed.

Model8Parameters = namedtuple(
    'Model8Parameters', 'tau mu s2 l2 t xr sample_counts')


def model8_obj(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (pi, lp, ln, mu, w) = \
        split(weights, 1, 1, 1, pair_count, feature_count)

    result = 0

    # P(w | l2)
    result += w.T.dot(parameters.l2 * w)

    # P(pi, lp, ln | ...)
    result += -distr.beta_logpdf(
        pi, parameters.mu.pi.alpha, parameters.mu.pi.beta)
    result += -distr.gamma_logpdf(
        lp, parameters.mu.lp.alpha, parameters.mu.lp.beta)
    result += -distr.gamma_logpdf(
        ln, parameters.mu.ln.alpha, parameters.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    result += np.sum(-distr.expmixture_logpdf(
        mu, pi, lp, ln))

    # sum_i sum_j sum_y P(t[i, j] | y) P(y | x)
    repeated_mu = np.repeat(mu, parameters.sample_counts)

    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, parameters.s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        result += parameters.xr.coefficient * \
            kl_term(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, parameters.t))

    return result


def model8_grad(weights, parameters, x, dt, u):
    pair_count = parameters.sample_counts.shape[0]
    feature_count = parameters.l2.shape[0]

    (pi, lp, ln, mu, w) = split(weights, 1, 1, 1, pair_count, feature_count)

    dpi = 0
    dlp = 0
    dln = 0
    dmu = np.zeros_like(mu)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, parameters.sample_counts)

    # P(w | l2)
    dw += 2 * parameters.l2 * w

    # P(pi, lp, ln | ...)
    dpi += -distr.beta_logpdf_dx(
        pi, parameters.mu.pi.alpha, parameters.mu.pi.beta)
    dlp += -distr.gamma_logpdf_dx(
        lp, parameters.mu.lp.alpha, parameters.mu.lp.beta)
    dln += -distr.gamma_logpdf_dx(
        ln, parameters.mu.ln.alpha, parameters.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    dmu += -distr.expmixture_logpdf_dx(mu, pi, lp, ln)
    dpi += np.sum(-distr.expmixture_logpdf_dpi(
        mu, pi, lp, ln))
    dlp += np.sum(-distr.expmixture_logpdf_dlp(
        mu, pi, lp, ln))
    dln += np.sum(-distr.expmixture_logpdf_dln(
        mu, pi, lp, ln))

    # sum_i sum_j sum_y P(t[i, j] | y) P(y | x)
    pt_y0 = -np.log(parameters.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, parameters.s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / parameters.t) * x.T.dot((f - 1) * g1) + \
        (1 / parameters.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu,
                                                 parameters.s2),
                   parameters.sample_counts)

    # Unlabeled samples.
    if parameters.xr.coefficient != 0:
        # XR term.
        dw += parameters.xr.coefficient * \
            kl_grad(w, u, parameters.t, parameters.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, parameters.t)) / parameters.t

    return np.concatenate((dpi, dlp, dln, dmu, dw))


# --- Model #9:
#
# P(t[i] | mu[i], λ[i], y) = I[y=1] ExpMixture(t[i] | mu[i], λ[i]) +
#                            I[y=0] U(t[i], 1/tau)
# P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
#
# P(mu[i] | π, lp, ln) = I[x>=0] π Exp(mu[i] | lp) +
#                        I[x<0] (1 - π) Exp(-mu[i] | ln)
# + P(λ[i] | λ.alpha, λ.beta)    = Gamma(λ[i] | λ.alpha, λ.beta)
#
# + P(π | π.alpha, π.beta) = Beta(π | π.alpha, π.beta)
# + P(λ_p | λ_p.alpha, λ_p.beta) = Gamma(λ_p | λ_p.alpha, λ_p.beta)
# + P(λ_n | λ_n.alpha, λ_n.beta) = Gamma(λ_n | λ_n.alpha, λ_n.beta)
# + P(w | l2) = N(w | 0, 1/l2)
#
# Weights are π, λ_p, λ_n, mu, λ, w.
# Other parameters are fixed.

Model9Parameters = namedtuple(
    'Model9Parameters', 'tau mu lam l2 t xr sample_counts')


def model9_obj(weights, params, x, dt, u):
    pair_count = params.sample_counts.shape[0]
    feature_count = params.l2.shape[0]

    (pi, lp, ln, mu, lam, w) = \
        split(weights, 1, 1, 1,
              pair_count, pair_count, feature_count)

    result = 0

    # P(w | l2)
    result += -distr.l2_log(w, params.l2)

    # P(pi, lp, ln | ...)
    result += -distr.beta_logpdf(pi, params.mu.pi.alpha, params.mu.pi.beta)
    result += -distr.gamma_logpdf(lp, params.mu.lp.alpha, params.mu.lp.beta)
    result += -distr.gamma_logpdf(ln, params.mu.ln.alpha, params.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    result += np.sum(-distr.expmixture_logpdf(mu, pi, lp, ln))

    # P(λ[i] | λ.alpha, λ.beta)    = Gamma(λ[i] | λ.alpha, λ.beta)
    result += np.sum(-distr.gamma_logpdf(lam, params.lam.alpha, params.lam.beta))

    # P(t[i] | mu[i], λ[i], y) = I[y=1] ExpMixture(t[i] | mu[i], λ[i]) +
    #                            I[y=0] U(t[i], 1/tau)
    # P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu  = np.repeat(mu, params.sample_counts)
    repeated_lam = np.repeat(lam, params.sample_counts)

    pt_y0 = -np.log(params.tau)
    pt_y1 = distr.symmexp_logpdf(dt, repeated_mu, repeated_lam)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if params.xr.coefficient != 0:
        # XR term.
        result += params.xr.coefficient * \
            kl_term(w, u, params.t, params.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, params.t))

    return result


def model9_grad(weights, params, x, dt, u):
    pair_count = params.sample_counts.shape[0]
    feature_count = params.l2.shape[0]

    (pi, lp, ln, mu, lam, w) = \
        split(weights, 1, 1, 1, pair_count, pair_count, feature_count)

    dpi = np.zeros_like(pi)
    dlp = np.zeros_like(lp)
    dln = np.zeros_like(ln)
    dmu = np.zeros_like(mu)
    dlam = np.zeros_like(lam)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, params.sample_counts)
    repeated_lam = np.repeat(lam, params.sample_counts)

    # P(w | l2)
    dw += -distr.l2_log_dw(w, params.l2)

    # P(pi, lp, ln | ...)
    dpi += -distr.beta_logpdf_dx(pi, params.mu.pi.alpha, params.mu.pi.beta)
    dlp += -distr.gamma_logpdf_dx(lp, params.mu.lp.alpha, params.mu.lp.beta)
    dln += -distr.gamma_logpdf_dx(ln, params.mu.ln.alpha, params.mu.ln.beta)

    # sum_i P(mu[i] | π, λ)
    dmu += -distr.expmixture_logpdf_dx(mu, pi, lp, ln)
    dpi += np.sum(-distr.expmixture_logpdf_dpi(mu, pi, lp, ln))
    dlp += np.sum(-distr.expmixture_logpdf_dlp(mu, pi, lp, ln))
    dln += np.sum(-distr.expmixture_logpdf_dln(mu, pi, lp, ln))

    # P(λ[i] | λ.alpha, λ.beta)    = Gamma(λ[i] | λ.alpha, λ.beta)
    dlam += -distr.gamma_logpdf_dx(lam, params.lam.alpha, params.lam.beta)

    # P(t[i] | mu[i], λ[i], y) = I[y=1] ExpMixture(t[i] | mu[i], λ[i]) +
    #                            I[y=0] U(t[i], 1/tau)
    # P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(params.tau)
    pt_y1 = distr.symmexp_logpdf(dt, repeated_mu, repeated_lam)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / params.t) * x.T.dot((f - 1) * g1) + \
        (1 / params.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.symmexp_logpdf_dmu(dt, repeated_mu, repeated_lam),
                   params.sample_counts)
    dlam += subsums(-g1 * distr.symmexp_logpdf_dlam(dt, repeated_mu, repeated_lam),
                   params.sample_counts)

    # Unlabeled samples.
    if params.xr.coefficient != 0:
        # XR term.
        dw += params.xr.coefficient * \
            kl_grad(w, u, params.t, params.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, params.t)) / params.t

    return np.concatenate((dpi, dlp, dln, dmu, dlam, dw))


Model10Parameters = namedtuple(
    'Model10Parameters', 'tau Mu s2 l2 t xr sample_counts')


def model10_obj(weights, params, x, dt, u):
    pair_count = params.sample_counts.shape[0]
    feature_count = params.l2.shape[0]

    (Mu, S2, mu, s2, w) = \
        split(weights,
              1, 1, pair_count, pair_count, feature_count)
    Mu, S2, s2 = np.squeeze(Mu), np.squeeze(S2), np.squeeze(s2)

    result = 0

    # P(w | l2) = N(w | 0, 1/l2)
    result += w.T.dot(params.l2 * w)

    # P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
    result += -distr.normal_logpdf(Mu, params.Mu.mu, params.Mu.s2)

    # P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
    result += np.sum(-distr.normal_logpdf(mu, Mu, S2))

    # P(s2[i] | s2.alpha, s2.beta) = InvGamma(s2[i] | s2.alpha, s2.beta)
    result += np.sum(-distr.invgamma_logpdf(s2, params.s2.alpha, params.s2.beta))

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu = np.repeat(mu, params.sample_counts)
    repeated_s2 = np.repeat(s2, params.sample_counts)

    pt_y0 = -np.log(params.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if params.xr.coefficient != 0:
        # XR term.
        result += params.xr.coefficient * \
            kl_term(w, u, params.t, params.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, params.t))

    return result


def model10_grad(weights, params, x, dt, u):
    pair_count = params.sample_counts.shape[0]
    feature_count = params.l2.shape[0]

    (Mu, S2, mu, s2, w) = \
        split(weights,
              1, 1, pair_count, pair_count, feature_count)

    dMu = np.zeros_like(Mu)
    dS2 = np.zeros_like(S2)
    dmu = np.zeros_like(mu)
    ds2 = np.zeros_like(s2)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, params.sample_counts)
    repeated_s2 = np.repeat(s2, params.sample_counts)

    # P(w | l2)
    dw += 2 * params.l2 * w

    # P(Mu | Mu0, Sigma0) = N(Mu | Mu0, Sigma0)
    dMu += -distr.normal_logpdf_dx(Mu, params.Mu.mu, params.Mu.s2)

    # P(mu[i] | Mu, Sigma) = N(mu[i] | Mu, Sigma)
    dmu += -distr.normal_logpdf_dx(mu, Mu, S2)
    dMu += np.sum(-distr.normal_logpdf_dmu(mu, Mu, S2))
    dS2 += np.sum(-distr.normal_logpdf_ds2(mu, Mu, S2))

    # P(s2[i] | s2.alpha, s2.beta) = InvGamma(s2[i] | s2.alpha, s2.beta)
    ds2 += -distr.invgamma_logpdf_dx(
        s2, params.s2.alpha, params.s2.beta)

    # P(t[i] | mu[i], s2, y) = I[y=1] N(t[i] | mu[i], s2) +
    #                          I[y=0] U(t[i], 1/tau)
    # P(y | x, w, t) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(params.tau)
    pt_y1 = distr.normal_logpdf(dt, repeated_mu, repeated_s2)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += (1 / params.t) * x.T.dot((f - 1) * g1) + \
          (1 / params.t) * x.T.dot((f - 0) * g0)
    dmu += subsums(-g1 * distr.normal_logpdf_dmu(dt, repeated_mu, repeated_s2),
                   params.sample_counts)
    ds2 += subsums(-g1 * distr.normal_logpdf_ds2(dt, repeated_mu, repeated_s2),
                   params.sample_counts)

    # Unlabeled samples.
    if params.xr.coefficient != 0:
        # XR term.
        dw += params.xr.coefficient * \
            kl_grad(w, u, params.t, params.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, params.t)) / params.t

    return np.concatenate((dMu, dS2, dmu, ds2, dw))

Model11Parameters = namedtuple(
    'Model11Parameters', 'tau l1 l2 k xr sample_counts')


def model11_obj(weights, params, x, dt, u):
    pair_count = params.sample_counts.shape[0]
    feature_count = params.l2.shape[0]

    (pi, lp, ln, mu, delta, w) = \
        split(weights, 1, 1, 1, pair_count, pair_count, feature_count)

    result = 0

    # P(w | l2) + P(w | l1)
    result += -distr.l1_log(w, params.l1)
    result += -distr.l2_log(w, params.l2)

    # # P(mu[i] | π, λ)
    result += np.sum(-distr.expmixture_logpdf(mu, pi, lp, ln))

    # P(t[i] | mu[i], λ[i], y) = I[y=1] GGD(t[i] | mu[i], Δ[i], k) +
    #                            I[y=0] U(t[i], 1/tau)
    # P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
    repeated_mu  = np.repeat(mu, params.sample_counts)
    repeated_delta = np.repeat(delta, params.sample_counts)
    ggd_params = (dt, repeated_mu, repeated_delta, params.k)

    pt_y0 = -np.log(params.tau)
    pt_y1 = distr.ggd_logpdf(*ggd_params)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)
    result += np.sum(-z)

    # Unlabeled samples.
    if params.xr.coefficient != 0:
        # XR term.
        result += params.xr.coefficient * \
            kl_term(w, u, 1.0, params.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        result += np.sum(-lr_log_probability(w, u, -1, 1.0))

    return result


def model11_grad(weights, params, x, dt, u):
    pair_count = params.sample_counts.shape[0]
    feature_count = params.l2.shape[0]

    (pi, lp, ln, mu, delta, w) = \
        split(weights, 1, 1, 1, pair_count, pair_count, feature_count)

    dpi = np.zeros_like(pi)
    dlp = np.zeros_like(lp)
    dln = np.zeros_like(ln)
    dmu = np.zeros_like(mu)
    ddelta = np.zeros_like(delta)
    dw = np.zeros_like(w)

    repeated_mu = np.repeat(mu, params.sample_counts)
    repeated_delta = np.repeat(delta, params.sample_counts)
    ggd_params = (dt, repeated_mu, repeated_delta, params.k)

    # P(w | l2) + P(w | l1)
    dw += -distr.l1_log_dw(w, params.l1)
    dw += -distr.l2_log_dw(w, params.l2)

    # # sum_i P(mu[i] | π, λ)
    dmu += -distr.expmixture_logpdf_dx(mu, pi, lp, ln)
    dpi += np.sum(-distr.expmixture_logpdf_dpi(mu, pi, lp, ln))
    dlp += np.sum(-distr.expmixture_logpdf_dlp(mu, pi, lp, ln))
    dln += np.sum(-distr.expmixture_logpdf_dln(mu, pi, lp, ln))

    # P(t[i] | mu[i], λ[i], y) = I[y=1] GGD(t[i] | mu[i], Δ[i], k) +
    #                            I[y=0] U(t[i], 1/tau)
    # P(y | x, w) = 1 / (1 + exp(w.dot(x) / t))
    pt_y0 = -np.log(params.tau)
    pt_y1 = distr.ggd_logpdf(*ggd_params)
    py1_x = lr_log_probability(w, x)
    py0_x = lr_log_probability(w, -x)
    z = np.logaddexp(pt_y0 + py0_x, pt_y1 + py1_x)

    g0 = np.exp(pt_y0 + py0_x - z)
    g1 = np.exp(pt_y1 + py1_x - z)
    f = np.exp(py1_x)

    dw += x.T.dot((f - 1) * g1) + x.T.dot((f - 0) * g0)

    dmu += subsums(-g1 * distr.ggd_logpdf_dmu(*ggd_params),
                   params.sample_counts)
    ddelta += subsums(-g1 * distr.ggd_logpdf_ddelta(*ggd_params),
                      params.sample_counts)

    # Unlabeled samples.
    if params.xr.coefficient != 0:
        # XR term.
        dw += params.xr.coefficient * \
            kl_grad(w, u, 1.0, params.xr.probability)
    else:
        # No XR term, assume that all unlabeled samples are negative.
        dw += u.T.dot(lr_probability(w, u, 1.0))

    return np.concatenate((dpi, dlp, dln, dmu, ddelta, dw))
