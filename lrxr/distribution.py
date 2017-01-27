#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.special as special

from collections import namedtuple
from numpy_testing import NumpyTest

def distr_scale(v, s):
    if isinstance(v, float):
        return v * s
    elif isinstance(v, Normal):
        return normal_scale(v, s)
    elif isinstance(v, InvGamma):
        return invgamma_scale(v, s)
    elif isinstance(v, ExpMixture):
        return expmixture_scale(v, s)
    elif isinstance(v, Gamma):
        return gamma_scale(v, s)
    else:
        assert False


def lr_log(w, x, y):
    """ Logistic regression """
    if not isinstance(x, scipy.sparse.spmatrix):
        x = np.asarray(x)
    w, y = np.asarray(w), np.asarray(y)
    return -np.logaddexp(0, -y * x.dot(w))

def lr_prob(w, x, y):
    return np.exp(lr_log(w, x, y))

def lr_log_dw(w, x, y):
    if not isinstance(x, scipy.sparse.spmatrix):
        x = np.asarray(x)
    w, y = np.asarray(w), np.asarray(y)

    y1 = lr_prob(w, x, 1)
    y0 = (y + 1) / 2
    return x.T.dot(y1 - y0)

class TestLR(NumpyTest):
    def setUp(self):
        super(TestLR, self).setUp()
        self.w = self.rng.randn(100)
        self.x = self.rng.randn(20, 100)
        self.y = self.rng.choice([1, -1], 20)

    def test_lr_log(self):
        w = [1, 2, -1, 4]
        x = [[0, 1, 2, 1], [-1, -2, 3, -1], [3, 0, 0, 4]]
        y = [1, -1, -1]

        l = lr_log(w, x, y)
        p = lr_prob(w, x, y)

        b = [-np.log(1 + np.exp(-4)),
             -np.log(1 + np.exp(-12)),
             -np.log(1 + np.exp(19))]
        self.assertAlmostEqual(l, b)
        self.assertAlmostEqual(p, np.exp(b))

    def test_lr_log_dw_1(self):
        self.checkGrad(
            f=lambda w: lr_log(w, self.x[0,:], self.y[0]),
            g=lambda w: lr_log_dw(w, self.x[0,:], self.y[0]),
            x0=self.w, atol=1e-5)

    def test_lr_log_dw(self):
        self.checkGrad(
            f=lambda w: np.sum(lr_log(w, self.x, self.y)),
            g=lambda w: lr_log_dw(w, self.x, self.y),
            x0=self.w, atol=1e-5)

def l2_log(w, l2):
    """ L2-regularization """
    w, l2 = np.asarray(w), np.asarray(l2)
    return -w.dot(l2 * w)

def l2_log_dw(w, l2):
    w, l2 = np.asarray(w), np.asarray(l2)
    return -2 * l2 * w

def l1_log(w, l1):
    """ L1-regularization """
    w, l1 = np.asarray(w), np.asarray(l1)
    return -np.sum(l1 * np.abs(w))

def l1_log_dw(w, l1):
    w, l1 = np.asarray(w), np.asarray(l1)
    return -l1 * np.sign(w)

class TestL2L1(NumpyTest):
    def setUp(self):
        super(TestL2L1, self).setUp()
        self.w = self.rng.randn(100)
        self.l2 = np.abs(self.rng.randn(100))
        self.l1 = np.abs(self.rng.randn(100))

    def test_l2(self):
        self.checkGrad(
            f=lambda w: l2_log(w, self.l2),
            g=lambda w: l2_log_dw(w, self.l2),
            x0=self.w, atol=1e-5)

    def test_l1(self):
        self.checkGrad(
            f=lambda w: l1_log(w, self.l1),
            g=lambda w: l1_log_dw(w, self.l1),
            x0=self.w, atol=1e-5)

GGD = namedtuple('GGD', 'mu delta k')

def ggd_logpdf(x, mu, delta, k):
    return \
        -np.log(2 * delta) +\
        -scipy.special.gammaln((k + 1) / k) +\
        -np.log(k) / k +\
        -(np.abs(x - mu) / delta) ** k / k


def ggd_pdf(x, mu, delta, k):
    return np.exp(ggd_logpdf(x, mu, delta, k))


def ggd_logpdf_dx(x, mu, delta, k):
    x0 = x - mu
    return -np.sign(x0) * np.abs(x0) ** (k - 1) / delta**k


def ggd_logpdf_dmu(x, mu, delta, k):
    x0 = x - mu
    return np.sign(x0) * np.abs(x0) ** (k - 1) / delta**k


def ggd_logpdf_ddelta(x, mu, delta, k):
    x0 = x - mu
    return ((np.abs(x0) / delta)**k - 1) / delta


def ggd_scale(s, mu, delta, k):
    return GGD(
        mu=mu * s,
        delta=delta * s,
        k=k)

def ggd_estimate_weighted(x, w, k):
    """
    Maximum likelihood estimator for a generalized gaussian distribution with
    given 'k'.
    """

    x = np.asarray(x)
    w = np.asarray(w)
    equal_weights = len(w.shape) == 0
    if equal_weights:
        w = np.ones_like(x) * w

    mu0 = np.sum(w * x) / np.sum(w)

    if k < 1:
        # A matrix of pairwise distances.
        d = np.abs(x[:, np.newaxis] - x) ** k
        i = np.argmin(np.sum(w * d, axis=1))
        mu = x[i]
    elif k == 1 and equal_weights:
        mu = np.median(x)
    elif k == 2:
        mu = mu0
    else:
        result = scipy.optimize.minimize(
            lambda mu: np.sum(w * np.abs(mu - x) ** k), mu0)
        if not result.success:
            raise Error("Could not find a solution for mu.")
        mu = result.x

    x0 = x - mu
    delta = (np.sum(w * np.abs(x0) ** k) / np.sum(w)) ** (1/k)
    return GGD(mu, delta, k)


class TestGGD(NumpyTest):
    def setUp(self):
        super(TestGGD, self).setUp()
        self.x = self.rng.randn(10, 30)
        self.mu = self.rng.randn(30)
        self.delta = self.rng.rand(30) + 1

    def testPDF(self):
        self.assertAlmostEqual(
            ggd_pdf(x=0, mu=0, delta=1, k=2), 1 / np.sqrt(2 * np.pi))
        self.assertAlmostEqual(
            ggd_pdf(x=0, mu=1, delta=1, k=2), np.exp(-1/2) / np.sqrt(2 * np.pi))

        self.assertAlmostEqual(
            ggd_pdf(x=0, mu=0, delta=1, k=1), 1 / 2)
        self.assertAlmostEqual(
            ggd_pdf(x=2, mu=2, delta=1, k=1), 1 / 2)
        self.assertAlmostEqual(
            ggd_pdf(x=2, mu=1, delta=1, k=1), np.exp(-1) / 2)

        # Calculated using mathematica.
        self.assertAlmostEqual(
            ggd_pdf(x=3/10, mu=1/2, delta=2, k=3/2), 0.2069306737)

    def testDerivativesMathematica(self):
        self.assertAlmostEqual(
            ggd_logpdf_dx(x=3/10, mu=1/2, delta=2, k=3/2),
            0.1581138830)
        self.assertAlmostEqual(
            ggd_logpdf_dmu(x=3/10, mu=1/2, delta=2, k=3/2),
            -0.1581138830)
        self.assertAlmostEqual(
            ggd_logpdf_ddelta(x=3/10, mu=1/2, delta=2, k=3/2),
            -0.4841886117)

    def testXDerivative1(self):
        x, mu, delta = self.x[:, 0], self.mu[0], self.delta[0]
        for k in [1, 2, 3]:
            self.checkGrad(
                f=lambda x: np.sum(ggd_logpdf(x, mu, delta, k)),
                g=lambda x: ggd_logpdf_dx(x, mu, delta, k),
                x0=x, atol=1e-5)

    def testMuDerivative1(self):
        x, mu, delta = self.x[0, :], self.mu, self.delta[0]
        for k in [1, 2, 3]:
            self.checkGrad(
                f=lambda mu: np.sum(ggd_logpdf(x, mu, delta, k)),
                g=lambda mu: ggd_logpdf_dmu(x, mu, delta, k),
                x0=np.array(mu), atol=1e-5)

    def testDeltaDerivative1(self):
        x, mu, delta = self.x[0, :], self.mu[0], self.delta
        for k in [1, 2, 3]:
            self.checkGrad(
                f=lambda delta: np.sum(ggd_logpdf(x, mu, delta, k)),
                g=lambda delta: ggd_logpdf_ddelta(x, mu, delta, k),
                x0=delta, atol=1e-5)

    def testEstimate(self):
        x = [1, 2, 3, 3/2, 4/3, 2/3]
        w = [1, 2, 3, 1, 1, 1]

        mu, delta, _ = ggd_estimate_weighted(x, 1, 1)
        self.assertAlmostEqual(mu, np.median(x), rtol=1e-4)
        self.assertAlmostEqual(delta, 0.5833333332, rtol=1e-4)

        mu, delta, _ = ggd_estimate_weighted(x, 1, 2)
        self.assertAlmostEqual(mu, np.mean(x), rtol=1e-4)
        self.assertAlmostEqual(delta, np.sqrt(np.var(x)), rtol=1e-4)

        mu, delta, _ = ggd_estimate_weighted(x, w, 2)
        self.assertAlmostEqual(mu, 1.944435324, rtol=1e-4)
        self.assertAlmostEqual(delta, 0.8461930536, rtol=1e-4)

        mu, delta, _ = ggd_estimate_weighted(x, 1, 2.5)
        self.assertAlmostEqual(mu, 1.655933035, rtol=1e-4)
        self.assertAlmostEqual(delta, 0.8110388682, rtol=1e-4)

        mu, delta, _ = ggd_estimate_weighted(x, w, 2.5)
        self.assertAlmostEqual(mu, 1.957936790, rtol=1e-4)
        self.assertAlmostEqual(delta, 0.8816749234, rtol=1e-4)

        mu, delta, _ = ggd_estimate_weighted(x, 1, 3)
        self.assertAlmostEqual(mu, 1.706116204, rtol=1e-4)
        self.assertAlmostEqual(delta, 0.8532530144, rtol=1e-4)

        mu, delta, _ = ggd_estimate_weighted(x, w, 3)
        self.assertAlmostEqual(mu, 1.963766328, rtol=1e-4)
        self.assertAlmostEqual(delta, 0.9092719010, rtol=1e-4)


Normal = namedtuple('Normal', 'mu s2')


def normal_pdf(x, mu, s2):
    """Gaussian pdf."""
    return 1 / np.sqrt(2 * np.pi * s2) * \
        np.exp(-(x - mu) ** 2 / (2 * s2))


def normal_logpdf(x, mu, s2):
    """Gaussian log pdf."""
    return -(1/2) * np.log(2 * np.pi * s2) - \
        (x - mu) ** 2 / (2 * s2)


def normal_logpdf_dx(x, mu, s2):
    """Gaussian log pdf."""
    return -(x - mu) / s2


def normal_logpdf_dmu(x, mu, s2):
    """Gaussian log pdf."""
    return (x - mu) / s2


def normal_logpdf_ds2(x, mu, s2):
    """Gaussian log pdf."""
    return ((x - mu) ** 2 - s2) / (2 * s2 ** 2)


def normal_estimate_weighted(x, w):
    """Maximum likelihood estimator for a weighted gaussian distribution."""
    mu = np.sum(w * x) / np.sum(w)
    s2 = np.sum(w * (x - mu)**2) / np.sum(w)
    return mu, s2


def normal_scale(d, s):
    return Normal(
        mu=d.mu * s,
        s2=distr_scale(d.s2, s**2))


InvGamma = namedtuple('InvGamma', 'alpha beta')


def invgamma_logpdf(x, alpha, beta):
    """Inverse-Gamma distribution."""
    return alpha * np.log(beta) + \
        -scipy.special.gammaln(alpha) + \
        -(alpha + 1) * np.log(x) + \
        -beta / x


def invgamma_logpdf_dx(x, alpha, beta):
    """Inverse-Gamma distribution."""
    return -(alpha + 1) / x + beta / x**2


def invgamma_from_mean_var(mean, var):
    """Inverse-Gamma distribution."""
    alpha = (mean ** 2 + 2 * var) / var
    beta = (mean * (mean ** 2 + var)) / var
    return InvGamma(alpha, beta)


def invgamma_scale(d, s):
    return InvGamma(
        alpha=d.alpha,
        beta=distr_scale(d.beta, s))


ExpMixture = namedtuple('ExpMixture', 'pi lp ln')


def expmixture_logpdf(x, pi, lp, ln):
    return np.where(
        x >= 0,
        np.log(pi) + np.log(lp) - lp * x,
        np.log(1 - pi) + np.log(ln) + ln * x)


def expmixture_pdf(x, pi, lp, ln):
    return np.exp(expmixture_logpdf(x, pi, lp, ln))


def expmixture_logpdf_dx(x, pi, lp, ln):
    return np.where(x >= 0, -lp, ln)


def expmixture_logpdf_dpi(x, pi, lp, ln):
    return np.where(x >= 0, 1 / pi, -1 / (1 - pi))


def expmixture_logpdf_dlp(x, pi, lp, ln):
    return np.where(x >= 0, 1 / lp - x, 0)


def expmixture_logpdf_dln(x, pi, lp, ln):
    return np.where(x >= 0, 0, 1 / ln + x)


def expmixture_estimate_weighted(x, w):
    """Maximum likelihood estimator for a mixture of exponentials."""
    wp = np.compress(x >= 0, w)
    wn = np.compress(x < 0, w)
    xp = np.compress(x >= 0, x)
    xn = -np.compress(x < 0, x)

    pi = np.sum(wp) / np.sum(w)
    lp = np.sum(wp) / np.sum(xp * wp)
    ln = np.sum(wn) / np.sum(xn * wn)
    return ExpMixture(pi, lp, ln)


def expmixture_scale(d, s):
    return ExpMixture(
        pi=d.pi,
        lp=distr_scale(d.lp, 1/s),
        ln=distr_scale(d.ln, 1/s))


class TestExpMixture(NumpyTest):
    def setUp(self):
        super(TestExpMixture, self).setUp()
        self.samples = 10
        self.params = 30

        self.x = self.rng.randn(self.samples, self.params)
        self.pi = self.rng.rand(self.params) * 0.6 + 0.2
        self.lp = self.rng.rand(self.params) + 1
        self.ln = self.rng.rand(self.params) + 1

    def testShape(self):
        self.assertEqual(
            expmixture_pdf(x=self.x, pi=self.pi, lp=self.lp, ln=self.ln).shape,
            self.x.shape)
        self.assertEqual(
            expmixture_pdf(x=1, pi=self.pi, lp=self.lp, ln=self.ln).shape,
            self.pi.shape)
        self.assertEqual(
            expmixture_pdf(x=1, pi=self.pi, lp=2, ln=3).shape,
            self.pi.shape)
        self.assertEqual(
            expmixture_pdf(x=self.x[:,0], pi=0.3, lp=2, ln=3).shape,
            self.x[:,0].shape)

        self.assertEqual(
            expmixture_logpdf_dx(x=self.x[:,0], pi=0.3, lp=2, ln=3).shape,
            self.x[:,0].shape)
        self.assertEqual(
            expmixture_logpdf_dpi(x=1, pi=self.pi, lp=2, ln=3).shape,
            self.pi.shape)

    def testPDF(self):
        self.assertAlmostEqual(
            expmixture_pdf(x=1, pi=1/2, lp=1, ln=1),
            1/2 * np.exp(-1))
        self.assertAlmostEqual(
            expmixture_pdf(x=-1, pi=1/2, lp=1, ln=1),
            1/2 * np.exp(-1))
        self.assertAlmostEqual(
            expmixture_pdf(x=1, pi=1/3, lp=1, ln=1),
            1/3 * np.exp(-1))
        self.assertAlmostEqual(
            expmixture_pdf(x=1, pi=1/3, lp=2, ln=1),
            2/3 * np.exp(-2))
        self.assertAlmostEqual(
            expmixture_pdf(x=-2, pi=1/3, lp=2, ln=2),
            4/3 * np.exp(-4))

    def testDerivativesNumerically(self):
        self.checkGrad(
            f=lambda x: np.sum(expmixture_logpdf(x, pi=1/3, lp=2, ln=2)),
            g=lambda x: expmixture_logpdf_dx(    x, pi=1/3, lp=2, ln=2),
            x0=np.array([-1/2, 1, -2, 3, -4, 5]), atol=1e-6)
        self.checkGrad(
            f=lambda x: np.sum(expmixture_logpdf(x=3, pi=x, lp=2, ln=2)),
            g=lambda x: expmixture_logpdf_dpi(   x=3, pi=x, lp=2, ln=2),
            x0=self.pi, atol=1e-5)
        self.checkGrad(
            f=lambda x: np.sum(expmixture_logpdf(x=3, pi=1/3, lp=x, ln=2)),
            g=lambda x: expmixture_logpdf_dlp(   x=3, pi=1/3, lp=x, ln=2),
            x0=self.lp, atol=1e-5)
        self.checkGrad(
            f=lambda x: np.sum(expmixture_logpdf(x=3, pi=1/3, lp=2, ln=x)),
            g=lambda x: expmixture_logpdf_dln(   x=3, pi=1/3, lp=2, ln=x),
            x0=self.ln, atol=1e-5)


SymmExp = namedtuple('SymmExp', 'mu lam')


def symmexp_logpdf(x, mu, lam):
    return np.log(lam / 2) - lam * np.abs(x - mu)


def symmexp_logpdf_dx(x, mu, lam):
    return -lam * np.sign(x - mu)


def symmexp_logpdf_dmu(x, mu, lam):
    return lam * np.sign(x - mu)


def symmexp_logpdf_dlam(x, mu, lam):
    return 1 / lam - np.abs(x - mu)


def symmexp_estimate_weighted(x, w):
    mu = np.median(x)
    lam = np.sum(w) / np.sum(np.abs(x - mu) * w)
    return SymmExp(mu, lam)


def symmexp_scale(d, s):
    return SymmExp(
        mu=distr_scale(d.lam, s),
        lam=distr_scale(d.lam, 1/s))


AsymmExp = namedtuple('SymmExp', 'mu pi lp ln')


def asymmexp_logpdf(x, mu, pi, lp, ln):
    return expmixture_logpdf(x - mu, pi, lp, ln)


def asymmexp_logpdf_dx(x, mu, lam):
    return expmixture_logpdf_dx(x - mu, pi, lp, ln)


def asymmexp_logpdf_dmu(x, mu, lam):
    return expmixture_logpdf(x - mu, pi, lp, ln)


def asymmexp_logpdf_dlam(x, mu, lam):
    return 1 / lam - np.abs(x - mu)


def asymmexp_estimate_weighted(x, w):
    mu = np.median(x)
    lam = np.sum(w) / np.sum(np.abs(x - mu) * w)
    return SymmExp(mu, lam)


def asymmexp_scale(d, s):
    return SymmExp(
        mu=distr_scale(d.lam, s),
        lam=distr_scale(d.lam, 1/s))


Beta = namedtuple('Beta', 'alpha beta')


def beta_logpdf(x, alpha, beta):
    lPx = special.xlog1py(beta - 1.0, -x) + \
        special.xlogy(alpha - 1.0, x)
    lPx -= special.betaln(alpha, beta)
    return lPx


def beta_pdf(x, alpha, beta):
    return np.exp(beta_logpdf(x, alpha, beta))


def beta_logpdf_dx(x, alpha, beta):
    return -(beta - 1.0) / (1 - x) + (alpha - 1.0) / x


def beta_from_mean_var(mean, var):
    x = (mean ** 2 - mean + var) / var
    return Beta(-mean * x, (mean - 1) * x)


class TestBeta(NumpyTest):
    def setUp(self):
        super(TestBeta, self).setUp()
        self.x = self.rng.rand(10, 30)
        self.alpha = self.rng.rand(30) + 1
        self.beta = self.rng.rand(30) + 1

    def testPDF(self):
        self.assertAlmostEqual(beta_pdf(x=0, alpha=2, beta=1/2), 0)
        self.assertAlmostEqual(beta_pdf(x=1/2, alpha=1/2, beta=1/2), 2/np.pi)
        self.assertAlmostEqual(beta_pdf(x=1/2, alpha=1, beta=1), 1)
        self.assertAlmostEqual(beta_pdf(x=1/2, alpha=2, beta=2), 3/2)

        self.assertAlmostEqual(
            beta_pdf(x=0.8004234217, alpha=1.9816004474, beta=1.5050812683),
            1.325883750)
        self.assertAlmostEqual(
            beta_pdf(x=0.4798047405, alpha=3.4568676737, beta=1.4885444138),
            0.941815081)

    def testDerivativesNumerically(self):
        self.checkGrad(
            f=lambda x: np.sum(beta_logpdf(x, alpha=3, beta=2)),
            g=lambda x: beta_logpdf_dx(x, alpha=3, beta=2),
            x0=np.array([1/2, 1/3, 2/3, 3/4, 1/4, 1/5]), atol=1e-6)
        self.checkGrad(
            f=lambda x: np.sum(beta_logpdf(x, alpha=1/2, beta=1/2)),
            g=lambda x: beta_logpdf_dx(x, alpha=1/2, beta=1/2),
            x0=np.array([1/2, 1/3, 2/3, 3/4, 1/4, 1/5]), atol=1e-6)


Gamma = namedtuple('Gamma', 'alpha beta')


def gamma_logpdf(x, alpha, beta):
    return \
        -special.gammaln(alpha) +\
        special.xlogy(alpha, beta) +\
        special.xlogy(alpha - 1, x) +\
        -beta * x


def gamma_pdf(x, alpha, beta):
    return np.exp(gamma_logpdf(x, alpha, beta))


def gamma_logpdf_dx(x, alpha, beta):
    return (alpha - 1) / x - beta


def gamma_from_mean_var(mean, var):
    return Gamma(mean ** 2 / var, mean / var)


def gamma_scale(d, s):
    return Gamma(
        alpha=d.alpha,
        beta=distr_scale(d.beta, 1/s))

class TestGamma(NumpyTest):
    def setUp(self):
        super(TestGamma, self).setUp()
        self.x = self.rng.randn(10, 30)
        self.alpha = self.rng.randn(30)
        self.beta = self.rng.rand(30) + 1

    def testPDF(self):
        self.assertAlmostEqual(gamma_pdf(x=1, alpha=2, beta=1/2),
                               1 / (4 * np.exp(1/2)))
        self.assertAlmostEqual(gamma_pdf(x=2, alpha=2, beta=1/2),
                               1 / (2 * np.exp(1)))
        self.assertAlmostEqual(gamma_pdf(x=2, alpha=1/2, beta=1/2),
                               1 / (2 * np.exp(1) * np.sqrt(np.pi)))

    def testDerivatives(self):
        self.assertAlmostEqual(gamma_logpdf_dx(x=2, alpha=1/2, beta=1/2), -3/4)
        self.assertAlmostEqual(gamma_logpdf_dx(x=2, alpha=1, beta=1/2), -1/2)
        self.assertAlmostEqual(gamma_logpdf_dx(x=1/2, alpha=2, beta=2), 0)

    def testDerivativesNumerically(self):
        self.checkGrad(
            f=lambda x: np.sum(gamma_logpdf(x, alpha=2, beta=2)),
            g=lambda x: gamma_logpdf_dx(x, alpha=2, beta=2),
            x0=np.array([1/2, 1, 2, 3, 4, 5]), atol=1e-6)

        self.checkGrad(
            f=lambda x: np.sum(gamma_logpdf(x, alpha=0.5, beta=0.5)),
            g=lambda x: gamma_logpdf_dx(x, alpha=0.5, beta=0.5),
            x0=np.array([1/2, 1, 2, 3, 4, 5]), atol=1e-6)

        self.checkGrad(
            f=lambda x: np.sum(gamma_logpdf(x, alpha=1.48218865207, beta=1.28928096607)),
            g=lambda x: gamma_logpdf_dx(x, alpha=1.48218865207, beta=1.28928096607),
            x0=np.array([1/2, 1, 2, 3, 4, 5, 0.1]), atol=1e-6)

# def gamma_logpdf(x, alpha, beta):
#     return special.xlogy(alpha - 1, x) - \
#         x / beta - special.gammaln(alpha) + \
#         special.xlogy(-alpha, beta)
#
#
# def gamma_pdf(x, alpha, beta):
#     return np.exp(gamma_logpdf(x, alpha, beta))
#
#
# def gamma_logpdf_dx(x, alpha, beta):
#     return (alpha - 1) / x - 1 / beta
#
#
# def gamma_from_mean_var(mean, var):
#     return Gamma(mean ** 2 / var, var / mean)
#
#
# def gamma_scale(d, s):
#     return Gamma(
#         alpha=d.alpha,
#         beta=distr_scale(d.beta, s))
#
#
# class TestGamma(NumpyTest):
#     def setUp(self):
#         super(TestGamma, self).setUp()
#         self.x = self.rng.randn(10, 30)
#         self.alpha = self.rng.randn(30)
#         self.beta = self.rng.rand(30) + 1
#
#     def testPDF(self):
#         self.assertAlmostEqual(gamma_pdf(x=1, alpha=2, beta=2),
#                                1 / (4 * np.exp(1/2)))
#         self.assertAlmostEqual(gamma_pdf(x=2, alpha=2, beta=2),
#                                1 / (2 * np.exp(1)))
#         self.assertAlmostEqual(gamma_pdf(x=2, alpha=1/2, beta=2),
#                                1 / (2 * np.exp(1) * np.sqrt(np.pi)))
#
#     def testDerivatives(self):
#         self.assertAlmostEqual(gamma_logpdf_dx(x=2, alpha=1/2, beta=2), -3/4)
#         self.assertAlmostEqual(gamma_logpdf_dx(x=2, alpha=1, beta=2), -1/2)
#         self.assertAlmostEqual(gamma_logpdf_dx(x=1/2, alpha=2, beta=1/2), 0)
#
#     def testDerivativesNumerically(self):
#         self.checkGrad(
#             f=lambda x: np.sum(gamma_logpdf(x, alpha=2, beta=2)),
#             g=lambda x: gamma_logpdf_dx(x, alpha=2, beta=2),
#             x0=np.array([1/2, 1, 2, 3, 4, 5]), atol=1e-6)
#
#         self.checkGrad(
#             f=lambda x: np.sum(gamma_logpdf(x, alpha=0.5, beta=0.5)),
#             g=lambda x: gamma_logpdf_dx(x, alpha=0.5, beta=0.5),
#             x0=np.array([1/2, 1, 2, 3, 4, 5]), atol=1e-6)
#
#         self.checkGrad(
#             f=lambda x: np.sum(gamma_logpdf(x, alpha=1.48218865207, beta=1.28928096607)),
#             g=lambda x: gamma_logpdf_dx(x, alpha=1.48218865207, beta=1.28928096607),
#             x0=np.array([1/2, 1, 2, 3, 4, 5, 2.81960494e-05]), atol=1e-6)
