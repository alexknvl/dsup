#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
if sys.version_info < (3, 0):
    import __builtin__
    range = __builtin__.xrange

import numpy as np
import scipy.optimize
import scipy.sparse

from numpy_testing import NumpyTest
import doctest
import unittest

from collections import namedtuple

#from lrxr import lr_logcdf, lr_logcdf_dw
from lrxr import lr_obj, lr_grad
from lrxr import lrxr_obj, lrxr_grad
from lrxr import kl_term, kl_grad

from lrxr import model1_obj, model1_grad, Model1Parameters
from lrxr import model2_obj, model2_grad, Model2Parameters
from lrxr import model3_obj, model3_grad, Model3Parameters
from lrxr import model4_obj, model4_grad, Model4Parameters
from lrxr import model5_obj, model5_grad, Model5Parameters
from lrxr import lrxrta5_obj, lrxrta5_grad
from lrxr import model6_obj, model6_grad, Model6Parameters
from lrxr import model7_obj, model7_grad, Model7Parameters
from lrxr import model8_obj, model8_grad, Model8Parameters
from lrxr import model9_obj, model9_grad, Model9Parameters
from lrxr import model10_obj, model10_grad, Model10Parameters
from lrxr import model11_obj, model11_grad, Model11Parameters
from lrxr import XR
from lrxr.distribution import Beta, InvGamma, Gamma, ExpMixture, Normal
import lrxr.distribution as distribution

def rand_partition(n, k, m=0, rng=None):
    rng = rng if rng is not None else np.random
    assert n >= k * m
    if m != 0:
        r = rand_partition(n - m * k, k, 0, rng)
        r += m
        return r

    s = set()
    for i in range(k-1):
        found = False
        while not found:
            x = rng.randint(1, n + (k - 1) + 1)
            if x not in s:
                found = True
                s.add(x)
    s = sorted(s)
    r = np.zeros(k, dtype=np.int32)
    r[0] = s[0] - 1
    for i in range(1, k - 1):
        r[i] = s[i] - s[i-1] - 1
    r[-1] = n + (k - 1) - s[-1]
    return r

def generate_xy_samples(rng, samples, dimensions,
                        positive_probability=0.5,
                        mean_scale=5.0,
                        class_scale=10.0):
    """
    Generates a list of features and labels (1 or -1) with the given number
    of samples and features.

    rng        - a random number generator.
    samples    - the number of samples to generate.
    dimensions - the number of dimensions each sample should have including
                 the bias feature.
    """
    x = np.zeros((samples, dimensions))
    y = np.zeros(samples)

    class1 = rng.randn(dimensions) * mean_scale
    class2 = rng.randn(dimensions) * mean_scale

    for i in range(samples):
        if rng.rand() < positive_probability:
            x[i, :] = rng.normal(class1, class_scale)
            y[i] = 1
        else:
            x[i, :] = rng.normal(class2, class_scale)
            y[i] = -1

    # Bias term
    x[:, -1] = 1
    return x, y


Data = namedtuple(
    'Data',
    ['y', 'x', 'u', 't', 'sample_counts', 'w',
     'mu_init', 's2_init', 's2_scalar_init', 'mu_expmixture_init',
     'Mu_init', 'S2_init', 'lam_init',
     'w_l2', 'w_l1',
     'tau0', 'scaler',
     'xr_params', 'Mu_params',
     'mu_normal_params',
     'mu_expmixture_params',
     'lam_params',
     's2_params'])

def generate_synthetic_data(rng,
                            labeled_samples=30,
                            unlabeled_samples=70,
                            labeled_pairs=5,
                            features=10,
                            time_scale=90.0):
    total_samples = labeled_samples + unlabeled_samples

    x, y = generate_xy_samples(
        rng, samples=total_samples, dimensions=features)

    # Forget labels for half the samples.
    y = y[:labeled_samples]
    x, u = x[:labeled_samples], x[labeled_samples:]
    # Generate time offsets for labeled samples.
    t = rng.randn(x.shape[0]) * time_scale
    # Partition labeled samples into "entity pair" buckets.
    sample_counts = rand_partition(x.shape[0], labeled_pairs, 1, rng=rng)

    tau0 = (rng.rand() + 1) * (max(t) - min(t))
    scaler = (rng.rand() + 1) * tau0 / 2

    w = rng.rand(features)
    w_l2 = np.zeros(features)
    w_l2[:-1] = rng.rand() + 1
    w_l2[-1] = rng.rand() + 1

    w = rng.rand(features)
    w_l1 = np.zeros(features)
    w_l1[:-1] = rng.rand() + 1
    w_l1[-1] = rng.rand() + 1

    mu_init = rng.randn(labeled_pairs) * time_scale
    s2_init = (rng.randn(labeled_pairs) * time_scale)**2
    s2_scalar_init = (rng.randn() * time_scale)**2
    Mu_init = rng.randn() * time_scale
    S2_init = (rng.randn() * time_scale) ** 2
    mu_expmixture_init = ExpMixture(
        rng.rand() * 0.6 + 0.2, 1 + rng.rand(), (1 + rng.rand()) * time_scale)
    lam_init = rng.rand(labeled_pairs) / (time_scale / 100)

    xr_params = XR(
        probability=rng.rand(),
        coefficient=((1 + rng.rand()) * 5) * labeled_samples)
    Mu_params = Normal(
        mu=rng.randn() * time_scale,
        s2=((rng.rand() + 1) * time_scale) ** 2)
    mu_normal_params = Normal(
        mu=rng.randn() * time_scale / 100.0,
        s2=((rng.rand() + 1) * time_scale / 100.0) ** 2)
    mu_expmixture_params = ExpMixture(
        pi=Beta(rng.rand() + 1, rng.rand() + 1),
        lp=Gamma(rng.rand() + 1, (rng.rand() + 1) * time_scale),
        ln=Gamma(rng.rand() + 1, (rng.rand() + 1) * time_scale))
    lam_params = Gamma(rng.rand() + 1, rng.rand() + 1)
    s2_params = InvGamma(rng.rand() + 1, rng.rand() + 1)

    return Data(
        y=y, x=x, u=u, t=t, sample_counts=sample_counts, w=w,
        mu_init=mu_init, s2_init=s2_init, Mu_init=Mu_init, S2_init=S2_init,
        s2_scalar_init=s2_scalar_init, mu_expmixture_init=mu_expmixture_init,
        lam_init=lam_init,
        w_l2=w_l2, w_l1=w_l1,
        tau0=tau0,
        scaler=scaler,
        xr_params=xr_params,
        Mu_params=Mu_params,
        mu_normal_params=mu_normal_params,
        mu_expmixture_params=mu_expmixture_params,
        lam_params=lam_params,
        s2_params=s2_params)

class TestLRXR(NumpyTest):
    def setUp(self):
        super(TestLRXR, self).setUp()
        self.data = generate_synthetic_data(self.rng)

    @property
    def basicData(self):
        # y, x, u, dt, acnt, w
        return (self.data.y, self.data.x, self.data.u, self.data.t,
                self.data.sample_counts, self.data.w)

    def testLRGradient(self):
        y, x, u, dt, acnt, w = self.basicData
        l2 = self.data.w_l2
        self.checkGrad(
            f=lambda w: lr_obj(w, x, y, l2),
            g=lambda w: lr_grad(w, x, y, l2),
            x0=w, rtol=1e-2)

    def testLRXRWithoutXRTerm(self):
        y, x, u, dt, acnt, w = self.basicData
        l2 = self.data.w_l2
        p_ex = self.data.xr_params.probability

        uy = np.ones(u.shape[0]) * (-1)
        fx = np.concatenate([x, u])
        fy = np.concatenate([y, uy])

        o1 = lr_obj(w, fx, fy, l2)
        o2 = lrxr_obj(w, x, y, u, l2, 1.0, p_ex, 0)
        self.assertAllClose(o1, o2)

        g1 = lr_grad(w, fx, fy, l2)
        g2 = lrxr_grad(w, x, y, u, l2, 1.0, p_ex, 0)
        self.assertAllClose(g1, g2)

    def testLRXRGradient(self):
        y, x, u, dt, acnt, w = self.basicData
        l2 = self.data.w_l2
        p_ex, xr = self.data.xr_params
        self.checkGrad(
            f=lambda w: lrxr_obj(w, x, y, u, l2, 1.0, p_ex, xr),
            g=lambda w: lrxr_grad(w, x, y, u, l2, 1.0, p_ex, xr),
            x0=w, rtol=1e-2)

    def testXRGradient(self):
        y, x, u, dt, acnt, w = self.basicData
        l2 = self.data.w_l2
        p_ex, xr = self.data.xr_params
        self.checkGrad(
            f=lambda w: kl_term(w, u, 1.0, p_ex),
            g=lambda w: kl_grad(w, u, 1.0, p_ex),
            x0=w, rtol=1e-2)

    def testLRXRTA5Gradient(self):
        y, x, u, dt, acnt, w = self.basicData
        l2 = self.data.w_l2
        p_ex, xr = self.data.xr_params
        tau0 = self.data.tau0
        MU_0, S2_0 = self.data.Mu_params
        alpha, beta = self.data.s2_params

        w1 = np.concatenate([self.data.mu_init, self.data.s2_init, w])
        params = (tau0, MU_0, S2_0, alpha, beta, l2, 1.0, p_ex, xr)
        self.checkGrad(
            f=lambda w: lrxrta5_obj(w, params, acnt, x, dt, u),
            g=lambda w: lrxrta5_grad(w, params, acnt, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel1Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            [self.data.Mu_init, self.data.S2_init],
            self.data.mu_init, [self.data.s2_scalar_init], w])
        params = Model1Parameters(
            tau=self.data.tau0,
            Mu=self.data.Mu_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model1_obj(w, params, x, dt, u),
            g=lambda w: model1_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel2Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            [self.data.Mu_init, self.data.S2_init],
            self.data.mu_init, self.data.s2_init, w])
        params = Model2Parameters(
            tau=self.data.tau0,
            Mu=self.data.Mu_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model2_obj(w, params, x, dt, u),
            g=lambda w: model2_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel3Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_init, [self.data.s2_scalar_init], w])
        params = Model3Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_normal_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model3_obj(w, params, x, dt, u),
            g=lambda w: model3_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel4Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_init, w])
        params = Model4Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_normal_params,
            s2=self.data.s2_scalar_init,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model4_obj(w, params, x, dt, u),
            g=lambda w: model4_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel5Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_init, self.data.s2_init, w])
        params = Model5Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_normal_params,
            s2=self.data.s2_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model5_obj(w, params, x, dt, u),
            g=lambda w: model5_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

        o1 = model5_obj(w1, params, x, dt, u)
        o2 = lrxrta5_obj(
            w1,
            (params.tau,
             params.mu.mu,
             params.mu.s2,
             params.s2.alpha,
             params.s2.beta,
             params.l2,
             params.t,
             params.xr.probability,
             params.xr.coefficient),
            acnt, x, dt, u)
        self.assertAllClose(o1, o2)

    def testModel6Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_expmixture_init,
            self.data.mu_init,
            self.data.s2_init, w])
        params = Model6Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_expmixture_params,
            s2=self.data.s2_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model6_obj(w, params, x, dt, u),
            g=lambda w: model6_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel7Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_expmixture_init,
            self.data.mu_init,
            [self.data.s2_scalar_init], w])
        params = Model7Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_expmixture_params,
            s2=self.data.s2_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model7_obj(w, params, x, dt, u),
            g=lambda w: model7_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel8Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_expmixture_init,
            self.data.mu_init, w])
        params = Model8Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_expmixture_params,
            s2=self.data.s2_scalar_init,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model8_obj(w, params, x, dt, u),
            g=lambda w: model8_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel9Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_expmixture_init,
            self.data.mu_init, self.data.lam_init, w])
        params = Model9Parameters(
            tau=self.data.tau0,
            mu=self.data.mu_expmixture_params,
            lam=self.data.lam_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model9_obj(w, params, x, dt, u),
            g=lambda w: model9_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel10Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            [self.data.Mu_init, self.data.S2_init],
            self.data.mu_init, self.data.s2_init, w])
        params = Model10Parameters(
            tau=self.data.tau0,
            Mu=self.data.Mu_params,
            s2=self.data.s2_params,
            l2=self.data.w_l2,
            t=1.0,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model10_obj(w, params, x, dt, u),
            g=lambda w: model10_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

    def testModel11Gradient(self):
        y, x, u, dt, acnt, w = self.basicData

        w1 = np.concatenate([
            self.data.mu_expmixture_init,
            self.data.mu_init, np.sqrt(self.data.s2_init), w])
        params = Model11Parameters(
            tau=self.data.tau0,
            k=1,
            l1=self.data.w_l1,
            l2=self.data.w_l2,
            xr=self.data.xr_params,
            sample_counts=self.data.sample_counts)
        self.checkGrad(
            f=lambda w: model11_obj(w, params, x, dt, u),
            g=lambda w: model11_grad(w, params, x, dt, u),
            x0=w1, rtol=1e-2)

if __name__ == "__main__":
    np.seterr(all='raise', under='ignore')
    doctest.testmod()
    unittest.main()
