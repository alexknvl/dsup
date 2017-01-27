#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import unittest
import scipy.optimize
import numpy as np

DEFAULT_RTOL = 1.0e-7
DEFAULT_ATOL = 0.0
DEFAULT_EQUAL_NAN = False
DEFAULT_EPSILON = np.sqrt(np.finfo(float).eps)

class NumpyTest(unittest.TestCase):
    def setUp(self):
        from numpy.random import RandomState
        self.rng = RandomState(534123122)
        self.numpy_err = np.seterr(all='raise', under='warn')

    def tearDown(self):
        np.seterr(**self.numpy_err)

    def checkGrad(self, f, g, x0,
                  rtol=DEFAULT_RTOL,
                  atol=DEFAULT_ATOL,
                  epsilon=DEFAULT_EPSILON):
        ng = scipy.optimize.approx_fprime(x0, f, epsilon)
        ag = g(x0)
        self.assertAllClose(ng, ag, rtol=rtol, atol=atol)

    def assertAlmostEqual(self, actual, desired,
                          rtol=DEFAULT_RTOL, atol=DEFAULT_ATOL):
        self.assertAllClose(actual, desired, rtol=rtol, atol=atol)

    def assertAllClose(self, actual, desired,
                       rtol=DEFAULT_RTOL,
                       atol=DEFAULT_ATOL,
                       equal_nan=DEFAULT_EQUAL_NAN):
        np.testing.assert_allclose(actual, desired,
                                   rtol=rtol, atol=atol,
                                   equal_nan=equal_nan)

    def assertAlmostZero(self, actual, atol=DEFAULT_RTOL):
        self.assertAllClose(actual, 0, rtol=0, atol=atol)
