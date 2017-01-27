#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import os
import re
import ujson as json

import fx

import common
from common import sample_from_json

from collections import namedtuple

Text = common.Text
Sample = common.Sample
Edit = common.Edit
TPRF = common.TPRF
Prediction = namedtuple('Prediction', ['y', 'p', 's'])

TAPair = namedtuple('TAPair', 'mu0 sigma0 mu sigma')
TAParams = namedtuple('TAParams', ['tau0', 'Mu', 'Sigma', 'pairs'])
WeightsFile = namedtuple('WeightsFile', ['ta_params', 'weights'])


def prediction_from_json(j):
    (y, p, sj) = j
    return Prediction(y, p, sample_from_json(sj))


def v1_get_all_output_dirs(path):
    pattern = re.compile(
        r'^([a-z0-9,_]+)_(normal|baseline)_([a-z0-9]+)_([\-0-9]+)$')
    for subdir in os.listdir(path):
        match = pattern.match(subdir)
        if match is not None and os.path.isdir(os.path.join(path, subdir)):
            yield (os.path.join(path, subdir), match.groups())


def v1_read_params(path, mode2):
    with open(os.path.join(path, 'paramOut')) as f:
        content = f.read()
        params = json.loads(content)
        return tuple(params)


def v1_read_scores(path):
    num = r"[\d\.]+(?:e[+-]\d+)?"
    line_pattern = re.compile(
        r"TPRF\(threshold=(" + num + r"), precision=(" + num + r"), " +
        r"recall=(" + num + r"), f1=(" + num + r")\)")

    def parse_tprf(line):
        match = line_pattern.match(line.strip())
        if match is not None:
            return map(float, match.groups())
        assert False

    with open(os.path.join(path, 'maxFout')) as f:
        content = map(parse_tprf, f.read().strip().split('\n'))
        return content


def v2_get_all_output_dirs(path):
    pattern = re.compile(
        r'^([a-z0-9,_]+)_(normal|baseline)_([a-z0-9]+)_([a-f0-9]{40})$')
    for subdir in os.listdir(path):
        match = pattern.match(subdir)
        if match is not None and os.path.isdir(os.path.join(path, subdir)):
            yield (os.path.join(path, subdir), match.groups())


def v2_read_params(path):
    with open(os.path.join(path, 'params.json')) as f:
        return json.loads(f.read())


def v2_read_scores(path):
    with open(os.path.join(path, 'scores.json')) as f:
        return json.loads(f.read())


def v2_read_predictions(path, dataset):
    with open(os.path.join(path, dataset + '_predictions.json')) as f:
        for line in f:
            yield prediction_from_json(json.loads(line))


def v2_read_weights(path):
    with_ta_params = False
    tau0 = None
    Mu, Sigma = None, None
    pairs = {}
    weights = {}

    def parse_mu_sigma(s):
        i = s.find('+-')
        return float(s[:i]), float(s[i+2:])

    with open(os.path.join(path, 'weights.csv')) as f:
        lines = (line.rstrip().split('\t') for line in f)

        # first, lines = fx.peek(lines, 1)
        # first = first[0]

        for args in lines:
            k = args[0]
            v = args[1:]
            if k == 'tau0':
                with_ta_params = True
                tau0 = float(v[0])
            elif k == 'Mu':
                with_ta_params = True
                Mu = float(v[0])
            elif k == 'Sigma':
                with_ta_params = True
                Sigma = float(v[0])
            elif k == 'mu_sigma':
                with_ta_params = True
                arg1 = v[0]
                arg2 = v[1]
                mu0, sigma0 = parse_mu_sigma(v[2])
                mu, sigma = parse_mu_sigma(v[3])
                pairs[(arg1, arg2)] = TAPair(mu0, sigma0, mu, sigma)
            elif k == '__MU0__' or k == '__S0__' or k == '__S2_0__':
                assert False
            else:
                pattern = v[0]
                weights[pattern] = float(v[1])

        ta_params = None
        if with_ta_params:
            ta_params = TAParams(tau0, Mu, Sigma, pairs)
        return WeightsFile(ta_params, weights)


def v2_read_offsets(path):
    try:
        f = open(os.path.join(path, 'offsets.json'))
        j = json.loads(f.read())
        f.close()
        return j
    except:
        return None


def v3_get_all_output_dirs(path):
    pattern = re.compile(r'^([a-zA-Z]+)_([a-f0-9]{40})$')
    for subdir in os.listdir(path):
        match = pattern.match(subdir)
        if match is not None and os.path.isdir(os.path.join(path, subdir)):
            yield (os.path.join(path, subdir), match.groups())
