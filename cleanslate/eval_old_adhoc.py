#!/usr/bin/python2
# -*- coding: utf-8 -*-

from __future__ import division

import sys
import os

import numpy as np
import scipy.sparse

import argparse
import ujson as json

import fx
import easytime
import lrxr

from common import Text, Sample, Edit, gen_tprf, mkdir_p, sha1_structure, normalize_attr
from vocab import Vocabulary

from collections import namedtuple
from recordtype import recordtype
from tabulate import tabulate

Result = namedtuple('Result', ['p', 'y', 'labeled'])

def main(path, name):
    with open(path) as f:
        lines = (json.loads(line) for line in f)

        def to_result(j):
            if isinstance(j, list):
                return Result(p=j[1], y=j[0], labeled=len(j[2]['edits']) > 0)
            else:
                return Result(p=j['pred'], y=j['y'],
                              labeled=len(j['wpEdits']) > 0)

        lines = (to_result(j) for j in lines)

        results = list(lines)
        labeled_results = [r for r in results if r.labeled]
        unlabeled_results = [r for r in results if not r.labeled]

        print "Labeled:   %s" % (len(labeled_results),)
        print "Unlabeled: %s" % (len(unlabeled_results),)

        unlabeled_subset_len = min(len(labeled_results) * 9,
                                   len(unlabeled_results))
        unlabeled_results = fx.reservoir_sample(unlabeled_results,
                                                unlabeled_subset_len)
        results = sorted(labeled_results + unlabeled_results,
                         key=lambda r: -r.p)

        #print results[:10]

        tprfs = list(gen_tprf(results))
        #print tprfs[:30]
        tprf = fx.max_by(tprfs, key=lambda x: x.f1)
        print '%s %s & %s & %s' % (name, tprf.precision, tprf.recall, tprf.f1)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[1])
