#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division

import re
import os
import errno
import sys
import ujson as json

import result_store


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

ID_PATTERN = re.compile(r'id":"(\d+)"')


def fast_read_predictions(path, dataset):
    with open(os.path.join(path, dataset + '_predictions.json')) as f:
        lines = (line.rstrip() for line in f)
        for line in lines:
            match = ID_PATTERN.search(line)
            if match is not None:
                yield (match.groups()[0], line)


if __name__ == "__main__":
    path = '/home/konovalo/dsup_event/new-data/annotated_ids.csv'
    sids = set(line.strip() for line in open(path) if line.strip() != '')

    dname = sys.argv[1]
    tname = sys.argv[2]

    for (path, (attr, hash)) in result_store.v3_get_all_output_dirs(dname):
        mkdir_p(os.path.join(tname, attr + "_" + hash))

        for dataset in ['test', 'test2']:
            opath = os.path.join(tname, attr + "_" + hash, dataset + '.json')
            if os.path.exists(opath):
                continue

            try:
                predictions = fast_read_predictions(path, dataset)
            except:
                continue

            try:
                with open(opath, 'w+') as f:
                    for (id, line) in predictions:
                        if id in sids:
                            f.write(line + '\n')
            except:
                os.remove(opath)
        print path
