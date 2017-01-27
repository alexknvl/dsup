#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import readline
from collections import namedtuple

TurkLine = namedtuple(
    'TurkLine',
    'annoScore attr mode date score editScore id arg1 arg2 text')


def parse_line(line):
    attr = line[0]
    mode = line[1]
    date = line[2]
    score = float(line[3])
    editScore = float(line[4])
    id = line[5]
    arg1, arg2 = line[6], line[7]
    text = line[8]
    annoScore = line[9] if len(line) >= 10 else None

    return TurkLine(
        annoScore,
        attr, mode, date, score, editScore,
        id, arg1, arg2, text)


def read_anno_score():
    while True:
        r = raw_input('Score: ').strip().lower()
        if r == 'y':
            return '+'
        elif r == 'n':
            return '-'
        elif r == 'm':
            return '?'
        elif r == 'a':
            return '?a'
        elif r == 's':
            return '-s'
        else:
            continue


if __name__ == "__main__":
    dname = sys.argv[1]
    tname = sys.argv[2]

    cache = { }

    for fname in sorted(os.listdir(dname)):
        if fname.startswith('BandMember') or fname.startswith('Television') or fname.startswith('TourBy'):
            continue
        #print(fname)
        lines = []

        if os.path.exists(os.path.join(tname, fname)):
            with open(os.path.join(dname, fname)) as f:
                lines = (x.rstrip().split('\t') for x in f)
                lines = [parse_line(x) for x in lines]
                for line in lines:
                    req = (line.attr, line.arg1, line.arg2, line.text)
                    cache[req] = line.annoScore
        else:
            with open(os.path.join(dname, fname)) as f:
                lines = (x.rstrip().split('\t') for x in f)
                lines = [parse_line(x) for x in lines]

                for index, line in enumerate(lines):
                    req = (line.attr, line.arg1, line.arg2, line.text)
                    if req in cache:
                        annoScore = cache[req]
                    else:
                        print('')
                        print('%d / %d : (%s, %s, %s)' %
                              (index + 1, len(lines),
                               line.attr, line.arg1, line.arg2))
                        print(line.text)
                        annoScore = read_anno_score()
                    cache[req] = annoScore
                    lines[index] = lines[index]._replace(annoScore=annoScore)

            with open(os.path.join(tname, fname), 'w+') as f:
                for line in lines:
                    f.write('\t'.join(map(str, line)) + '\n')
