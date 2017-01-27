import os
import sys
import numpy as np
import fileinput
from itertools import imap
from collections import namedtuple
from tabulate import tabulate

GSLine = namedtuple('GSLine',
                    ['attr', 'mode', 'lr', 'fwd',
                     'nsr', 'p_ex', 'f1'])


def read_gs_results(lines):
    lines = (line.strip().split('\t') for line in lines)
    for line in lines:
        attr, mode, lr = line[:3]
        fwd, nsr = [x for x in line[3:5]]
        pex, f1 = [float(x) for x in line[5:]]
        yield GSLine(attr, mode, lr, fwd, nsr, pex, f1)


def groupby(iterable, key):
    result = {}
    for v in iterable:
        result.setdefault(key(v), []).append(v)
    return result


def itemids(lst):
    return dict(imap(reversed, enumerate(lst)))


def maxby(iterable, key):
    result = None
    result_key = None
    for v in iterable:
        k = key(v)
        if result_key is None or result_key < k:
            result_key = k
            result = v
    return result


TableEntry = namedtuple('TableEntry',
                        'attr normal_lrxr_f1 normal_lr_f1 baseline_lr_f1')


def main():
    data = list(read_gs_results(fileinput.input()))
    attrs = set([x.attr for x in data])

    table = []
    for attr in attrs:
        top_normal_lrxr = maxby((x for x in data
                                 if (x.mode, x.lr) == ("normal", "lrxr")
                                 and x.attr == attr),
                                key=lambda x: x.f1)
        top_normal_lr = maxby((x for x in data
                               if (x.mode, x.lr) == ("normal", "lr")
                               and x.attr == attr),
                              key=lambda x: x.f1)
        top_baseline_lr = maxby((x for x in data
                                 if (x.mode, x.lr) == ("baseline", "lr")
                                 and x.attr == attr),
                                key=lambda x: x.f1)

        table.append(TableEntry(attr, top_normal_lrxr.f1, top_normal_lr.f1,
                                top_baseline_lr.f1))

    table.sort(key=lambda t: -t.normal_lrxr_f1)
    # print(tabulate(sorted(table, key=lambda t: -t.normal_lrxr_f1), headers=[
    #     "attr_name", "max_f1",
    #     "max_lr_f1", "max_baseline_f1"]))

    print("""
    \\begin{tabular}{|c|l|l|l|} \hline
    {\\bf Attribute}     & {\\bf LRXR F1} & {\\bf LR F1} & {\\bf Baseline F1}
    \\\\ \\hline
    """)

    for e in table:
        line = ""
        attr_name = ''.join(s[0].upper() + s[1:] for s in e.attr.split('_'))
        line += "{\\sc %s}" % attr_name

        values = [e.normal_lrxr_f1, e.normal_lr_f1, e.baseline_lr_f1]
        max_value = max(float("%2.2f" % v) for v in values)
        for v in values:
            v = float("%2.2f" % v)
            if v == max_value:
                line += "& {\\bf %2.2f\\%% }" % v
            else:
                line += "& %2.2f\\%%" % v
        line += "\\\\ \\hline"

        print(line)

    print("""\end{tabular}""")


if __name__ == '__main__':
    main()
