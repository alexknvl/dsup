#!/usr/bin/python2
# -*- coding: utf-8 -*-
from __future__ import division

import os
import sys
import ujson as json
import re
import numpy as np

import result_store


def main(dir):
    SYNONYMS = {
        'currentclub,currentteam,team': 'CurrentTeam',
        'death_place': 'DeathPlace',
        'spouse': 'Spouse'
    }

    dirs = [(path, attr, hash)
            for path, (attr, mode1, mode2, hash) in
            result_store.v2_get_all_output_dirs(dir)]
    dirs += [(path, attr, hash)
             for path, (attr, hash) in
             result_store.v3_get_all_output_dirs(dir)]

    for (path, attr, hash) in dirs:
        #print path
        params = result_store.v2_read_params(path)
        scores = result_store.v2_read_scores(path)
        offsets = result_store.v2_read_offsets(path)
        if offsets is not None and 'all' in offsets:
            offsets = offsets['all']
        else:
            offsets = {}

        attr = params['attr']
        attr = SYNONYMS.get(attr, attr)
        mode = params['mode']
        #print mode

        line = []

        if mode == 'normal_lrxr' or mode == 'ad-hoc' or mode == 'baseline':
            # Attribute Model   Optimization
            line.append(attr)
            line.append(mode)
            line.append('L-BFGS')

            # Bias λ₂ Feature λ₂
            line.append(params['bias_l2'])
            line.append(params['feature_l2'])
            # pₑₓ xr
            line.append(params['p_ex'])
            line.append(params['xr'])
            # M   Σ
            line.append('')
            line.append('')
            # α   β   μ   σ
            line.append('')
            line.append('')
            line.append('')
            line.append('')
            # Δσ²
            line.append('')

            # F₁ (train)  F₁ (test)   F₁ (mturk)
            line.append('%.3f' % scores['train']['f1'])
            line.append('%.3f' % scores['test']['f1'])
            line.append('%.3f' % scores['mturk']['f1'])

            # Edit Δt Acc Edit Δt Prec
            if 'edit' in offsets:
                line.append('%.1f' % offsets['edit'][0])
                line.append('%.1f' % np.sqrt(offsets['edit'][1]))
            else:
                line.append('')
                line.append('')

            # Mode Δt Acc Mode Δt Prec
            if 'mode' in offsets:
                line.append('%.1f' % offsets['mode'][0])
                line.append('%.1f' % np.sqrt(offsets['mode'][1]))
            else:
                line.append('')
                line.append('')

            # Mean Δt Acc Mean Δt Prec
            if 'mean' in offsets:
                line.append('%.1f' % offsets['mean'][0])
                line.append('%.1f' % np.sqrt(offsets['mean'][1]))
            else:
                line.append('')
                line.append('')

            # Folder name
            line.append(path)
        elif mode == 'normal_lrxrta5' or mode == 'automatic':
            # Attribute Model   Optimization
            line.append(params['attr'])
            line.append(mode)
            line.append('AdaDelta')

            # Bias λ₂ Feature λ₂
            line.append(params['bias_l2'])
            line.append(params['feature_l2'])
            # pₑₓ xr
            line.append(params['p_ex'])
            line.append(params['xr'])
            # M   Σ
            line.append(params['Mu'])
            line.append(params['Sigma'])
            # α   β   μ   σ
            line.append(params['alpha'])
            line.append(params['beta'])
            line.append(params['mu'])
            line.append(params['sigma'])
            # Δσ²
            line.append(params['s2_shift'])

            # F₁ (train)  F₁ (test)   F₁ (mturk)
            line.append('%.3f' % scores['train']['f1'])
            line.append('%.3f' % scores['test']['f1'])
            line.append('%.3f' % scores['mturk']['f1'])

            # Edit Δt Acc Edit Δt Prec
            if 'edit' in offsets:
                line.append('%.1f' % offsets['edit'][0])
                line.append('%.1f' % np.sqrt(offsets['edit'][1]))
            else:
                line.append('')
                line.append('')

            # Mode Δt Acc Mode Δt Prec
            if 'mode' in offsets:
                line.append('%.1f' % offsets['mode'][0])
                line.append('%.1f' % np.sqrt(offsets['mode'][1]))
            else:
                line.append('')
                line.append('')

            # Mean Δt Acc Mean Δt Prec
            if 'mean' in offsets:
                line.append('%.1f' % offsets['mean'][0])
                line.append('%.1f' % np.sqrt(offsets['mean'][1]))
            else:
                line.append('')
                line.append('')

            # Folder name
            line.append(path)
        else:
            assert False

        sys.stdout.write('\t'.join(map(str, line)) + '\n')

    dirs = result_store.v1_get_all_output_dirs(dir)
    for path, (attr, mode1, mode2, hash) in dirs:
        params = result_store.v1_read_params(path, mode2)
        scores = result_store.v1_read_scores(path)

        if mode2 == "lrxrta5":
            (bias_l2, feature_l2, p_ex, xrv,
             MU_0, S2_0, alpha, beta, s2_shift) = params
            Sigma = np.sqrt(S2_0)
            #print alpha, beta
            if alpha > 1:
                mu = beta / (alpha - 1) * 993.0**2
            else:
                mu = ""

            if alpha > 2:
                sigma = np.sqrt(beta**2 / (alpha - 1)**2 / (alpha - 2)) * 993.0**2
            else:
                sigma = ""
            #print mu, s2
            train, test, mturk = scores
            train_f1 = '%.3f' % train[-1]
            test_f1 = '%.3f' % test[-1]
            mturk_f1 = '%.3f' % mturk[-1]

            print '\t'.join(map(str, [attr, mode2, 'L-BFGS',
                                      bias_l2, feature_l2,
                                      p_ex, xrv, MU_0, Sigma, alpha, beta,
                                      mu, sigma, s2_shift,
                                      train_f1, test_f1, mturk_f1,
                                      '', '', '', '', '', '',
                                      path]))
        elif mode2 == "lr":
            bias_l2, feature_l2 = params
            p_ex, xrv, MU_0, Sigma, alpha, beta, mu, sigma, s2_shift = \
                "", "", "", "", "", "", "", "", ""
            train, test, mturk = scores
            train_f1 = '%.3f' % train[-1]
            test_f1 = '%.3f' % test[-1]
            mturk_f1 = '%.3f' % mturk[-1]
            print '\t'.join(map(str, [attr, mode2, 'L-BFGS',
                                      bias_l2, feature_l2,
                                      p_ex, xrv, MU_0, Sigma, alpha, beta,
                                      mu, sigma, s2_shift,
                                      train_f1, test_f1, mturk_f1,
                                      '', '', '', '', '', '',
                                      path]))
        else:
            assert False


if __name__ == "__main__":
    main(sys.argv[1])
