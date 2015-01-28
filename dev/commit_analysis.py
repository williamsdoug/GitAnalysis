#
# commit_analysis.py - Code for statistical analysis of commit data
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently configured for OpenStack, tested with Nova.
#
# Last updated 1/27/2014
#
# History:
# - 1/27/15: Initial version of file based on contents of NovaAnalysis notebook
#
#
# Top Level Routines:
#    from commit_analysis import normalize_blame_by_commit
#    from commit_analysis import normalize_blame_by_file
#    from commit_analysis import parse_author
#


# import numpy as np
from collections import defaultdict
import re

# import sys
# from jp_load_dump import jload


def normalize_blame_by_commit(blameset, exp_weighting=True, exp=2.0):
    """returns list of commits with weighted blame based on
       proximity to changes
    """
    result = defaultdict(float)
    total = 0.0
    for per_file in blameset['blame'].values():
        if per_file:       # validate not null entry
            for per_line in per_file:
                found = True
                if exp_weighting:
                    weight = 1.0/(exp**(per_line['proximity']-1))
                else:
                    weight = 1.0/float(per_line['proximity'])
                result[per_line['commit']] += weight
                total += weight
    if total > 0:
        return dict([[k, v/total] for k, v in result.items()])
    else:
        return {}


re_author = re.compile('"([^"]*)"')

anon = {}
anon_ctr = 0


def get_anon_name(s):
    global anon_ctr
    parts = s.split('@')
    if len(parts) == 2:
        if parts[0] not in anon:
            anon[parts[0]] = 'anon_' + str(anon_ctr)
            anon_ctr += 1

        return anon[parts[0]] + '@' + parts[1]
    else:
        return s


def parse_author(s, anonymize=True):
    m = re_author.search(s)
    if m:
        if anonymize:
            return get_anon_name(m.groups(1)[0].encode('ascii', 'ignore'))
        else:
            return m.groups(1)[0].encode('ascii', 'ignore')
    else:
        return '**unknown**'


def normalize_blame_by_file(blameset, exp_weighting=True):
    """returns list of files with weighted blame"""
    result = defaultdict(float)
    total = 0.0
    for fname, per_file in blameset['blame'].items():
        if per_file:       # validate not null entry
            weight = 0.0
            for per_line in per_file:
                if exp_weighting:
                    weight += 1.0/(2.0 ** (per_line['proximity'] - 1))
                else:
                    weight += 1.0/float(per_line['proximity'])
            result[fname] = weight
            total += weight

    return dict([[k, v/total] for k, v in result.items()])
