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
# - 9/2/14:  Initial version (initially contained in NovaSampleData).
#            Updated normalize_blame to handle null entries
# - 9/2/14:  Update nova_all_blame, filtering out all entries > 3000 diff
#            lines that were contributing to huge file size.  Corrected
#            blame_compute_normalized_guilt computation, added
#            normalize_blame_by_file
# - 1/27/15: Initial version of commit_analysis.py based on contents
#            of NovaAnalysis notebook
#
#
# Top Level Routines:
#    from commit_analysis import blame_compute_normalized_guilt
#    from commit_analysis import normalize_blame_by_file
#    from commit_analysis import parse_author
#    from commit_analysis import get_commit_count_by_author
#    from commit_analysis import get_blame_by_commit
#


# import numpy as np
from collections import defaultdict
import re

# import sys
# from jp_load_dump import jload

# should we clip based on max distance???
def blame_compute_normalized_guilt(blameset, exp_weighting=True, exp=2.0):
    """Apportions guilt for each blame entry to individual commits
       based on proximity to changed lines and number of occurances,
       where total guilt for each blame entry is 1.  Guild is weighted
       based on proximity, where weight is either based on inverse linear
       distance or exponentially diminishing (default).

       exp_weighting:  Determines whether proximity-vased weighting
                       is either linear or exponential.
       exp: Specifies power functin if exponential weighting
    """
    result = defaultdict(float)
    total = 0.0
    for per_file in blameset['blame'].values():
        if per_file:       # validate not null entry
            for per_line in per_file:
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
    global anon
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


def get_commit_count_by_author(combined_commits):
    commits_by_author = defaultdict(float)
    for x in combined_commits.values():
        author = parse_author(x['author'])
        commits_by_author[author] += 1.0

    return commits_by_author


def get_blame_by_commit(combined_commits, all_blame):
    blame_by_commit = defaultdict(float)
    for x in all_blame:
        for commit, weight in \
            blame_compute_normalized_guilt(x, exp_weighting=True,
                                           exp=4.0).items():
            author = parse_author(combined_commits[commit]['author'])
            blame_by_commit[author] += weight

    return blame_by_commit
