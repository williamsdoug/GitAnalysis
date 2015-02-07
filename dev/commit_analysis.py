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
# - 2/4/15 - Added compute_guilt(), previously in BlameAnalysis Spreadsheet
# - 2/6/15 - Added top level routines load_all_analysis_data(),
#            load_core_analysis_data() and rebuild_all_analysis_data()
# - 2/6/15 - Modified compute_guilt() to use filter_bug_fix_combined_commits()
#            when selecting blame entries for guilt calculation.
# - 2/7/15 - moved functions from notebook: trim_entries(), parse_author(),
#            create_feature(), extract_features()
#
# Top Level Routines:
#    from commit_analysis import blame_compute_normalized_guilt
#    from commit_analysis import normalize_blame_by_file
#    from commit_analysis import parse_author
#    from commit_analysis import get_commit_count_by_author
#    from commit_analysis import get_blame_by_commit
#    from commit_analysis import compute_guilt
#    from commit_analysis import extract_features
#
#    from commit_analysis import load_core_analysis_data
#    from commit_analysis import load_all_analysis_data
#    from commit_analysis import rebuild_all_analysis_data
#


# import numpy as np
from collections import defaultdict
import re
import math
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer

from LPBugsDownload import build_lp_bugs, load_lp_bugs

from GerritDownload import build_gerrit_data
from GerritDownload import load_gerrit_changes, load_gerrit_change_details

from Git_Extract_Join import build_git_commits, load_git_commits
from Git_Extract_Join import build_joined_LP_Gerrit_git, load_combined_commits
from Git_Extract_Join import build_all_blame, load_all_blame


from Git_Extract_Join import filter_bug_fix_combined_commits

# import sys
# from jp_load_dump import jload


#
# Top level routines to load and update analysis data
#


def load_core_analysis_data(project):
    """ Loads combined_commits and all_blame."""

    combined_commits = load_combined_commits(project)
    print 'combined_commits:', len(combined_commits)

    all_blame = load_all_blame(project)
    print 'all blame:', len(all_blame)

    return combined_commits, all_blame


def load_all_analysis_data(project):
    """ Loads downloaded_bugs, all_changes, all_change_details,
        commits, combined_commits and all_blame.
    """
    print 'loading bug data'
    downloaded_bugs = load_lp_bugs(project)
    print

    print 'loading Git commit data'
    commits = load_git_commits(project)

    print 'loading change data'
    all_change_details = load_gerrit_change_details(project)
    print 'all_change_details:', len(all_change_details)

    all_changes = load_gerrit_changes(project)
    print 'all_changes:', len(all_changes)

    combined_commits = load_combined_commits(project)
    print 'combined_commits:', len(combined_commits)

    all_blame = load_all_blame(project)
    print 'all blame:', len(all_blame)

    return downloaded_bugs, all_changes, all_change_details, \
        commits, combined_commits, all_blame


def rebuild_all_analysis_data(project, repo_name, update=True):
    """Rebuilds core datasets"""
    cachedir = './cache/' + project + '/'

    build_lp_bugs(project, update=update, cachedir=cachedir)

    print
    print 'rebuilding Gerrit data'
    build_gerrit_data(project, update=update)

    print
    print'building Git data'
    build_git_commits(project, repo_name, update=update)

    print 'Preparation for join'
    downloaded_bugs = load_lp_bugs(project)
    commits = load_git_commits(project)
    all_change_details = load_gerrit_change_details(project)

    print
    print 'Building combined_commits'
    build_joined_LP_Gerrit_git(project, commits, downloaded_bugs,
                               all_change_details)

    print
    print 'Building all blame'
    combined_commits = load_combined_commits(project)
    build_all_blame(project, combined_commits, repo_name, update=update)

#
# Routines for post-processing Dataset
#


def trim_entries(combined_commits, all_blame):
    """Returns order range (min, max) based on first and list
    bug fix commit. First entry is after first bug fix.
    """
    buglist = [(combined_commits[be['cid']]['order'], be['cid'])
               for be in all_blame]
    buglist = sorted(buglist, key=lambda x: x[0])
    return buglist[0][0] + 1, buglist[-1][0]


RE_AUTH = re.compile('<(\S+@\S+)>')
RE_AUTH2 = re.compile('"(\S+@\S+)\s')
RE_AUTH3 = re.compile('"(\S+)\s')


def parse_author_and_org(auth):
    result = RE_AUTH.search(auth)
    if not result:  # Try alternative pattern
        result = RE_AUTH2.search(auth)
    if not result:  # Try alternative pattern
        result = RE_AUTH3.search(auth)
    if not result:
        if 'docs.openstack.org' in str(auth):
            return 'openstack-tool@openstack.org', 'openstack.org'
        else:
            raise Exception('Unable to parse author: ' + str(auth))

    author_name = result.groups()[0]
    author_org = author_name.split('@')[-1]
    return author_name, author_org


def create_feature(c):
    """Extract features from combined_commits entry"""
    label = c['guilt']
    cid = c['cid']

    feats = {}
    # feats['order'] = math.log(c['order'])

    # General information about author
    author_name, author_org = parse_author_and_org(c['author'])
    feats['author'] = author_name
    feats['author_org'] = author_org
    feats['author_order'] = math.log(c['author_order'])

    # if this commit associated with a bug fix itself
    feats['is_bug_fix'] = 'lp:id' in c

    # General information around change (size (loc), code maturity,
    # prior bugs in module)
    for fname in c['files']:
        feats[fname] = 1
    if c['file_order']:
        feats['min_file_order'] = math.log(min([v for v
                                                in c['file_order'].values()]))
        feats['max_file_order'] = math.log(max([v for v
                                                in c['file_order'].values()]))

    # Information about committer, approver and reviewers
    committer_name, _ = parse_author_and_org(c['committer'])
    feats['committer'] = committer_name

    # Information about code changes
    feats['loc_add'] = c['loc_add']
    feats['loc_change'] = c['loc_change']
    for fname, detail in c['loc_detail']. items():
        feats['loc_add_' + fname] = detail['add']
        feats['loc_changes_' + fname] = detail['changes']

    """
    # Features below commented out since no impact on recall and F1
    if False: #'g:labels' in c:
        feats['approved'] = c['g:labels']['Code-Review']['approved']['name']
        for r in c['g:labels']['Code-Review']['all']:
            feats['reviewer'+ r['name']] = r['value']

        feats['votes'] = sum([r['value'] for r
                              in c['g:labels']['Code-Review']['all']])

    if False: # 'g:messages' in c and c['g:messages']:
        feats['revision'] = max([msg['_revision_number']
                                 for msg in c['g:messages']])
    """

    if 'lp:message_count' in c:
        feats['lp_messages'] = c['lp:message_count']

    return cid, label, feats


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


def extract_features(combined_commits, all_blame, threshold=False,
                     clip=False, min_order=False, max_order=False,
                     offset=0, limit=0, equalize=False,
                     debug=True):
    """Extracts features from combined_commits
    Parameters:
    - threshold -- Used for classification, determines 1 /0 labels. False
      for regression (default)
    - clip -- Limits max value of label for regression problems.
    - min_order, max_order -- range of included commits.  full range
      by default
    - offset -- relative start, either as integer or percent
    - limit -- overall entries, either integer or percent

    Returns:
    - Labels
    - Feature Matrix
    - Feature matrix column names
    """

    if not min_order and not max_order:
        min_order, max_order = trim_entries(combined_commits, all_blame)
    elif not min_order:
        min_order, _ = trim_entries(combined_commits, all_blame)
    elif not max_order:
        _, max_order = trim_entries(combined_commits, all_blame)

    order_range = max_order - min_order

    if offset == 0:
        pass
    elif type(offset) is int:
        min_order += offset
    elif type(offset) is float and offset < 1.0:
        min_order += int(order_range*offset)
    else:
        raise Exception('extract_features: Invalid offset value '
                        + str(offset))

    if limit == 0:
        pass
    elif type(limit) is int:
        max_order = min_order + limit - 1
    elif type(limit) is float and offset < 1.0:
        max_order = min_order + int(order_range*limit)
    else:
        raise Exception('extract_features: Invalid limit value ' + str(limit))

    cid, Y, features = zip(*[create_feature(x)
                             for x in combined_commits.values()
                             if (x['order'] >= min_order
                                 and x['order'] <= max_order)])

    vec = DictVectorizer()
    X = vec.fit_transform([f for f in features]).toarray()

    Y = np.asarray(Y)
    if clip:
        Y = np.minimum(Y, float(clip))

    scaler = MinMaxScaler()     # Use MinMaxScaler non-gaussian data
    X = scaler.fit_transform(X)

    if debug:
        print 'total features:', len(features)
    if threshold:   # Quantize guilt
        Y = np.asarray(Y) > threshold
        if debug:
            print 'bugs based on threshold:', sum(Y)
            print 'actual bugs:', sum([1 for f in features if f['is_bug_fix']])

    return cid, Y, X, vec.get_feature_names()


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


def compute_guilt(combined_commits, all_blame, importance='high+'):
    for c in combined_commits.values():  # initialize guilt values
        c['guilt'] = 0.0

    for be in all_blame:    # now apply weighted guilt for each blame
        v = combined_commits[be['cid']]
        if filter_bug_fix_combined_commits(v, importance=importance):
            for c, g in \
                    blame_compute_normalized_guilt(be,
                                                   exp_weighting=True).items():
                combined_commits[c]['guilt'] += g

    total = len(combined_commits)
    guilty = sum([1 for v in combined_commits.values() if v['guilt'] > 0])
    min_guilt = min([v['guilt']
                     for v in combined_commits.values() if v['guilt'] > 0])
    max_guilt = max([v['guilt']
                     for v in combined_commits.values() if v['guilt'] > 0])

    print 'guilty: ', guilty, 'out of',  total,
    print '(',  100.0 * float(guilty) / float(total), '%', ')'
    print 'smallest guilt:', min_guilt
    print 'largest guilt:', max_guilt
