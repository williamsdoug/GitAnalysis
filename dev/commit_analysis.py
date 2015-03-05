#
# commit_analysis.py - Code for statistical analysis of commit data
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently configured for OpenStack, tested with Nova.
#
# Last updated 3/5/2015
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
# - 2/9/15 - added autoset_threshold() and helper function
#            count_guilty_commits().  Added fit_features()
# - 2/19/15 - New consistency checking routine verify_missing_bugs().
# - 2/20/15 - updated rebuild_all_analysis_data() to remove repo_name
#             including api calls to build_git_commits(), build_all_blame().
#             also remove cachedir from calls to build_lp_bugs()
# - 2/23/15 - New join code - join_all()
# - 2/24/15 - Integrated join into rebuild_all_analysis_data
# - 2/25/15 - Updated to reflect single change_id per commit.  Added error
#             handling when loading all_blame
# - 2/26/15 - Clean-up handling of change_id
# - 3/3/15  - Added additional options to rebuild_all_analysis_data()
# - 3/5/15  - Updated feature extraction to reflect new schema

#
# Top Level Routines:
#    from commit_analysis import blame_compute_normalized_guilt
#    from commit_analysis import normalize_blame_by_file
#    from commit_analysis import parse_author
#    from commit_analysis import get_commit_count_by_author
#    from commit_analysis import get_blame_by_commit
#    from commit_analysis import compute_guilt
#    from commit_analysis import fit_features
#    from commit_analysis import extract_features
#    from commit_analysis import autoset_threshold
#
#    from commit_analysis import load_core_analysis_data
#    from commit_analysis import load_all_analysis_data
#    from commit_analysis import rebuild_all_analysis_data
#    from commit_analysis import verify_missing_bugs
#

# import numpy as np
import collections
import re
import math
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction import DictVectorizer

from LPBugsDownload import build_lp_bugs, load_lp_bugs

from GerritDownload import build_gerrit_data
from GerritDownload import load_gerrit_changes, load_gerrit_change_details

from Git_Extract import build_git_commits, load_git_commits
# from Git_Extract import build_all_blame, load_all_blame
from Git_Extract import filter_bug_fix_combined_commits
from Git_Extract import project_to_fname
from jp_load_dump import convert_to_builtin_type, jload, jdump


# import sys
# from jp_load_dump import jload

#
# Global Data (constants)
#
global BUG_SEVERITY_VALUES
global BUG_PRECEDENCE_VALUES

BUG_SEVERITY_VALUES = {
    'Critical': 4,
    'High': 3,
    'Medium': 2,
    'Low': 1,
    'Wishlist': 0,
    'Unknown': 0,
    'Undecided': 0,
}

BUG_PRECEDENCE_VALUES = {
    'Critical': 6,
    'High': 5,
    'Medium': 4,
    'Low': 3,
    'Wishlist': 2,
    'Unknown': 1,
    'Undecided': 0,
}

#
# Routines to consistency check Git, Gerrit and Lanuchpad Data
#


def verify_missing_bugs(project):
    """Make sure all bugs references in commits have been downloaded,
    if not, attempt to load these bugs"""

    all_bugs = load_lp_bugs(project)
    commits = load_git_commits(project)

    all_bugs_in_commits = set([b for c in commits.values()
                               if 'bugs' in c for b in c['bugs']])
    known_bugs = set(all_bugs.keys())

    missing_bugs = all_bugs_in_commits.difference(known_bugs)
    if len(missing_bugs) > 0:
        build_lp_bugs(project, update=missing_bugs)
    else:
        print 'no missing bugs'

#
# Top level routines to load and update analysis data
#


def load_core_analysis_data(project):
    """ Loads combined_commits and all_blame."""

    combined_commits = load_combined_commits(project)
    print 'combined_commits:', len(combined_commits)

    # all_blame = load_all_blame(project)
    # print 'all blame:', len(all_blame)

    # return combined_commits, all_blame
    return combined_commits


def load_all_analysis_data(project):
    """ Loads downloaded_bugs, all_changes, all_change_details,
        commits and combined_commits.
    """
    print 'loading bug data'
    downloaded_bugs = load_lp_bugs(project)

    print 'loading Git commit data'
    commits = load_git_commits(project)

    print 'loading change data'
    all_change_details = load_gerrit_change_details(project)
    print '  all_change_details:', len(all_change_details)

    all_changes = load_gerrit_changes(project)
    print '  all_changes:', len(all_changes)

    combined_commits = load_combined_commits(project)
    print 'combined_commits:', len(combined_commits)

    return downloaded_bugs, all_changes, all_change_details, \
        commits, combined_commits


def rebuild_all_analysis_data(project, update=True,
                              download=True,
                              build_blame=False,
                              build_launchpad=True,
                              build_gerrit=True,
                              build_git=True,
                              build_combined=True):
    """Rebuilds core datasets"""

    if download and build_launchpad:
        print
        print 'rebuilding Launchpad (bug) data'
        build_lp_bugs(project, update=update)

    if download and build_gerrit:
        print
        print 'rebuilding Gerrit data'
        build_gerrit_data(project, update=update)

    if build_git:
        print
        print 'building Git data'
        build_git_commits(project, update=update)

    if download and build_launchpad:
        print
        print 'Load any missing bugs, if needed'
        verify_missing_bugs(project)

    if build_combined:
        print
        print 'Build combined_commits by joining with bugs and gerrit data'
        combined_commits = join_all(project)

    #  Needs to be updated
    """
    if build_blame:
        print
        print 'Building all blame'
        combined_commits = load_combined_commits(project)
        build_all_blame(project, combined_commits, update=update)
    """

#
# Join related routines
#


def load_combined_commits(project):
    """Loads combined_commit data from disk"""
    return jload(project_to_fname(project, combined=True))


def join_all(project):
    """Top level join routine"""
    all_bugs = load_lp_bugs(project)
    commits = load_git_commits(project)
    all_changes = load_gerrit_changes(project)
    all_change_details = load_gerrit_change_details(project)

    # clone commits
    combined = dict([[k, v.copy()] for k, v in commits.items()])

    # include Launchpad bug details information
    print 'Joining with Launchpad Data'
    combined = join_with_bugs(combined, all_bugs)

    # include Gerrit review details
    print 'Joining with Gerrit Data'
    combined = join_with_gerrit(project, combined,
                                all_changes, all_change_details)

    jdump(combined, project_to_fname(project, combined=True))
    return combined


def join_with_bugs(commits, all_bugs):
    """Joins Launchpad Data with corresponding commits"""
    idx_bugno_by_cid = collections.defaultdict(list)
    for b in all_bugs.values():
        for e in b['commits']:
            idx_bugno_by_cid[e['cid']].append(b['id'])

    for cid, c in commits.items():
        c['bug_details'] = {}
        if 'bugs' in c:                # Join on bug number
            for bugno in c['bugs']:
                if bugno in all_bugs:
                    c['bug_details'][bugno] = all_bugs[bugno]
        if cid in idx_bugno_by_cid:  # Join on commit id
            for bugno in idx_bugno_by_cid[cid]:
                if bugno not in c['bug_details']:
                    c['bug_details'][bugno] = all_bugs[bugno]
    return commits


def join_with_gerrit(project, commits, all_changes, all_change_details):
    """Joins Gerrit data with corresponding commits"""
    # Index gerrit changes, pruning away changes off master branch
    change_by_changeid = dict([[c['change_id'], c]
                               for c in all_changes
                               if c['branch'] == 'master'
                               and c['project'].endswith(project)
                               and c['status'] == 'MERGED'])

    change_details_by_changeid = dict([[c['change_id'], c]
                                       for c in all_change_details
                                       if c['branch'] == 'master'
                                       and c['project'].endswith(project)
                                       and c['status'] == 'MERGED'])

    for cid, c in commits.items():
        c['change_details'] = False
        if 'change_id' in c:                # Join on chage_id
            change_id = c['change_id']
            if (change_id in change_by_changeid
                    and change_id in change_details_by_changeid):
                change = change_by_changeid[change_id]
                change.update(change_details_by_changeid[change_id])
                c['change_details'] = change
    return commits


#
# Routines for post-processing Dataset
#


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


def add_commit_features(c, feats,
                        include_committer=True,
                        include_order=True,
                        include_files=True,
                        include_lines_of_code=True,
                        include_blueprint=True,
                        include_cherrypick=True,):
    """Extract informaton related to Git Commit"""

    # General information about author
    author_name, author_org = parse_author_and_org(c['author'])
    feats['author'] = author_name
    feats['author_org'] = author_org
    feats['author_order'] = math.log(c['author_order'])

    # Information about committer
    if not include_committer:
        pass
    elif c['author'] != c['committer']:
        committer_name, _ = parse_author_and_org(c['committer'])
        feats['committer'] = committer_name
    else:
        feats['committer'] = 'same'

    if include_order:
        feats['order'] = math.log(c['order'])

    # General information around change (size (loc), code maturity,
    # prior bugs in module)
    if include_files:
        for fname in c['files']:
            feats[fname] = 1

            if include_order and c['file_order']:
                feats['min_file_order'] = \
                    math.log(min([v for v in c['file_order'].values()]))
                feats['max_file_order'] = \
                    math.log(max([v for v in c['file_order'].values()]))

    # Information about code changes
    if include_lines_of_code:
        feats['loc_add'] = c['loc_add']
        feats['loc_change'] = c['loc_change']
        if include_files:
            for fname, detail in c['loc_detail'].items():
                feats['loc_add_' + fname] = detail['add']
                feats['loc_changes_' + fname] = detail['changes']

    if include_blueprint:
        feats['blueprint'] = 'blueprint' in c

    if include_cherrypick:
        feats['cherry_picked_to'] = 'cherry_picked_to' in c
        feats['cherry_picked_from'] = 'cherry_picked_from' in c
    return


def add_bug_features(c, feats):
    """Extract information related to information in Launchpad"""
    global BUG_SEVERITY_VALUES
    global BUG_PRECEDENCE_VALUES
    bug_details = c['bug_details']  # list of bugs

    feats['lauchpad_bugs'] = len(bug_details)
    if len(bug_details) == 0:
        return

    # if this commit associated with a bug fix itself
    # feats['is_bug_fix'] = 'lp:id' in c

    feats['lauchpad_heat'] = max([bug['heat']
                                  for bug in bug_details.values()])
    feats['lauchpad_severity'] = max([BUG_SEVERITY_VALUES[bug['importance']]
                                      for bug in bug_details.values()
                                      if bug['importance']] + [0])
    feats['lauchpad_precedence'] = \
        max([BUG_PRECEDENCE_VALUES[bug['importance']]
             for bug in bug_details.values() if bug['importance']] + [0])
    feats['lauchpad_messages'] = \
        sum([len(bug['messages']) for bug in bug_details.values()])
    feats['lauchpad_security_related'] = \
        sum([1 for bug in bug_details.values()
             if bug['security_related']]) > 0
    feats['lauchpad_cve'] = sum([len(bug['cves'])
                                 for bug in bug_details.values()])
    return


def add_gerrit_features(c, feats, include_gerrit_details=True):
    """Extract features related to Gerrit history"""
    change_details = c['change_details']  # list of bugs
    if not change_details:
        feats['gerrit_has_data'] = False
        return
    feats['gerrit_has_data'] = True
    max_revision = max([message['_revision_number']
                        for message in change_details['messages']
                        if '_revision_number' in message])
    feats['gerrit_revision'] = max_revision

    if not include_gerrit_details:
        return

    if 'Workflow' in change_details['labels']:
        feats['gerrit_approved_workflow'] = \
            change_details['labels']['Workflow']['approved']['name']
    feats['gerrit_approved_code'] = \
        change_details['labels']['Code-Review']['approved']['name']
    feats['gerrit_votes'] = \
        sum([review['value'] for review
             in change_details['labels']['Code-Review']['all']])
    for review in change_details['labels']['Code-Review']['all']:
            feats['gerrit_reviewer_' + review['name']] = review['value']
    return


def create_feature(c,
                   include_committer=True,
                   include_order=True,
                   include_files=True,
                   include_lines_of_code=False,
                   include_blueprint=True,
                   include_cherrypick=True,
                   include_bug=True,
                   include_gerrit=True,
                   include_gerrit_details=True):
    """Extract features from combined_commits entry"""
    label = c['guilt']
    cid = c['cid']

    feats = {}
    add_commit_features(c, feats,
                        include_committer,
                        include_order,
                        include_files,
                        include_lines_of_code,
                        include_blueprint,
                        include_cherrypick)
    if include_bug:
        add_bug_features(c, feats)
    if include_gerrit:
        add_gerrit_features(c, feats, include_gerrit_details)

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
    result = collections.defaultdict(float)
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


def extract_features_helper(combined_commits, all_blame,
                            min_order, max_order,
                            offset, limit):
    """
    if not min_order and not max_order:
        min_order, max_order = trim_entries(combined_commits, all_blame)
    elif not min_order:
        min_order, _ = trim_entries(combined_commits, all_blame)
    elif not max_order:
        _, max_order = trim_entries(combined_commits, all_blame)
    """

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

    return cid, Y, features


def fit_features(combined_commits, all_blame,
                 min_order=False, max_order=False,
                 offset=0, limit=0):
    """Fits features in preparation for extract_features()
    Parameters:
    - min_order, max_order -- range of included commits.  full range
      by default
    - offset -- relative start, either as integer or percent
    - limit -- overall entries, either integer or percentd

    Returns (except for fit=True):
    - Labels
    - Feature Matrix
    - Feature matrix column names
    """

    vec = DictVectorizer()
    scaler = MinMaxScaler()
    extract_state = {'vec': vec, 'scaler': scaler}

    cid, Y, features = extract_features_helper(combined_commits,
                                               all_blame,
                                               min_order, max_order,
                                               offset, limit)

    X = vec.fit_transform([f for f in features]).toarray()
    X = scaler.fit_transform(X)
    return extract_state


def extract_features(combined_commits, all_blame, extract_state,
                     threshold=False,
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

    vec = extract_state['vec']
    scaler = extract_state['scaler']

    cid, Y, features = extract_features_helper(combined_commits,
                                               all_blame,
                                               min_order, max_order,
                                               offset, limit)

    X = vec.transform([f for f in features]).toarray()

    Y = np.asarray(Y)
    if clip:
        Y = np.minimum(Y, float(clip))

    # scaler = MinMaxScaler()     # Use MinMaxScaler non-gaussian data
    # X = scaler.fit_transform(X)
    X = scaler.transform(X)

    if debug:
        print 'total features:', len(features)
    if threshold:   # Quantize guilt
        Y = np.asarray(Y) > threshold
        if debug:
            print 'bugs based on threshold:', sum(Y)

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
    result = collections.defaultdict(float)
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
    commits_by_author = collections.defaultdict(float)
    for x in combined_commits.values():
        author = parse_author(x['author'])
        commits_by_author[author] += 1.0

    return commits_by_author


def get_blame_by_commit(combined_commits, all_blame):
    blame_by_commit = collections.defaultdict(float)
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
    skipped = 0
    instances = 0
    for be in all_blame:    # now apply weighted guilt for each blame
        v = combined_commits[be['cid']]
        if filter_bug_fix_combined_commits(v, importance=importance):
            for c, g in \
                    blame_compute_normalized_guilt(be,
                                                   exp_weighting=True).items():
                if c in combined_commits:
                    combined_commits[c]['guilt'] += g
                else:
                    skipped += 1
                instances += 1
    if skipped > 0:
        print
        print 'Warning - compute_guilt:  Skipped', skipped, 'entries out of',
        print instances

    total = len(combined_commits)
    guilty = sum([1 for v in combined_commits.values() if v['guilt'] > 0])
    min_guilt = min([v['guilt']
                     for v in combined_commits.values() if v['guilt'] > 0])
    max_guilt = max([v['guilt']
                     for v in combined_commits.values() if v['guilt'] > 0])

    print 'entries with non-zero guilt: ', guilty, 'out of',  total,
    print '(',  100.0 * float(guilty) / float(total), '%', ')'
    print 'smallest guilt:', min_guilt
    print 'largest guilt:', max_guilt


#
# Routines for selecting guilt threshold value
#

def count_guilty_commits(combined_commits, threshold):
    """Helper function for autoset_threshold()"""
    return sum([1 for v in combined_commits.values()
                if v['guilt'] >= threshold])


def autoset_threshold(combined_commits, actual):
    """Computes threshold value for guilt, later used to create
       Y labels during feature extraction, such that number of
       positive labels matches number of actual bugs
    """
    lower_thresh = min([v['guilt']
                        for v in combined_commits.values() if v['guilt'] > 0])
    upper_thresh = max([v['guilt']
                        for v in combined_commits.values() if v['guilt'] > 0])

    lower_count = count_guilty_commits(combined_commits, lower_thresh)
    upper_count = count_guilty_commits(combined_commits, upper_thresh)

    # verify that target bug count is within range
    if upper_count >= actual:
        return upper_thresh
    elif lower_count <= actual:
        return lower_threshold

    # use binary search to hoem in on threshold

    limit = 20
    last_mid_count = -1
    while limit > 0:
        limit -= 1

        mid_thresh = (upper_thresh + lower_thresh) / 2.0
        mid_count = count_guilty_commits(combined_commits, mid_thresh)

        if mid_count == actual or last_mid_count == mid_count:
            break

        last_mid_count = mid_count

        if mid_count < actual:
            upper_thresh = mid_thresh
            upper_count = mid_count
        else:
            lower_thresh = mid_thresh
            lower_count = mid_count

    return mid_thresh, mid_count
