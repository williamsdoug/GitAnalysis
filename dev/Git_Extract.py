#
# Git_Extract_Join.py - Code to extract Git commit data, metadata, diff
#                       and blame.
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently being tested using OpenStack (Nova, Swift, Glance, Cinder, Heat)
#
# Last updated 2/20/2014
#
# History:
# - 8/10/14: fix change_id (was Change-Id) for consistency, make leading I in
#            value uppercase
# - 8/24/14: try to address some of the issues preventing json persistance
# - 8/25/14: merge LP-Gerrit-Git-OpenStack-Analysis
# - 8/27/14: special case handling for jenkins@review.openstack.org commits
# - 8/31/14: only compute blame for selected file types (source code files) to
#            avoid git blame fatal errors and excessive runtime
# - 9/1/14:  consolidate top level routines into build_ and load_, introduce
#            standard naming based on project name, move code from iPython
#            notebook to .py file
# - 9/2/14:  Use additional patch information from Jenkins to annotate
#            commits with author jenkins@review.openstack.org
# - 9/3/14:  Filter out huge blame entries (default is >3000 total lines per
#            commit
# - 9/9/14:  Collect all file names
# - 9/10/14: Add exception handler to work-around unknown encoding error for
#            process_commits
# - 9/11/14: Fix committer field in update_commits_with_patch_data
# - 1/23/15: replace depricated os.popen() is subprocess.Popen()
# - 1/24/15: Changes to get_blame() to address fatal errors due to /dev/null
#            files.
# - 1/25/15: PEP-8 clean-up. Added ranges to git blame calls.
# - 1/25/15: Enabled use of multiprocessing in process_commit_details().
#            Refactored APIs for assign_blame(), get_blame() and parse_diff()
#            since multiprocessing.Pool requires that parameters in call to
#            assign_blame() be picked for insertion in Queue
# - 1/27/15: Created git_annotate_order() and supporting routines
# - 1/28/15: Removed nova as default project, clean-up defaults for
#            repo_name and project.  Remove nova hard-coding from
#            get_patch_data()
# - 1/30/15: Added annotate_commit_loc().  Computes lines-of-code-changed
#            for each commit.
# - 2/3/15 - Add update option to build_git_commits()
# - 2/3/15 - Eliminate use of globals
# - 2/3/15 - Enhanced build_all_blame to avoid re-computing large blame entries
# - 2/6/15 - cleaned-up filter_bug_fix_commits() and
#            filter_bug_fix_combined_commits()
# - 2/13/15 - Multiple new routines to handle git merge consolidation
# - 2/13/15 - moved file filter values to config variable FILTER_FILE_SUFFIX
#             process_commit_files(), process_commit_details() and
#             annotate_commit_loc().  Also extended to exclude certain
#             file prefixes
# - 2/13/15 - Disable patch data annotation by default in build_git_commits()
#             Enhancements to merge code appear to make this obsolete.
# - 2/13/15 - Added error handling for missing file to build_all_blame()
#             and build_git_commits()
# - 2/13/15 - Modify process_commit_files(), process_commit_details()
#             and annotate_commit_loc() to only follow primary parent [0]
# - 2/13/15 - Allow commits file to remember pruned entries (represented
#             as False). Add option load_git_commits() to filter prunced
#             entries to avoid impact to downstream code.
# - 2/13/15 - Second attempt at pruning, this time using tombstones.  Also
#             remember ancestry for pruned commits.
# - 2/17/15 - New parse_bugs() routine returns potentially multiple bugs
#             per commit.  Renamed field from commit['bug'] to commit['bugs']
# - 2/17/15 - Added aggregate_merge_bugs_and_changes() to merge flow
# - 2/17/15 - Temporarily disable combine_merge_commit(), move to a
#             post-processing step.
# - 2/19/15 - Extend parse_bugs() to support multiple bugs per line
# - 2/20/15 - clean-up usage of file_filter, updates to filter_file(),
#             process_commits(), process_commit_details() and
#             annotate_commit_loc().  Removed process_commit_files()
# - 2/20/15 - project_to_fname() leverages configuration data.
# - 2/20/15 - Downloads all commits in repo.  Commits in Master branch
#             tagged as ['on_master_branch'].  Commits on primary merge
#             path tagged as ['on_mainline'].  Distance now mreasured as
#             in c['distance_from_mainline']. ['is_master'] tag now
#             ['is_master_commit']
# - 2/20/15 - Consistently store c['change_id'] as list to simplify
#             post-processing
# - 2/24/15 - Join code now in commit_analysis (also re-written).
# - 2/24/15 - Renamed this file Git_Extract.py (was Git_Extract_Join.py).
# - 2/25/15 - Handle special case in aggregate_merge_bugs_and_changes()
#             where no ancestors.
# - 2/25/15 - Store single change_id value per commit.  Revert change
#             2/20/15 after verifying that multiple change_id per commit
#             is a spurious result and that the last change_id per message
#             if the definitive value.
# - 3/3/15  - fixed runaway limit in annotate_mainline().  Updated template
#             to detect Change_Id to be more forgiving of whitespace
# - 3/4/15  - Added new routines extract_master_commit(),
#             extract_origin_commit() and get_all_files_from_commit() used
#             by reachability code
#
# Top Level Routines:
#    from Git_Extract import build_git_commits, load_git_commits
#    from Git_Extract import build_all_blame, load_all_blame
#
#    from Git_Extract import get_git_master_commit, get_authors_and_files
#    from Git_Extract import filter_bug_fix_commits
#    from Git_Extract import  filter_bug_fix_combined_commits
#
#     from GitExtract import extract_master_commit, extract_origin_commit
#     from GitExtract import get_all_files_from_commit
#

from git import *
import git
import time
import pprint as pp
import collections
from collections import defaultdict
import re
import urllib2

from jp_load_dump import convert_to_builtin_type, pload, pdump, jload, jdump
from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool

from git_analysis_config import get_filter_config
from git_analysis_config import get_repo_name, get_corpus_dir


#
# Configuration Variables
#

INSANELY_HUGE_BRANCH_DISTANCE = 1000000000


#
# Helper
#


def project_to_fname(project, patches=False, combined=False, blame=False):
    prefix = get_corpus_dir(project)
    if patches:
        return prefix + project + "_patch_data.jsonz"
    elif combined:
        return prefix + project + "_combined_commits.jsonz"
    elif blame:
        return prefix + project + "_all_blame.jsonz"
    else:
        return prefix + project + "_commits.jsonz"


def filter_file(fname, filter_config):
    for prefix in filter_config['exclude_prefix']:
        if fname.startswith(prefix):
            return False
    for suffix in filter_config['include_suffix']:
        if fname.endswith(suffix):
            return True
    return False

#
# Parse Commit Messages
#

RE_GERRIT_TEMPLATE = re.compile('[Cc]hange-[Ii]d:\s*(I([a-f0-9]){40})')


def parse_all_changes(msg):
    """Parses all Change-Id: entries, typically on one per commit

    Example:
    parse_all_changes('Change-Id: I8fa8c4f36892b96d406216cb3c64854a94ca9df7')

    >>> 'I8fa8c4f36892b96d406216cb3c64854a94ca9df7'
    """
    result = []
    for line in msg.split('\n'):
        m = RE_GERRIT_TEMPLATE.search(line)
        if line.lower().startswith('change') and m:
            result.append(m.group(1))

    return result


bug_template = re.compile('bug[s:\s/-]*(?:lp|lp:|lp:#|lp:)*[\s#]*(\d+)',
                          re.IGNORECASE)


def parse_bugs(msg):
    """Extracts all bug id's from commit message"""
    bugs = []
    msg = msg.lower()
    for line in msg.split('\n'):
        if 'bug' not in line:
            # print 'rejected'
            continue

        # Immediate Acceptance
        if (line.startswith('closes')
                or line.startswith('fixes bug')
                or line.startswith('bug')
                or line.startswith('lp bug')
                or line.startswith('partial')
                or line.startswith('resolves')
                or line.startswith('backport for bug')):
            pass

        # Check for immediate rejections:
        elif ('depends on' in line
              or 'related' in line
              or 'http:/' in line
              or 'https://' in line
              or '/bug' in line
              ):

            continue
        # More complex acceptances
        elif not ('fix' in line
                  or '* bug' in line
                  or 'resolves bug' in line
                  or 'refer to bug' in line
                  or 'addresses bug' in line
                  ):
            continue

        """
        m = bug_template.search(line)
        if m:
            bugs.append(m.group(1))
        """

        m = bug_template.findall(line)
        for x in m:
            bugs.append(x)
    return list(set(bugs))


def test_parse_bugs():
    true_tests = ['fixed bug #1185609',
                  'Closes-bug: 1250158',
                  'Closes-Bug: #1300546',
                  'Partial-bug: #1277104',
                  'Fixes bug 1074132.',
                  'Fixes LP Bug#879136 - keyerror',
                  'Fixes: bug #1175815',
                  'This is a continuous fix to bug #1166957',
                  'This change fixes bug #1010560.',
                  'Bug #930543',
                  'Addresses bug 1154606.',
                  'Resolves bug 1030396.',
                  'Fixing bug 794582 - Now able to stream http(s) images',
                  'Disambiguates HTTP 401 and HTTP 403 in Glance. '
                  + 'Fixes bug 956513.'
                  'be removed from disk safely or not. Resolves bug 1030742.',
                  'check connection in Listener. refer to Bug #943031',
                  '  * Bug 955475',
                  'LP Bug#912800 - Delete image remain in cache',
                  'Fixes bug 1059634. Related to '
                  + 'I6cff4ee7f6c1dc970397b66fd2d15fa22b0a63a3',
                  'Backport for bug 803055',
                  'This addresses bug 767344.',
                  ]

    multi_tests = ['Fixes bug 1008874, bug 1046433.',
                   ]

    false_tests = ['Related-Bug: 1367908',
                   'launchpad bug https:'
                   + '//bugs.launchpad.net/oslo/+bug/1158807.',
                   'side effect it tests for the problem found '
                   + 'in bug: 1068051',
                   'currently impossible due to bug 1042925.',
                   'depends on bug 1214830',
                   'Prevent regression on bug 888370',
                   'pipes@serialcoder:~/repos/glance/bug730213$ '
                   + 'GLANCE_TEST_MIGRATIONS_CONF=/tmp/glance_test_'
                   + 'migrations.conf ./run_tests.sh -V test',
                   ]

    # testing positive matches
    for test in true_tests:
        bugs = parse_bugs(test)
        if len(bugs) == 0:
            print bugs, test

    for test in multi_tests:
        bugs = parse_bugs(test)
        if len(bugs) < 2:
            print bugs, test

    # testing negative matches
    for test in false_tests:
        bugs = parse_bugs(test)
        if len(bugs) != 0:
            print bugs, test


def parse_msg(msg, patch=False):
    """
    Overall routine for processing commit messages, also used for patch msgs
    """
    result = {}
    if not msg:
        return {}

    bugs = parse_bugs(msg)
    if bugs:
        result['bugs'] = bugs

    changes = parse_all_changes(msg)
    if changes:
        result['change_id'] = changes[-1]   # always use last value

    if patch:
        for line in msg.split('\n'):
            lline = line.lower()
            if line.startswith('From: '):
                try:
                    result['pAuth'] = '<git.Actor "'
                    + line[len('From: '):]+'">'
                except Exception:
                    pass
            elif line.startswith('Subject: '):
                try:
                    result['pSummary'] = line[len('Subject: '):]
                except Exception:
                    pass

    return result


#
# Basic Commit Processing
#

def process_commits(repo, commits, filter_config, max_count=False):
    """Extracts all commit from git repo, subject to max_count limit"""
    total_operations = 0
    total_errors = 0

    for h in repo.heads:
        for c in repo.iter_commits(h):
            # for c in repo.iter_commits('master', max_count=max_count):
            cid = c.hexsha
            if cid in commits:
                continue

            # try:
            commits[cid] = {'author': convert_to_builtin_type(c.author),
                            'date': c.committed_date,
                            'cid': c.hexsha,
                            'committer': convert_to_builtin_type(c.committer),
                            'msg': c.message.encode('ascii', 'ignore'),
                            'parents': [p.hexsha for p in c.parents]
                            }
            all_commit_files = process_commit_files_unfiltered(c)
            commits[cid]['unfiltered_files'] = all_commit_files
            commits[cid]['files'] = [f for f in all_commit_files
                                     if filter_file(f, filter_config)]

            commits[cid].update(parse_msg(c.message))
            total_operations += 1
            if total_operations % 100 == 0:
                print '.',
            if total_operations % 1000 == 0:
                print total_operations,

            """except Exception:
                print 'x',
                total_errors += 1"""

    if total_errors > 0:
        print
        print 'Commits skipped due to error:', total_errors

    # Now identify those commits within Master branch
    print
    print 'Annotating Master Branch'
    for cid in commits.keys():
        commits[cid]['on_master_branch'] = False
    for c in repo.iter_commits('master'):
            cid = c.hexsha
            if cid in commits:
                commits[cid]['on_master_branch'] = True

    return commits


def process_commit_files_unfiltered(c):
    """Determine files associated with an individual commit"""

    files = []
    # for p in c.parents:    # iterate through each parent
    if len(c.parents) > 0:
        p = c.parents[0]
        i = c.diff(p, create_patch=False)

        for d in i.iter_change_type('A'):
            if d.b_blob:
                files.append(d.b_blob.path)

        for d in i.iter_change_type('D'):
            if d.a_blob:
                files.append(d.a_blob.path)

        for d in i.iter_change_type('R'):
            if d.b_blob:
                files.append(d.b_blob.path)

        for d in i.iter_change_type('M'):
            if d.b_blob:
                files.append(d.b_blob.path)
    return files

hdr_re = re.compile('@@\s+-(?P<nstart>\d+)(,(?P<nlen>\d+))?\s+\+'
                    + '(?P<pstart>\d+)(,(?P<plen>\d+))?\s+@@')


def parse_diff(diff_text, proximity_limit=4):
    """
    Parses individual git diff str, returns range and line proximity to changes
    """

    p_pos = 0
    n_pos = 0
    changes = []
    line_range = []
    for line in diff_text.split('\n'):
        # print line
        if line.startswith('@@'):
            # print line
            m = hdr_re.search(line)
            if m:
                n_start = int(m.group('nstart'))
                n_pos = n_start
                try:
                    n_len = int(m.group('nlen'))
                except Exception:
                    n_len = 0
                p_start = int(m.group('pstart'))
                p_pos = p_start
                try:
                    p_len = int(m.group('plen'))
                except Exception:
                    p_len = 0

                line_range += range(n_start, n_start+n_len)
            else:
                print 'not found', line
        elif line.startswith('---'):
            delete_flag = False
            pass
        elif line.startswith('+++'):
            delete_flag = False
            pass
        elif line.startswith(' '):
            delete_flag = False
            p_pos += 1
            n_pos += 1
        elif line.startswith('+'):
            if delete_flag:
                changes.append(n_pos-1)
            else:
                changes.append(n_pos-1)
                changes.append(n_pos)
            delete_flag = False
            p_pos += 1
        elif line.startswith('-'):
            changes.append(n_pos)
            delete_flag = True
            n_pos += 1

    changes = set(sorted(set(changes)))
    window = set(sorted(set(line_range)))

    # print 'change:', changes
    # print 'range:', line_range

    proximity = 1
    prior = set([])
    result = []

    # while window.difference(set([x[0] for x in result])):
    while window.difference(prior):
        # print 'available:', window.difference(prior)
        result += [{'lineno': x, 'proximity': proximity} for x in changes]
        proximity += 1
        prior = prior.union(changes)

        # grow changes
        changes = set([x-1 for x in changes] + [x+1 for x in changes])
        changes = changes.difference(prior)
        # print 'new:', changes
        # print

        if proximity > proximity_limit:
            break

    return sorted([x for x in result if x['lineno'] in window])

#
# Routines to handle merge commits
#


def is_special_git_actor(field, names):
    """Helper function for is_special_author and is_special_committer"""
    for name in names:
        if name in field:
            return True
    return False


def is_special_author(commit, special_names=['jenkins@review.openstack.org',
                                             'jenkins@openstack.org']):
    """Identifies commits related to tools chain"""
    return is_special_git_actor(commit['author'], special_names)


def is_special_committer(commit,
                         special_names=['review@openstack.org',
                                        'hudson@openstack.org',
                                        'Tarmac']):
    """Identifies commits related to tools chain"""
    return is_special_git_actor(commit['committer'], special_names)


def is_merge_commit(commit, include_special_actor=False):
    """Identifies non-fastforward Git Merge commits"""
    if not (commit['on_master_branch']
            and commit['on_mainline']
            and len(commit['parents']) >= 2):
        return False
    if include_special_actor:
        return is_special_committer(commit)
    else:
        return True

"""
def combine_merge_commit(c, commits):
    "Promotes relevant information from second parent into
    merge commit
    "

    parent = commits[c['parents'][1]]

    if is_special_committer(c) and is_special_committer(parent):
        c['committer'] = c['author']
        c['author'] = parent['author']
    elif is_special_committer(c):
        c['author'] = parent['author']
        c['committer'] = parent['committer']
    else:
        c['author'] = parent['author']

    if 'change_id' in parent:
        c['change_id'] = parent['change_id']
    if 'bug' not in c and 'bug' in parent:
        c['bug'] = parent['bug']

    # c['msg'] = c['msg'] + '\n' + parent['msg']
    # c['parents'] = c['parents'][0:1]
"""


def annotate_mainline(commits, master_commit, runaway=1000000):
    """Master branch mainline is considered as lineage from Master
    commit by following first parent
    """
    # Reset branch-related values
    for v in commits.values():
        v['on_mainline'] = False
        v['distance_from_mainline'] = INSANELY_HUGE_BRANCH_DISTANCE
        v['is_master_commit'] = False
        v['ancestors'] = False
        v['tombstone'] = False
        v['merge_commit'] = False

    commits[master_commit]['is_master_commit'] = True
    commits[master_commit]['distance_from_mainline'] = 0
    current = master_commit

    while current and runaway > 0:
        runaway -= 1
        c = commits[current]
        c['on_mainline'] = True
        c['distance_from_mainline'] = 0
        if c['parents']:
            current = c['parents'][0]
        else:
            current = False

    return commits


def aggregate_merge_bugs_and_changes(commits):
    """Associates bugs and changes with each merge commit"""
    for k, c in commits.items():  # Initialize
        if c['on_mainline']:
            commits[k]['all_bugs'] = []
            commits[k]['all_changes'] = []
            commits[k]['children'] = []

    for k, c in commits.items():  # Accumulate bugs and changes for children
        if (c['on_master_branch'] and not c['on_mainline']
                and 'ancestors' in c and c['ancestors']):
            a = c['ancestors'][0]
            if not a:
                continue

            commits[a]['children'].append(k)

            if 'bugs' in c:
                commits[a]['all_bugs'] = commits[a]['all_bugs'] + c['bugs']

            if 'change_id' in c:
                commits[a]['all_changes'].append(c['change_id'])

    for k, c in commits.items():  # Dedupe results
        if c['on_mainline']:
            commits[k]['all_bugs'] = list(set(commits[k]['all_bugs']))
            commits[k]['all_changes'] = list(set(commits[k]['all_changes']))
    return commits


def consolidate_merge_commits(commits, master_commit,
                              verbose=True, runaway=1000000):
    """Clean-up for Git Merge commits (non-fast fordward)
    - OBSOLETE:Consolidates  all change-related information into merge commit
      - Author, Committer, change_id, bug ...
    - Eliminates all commits related to second parent
      - Including garbage collection for parents of parents ...
    - Established overall commit ordering
    """
    # Annotate mainline within master branch
    print 'Master Commit:', master_commit
    commits = annotate_mainline(commits, master_commit)

    # sanity-check merges (relative to master branch')
    # check_merge_stats(commits, verbose=verbose)

    # populate pruning last from merge commits
    prune = []    # prune entries [cid, ancestors]
    for c in commits.values():
        if is_merge_commit(c):
            c['merge_commit'] = True
            for p in c['parents'][1:]:
                ancestors = [c['cid']]
                prune.append([p, ancestors])
            # combine_merge_commit(c, commits)
        else:
            c['merge_commit'] = False

    if verbose:
        print 'initial commmits to be pruned:', len(prune)
        print 'starting commits:', len(commits)

    # prune commits and any predecessor commits no on master branch

    while prune and runaway > 0:
        runaway -= 1
        cid, ancestors = prune[0]
        del prune[0]
        if cid not in commits or not commits[cid]:
            continue
        c = commits[cid]

        for p in c['parents']:
            if (p in commits and commits[p]
               and not commits[p]['on_mainline']):
                prune.append([p, ancestors + [cid]])
        commits[cid]['tombstone'] = True
        this_branch_distance = len(ancestors)
        if this_branch_distance < commits[cid]['distance_from_mainline']:
            commits[cid]['ancestors'] = ancestors
            commits[cid]['distance_from_mainline'] = this_branch_distance

    # Prune anything outside of Master Branch
    for c in commits.values():
        if not c['on_master_branch']:
            c['tombstone'] = True

    # order commits based on timestamp
    commit_order = [[c['cid'], c['date']] for c in commits.values()
                    if not c['tombstone']]
    commit_order = sorted(commit_order, key=lambda z: z[1])
    for i, (cid, _) in enumerate(commit_order):
        commits[cid]['order'] = i + 1

    # aggregate bug and change information
    commits = aggregate_merge_bugs_and_changes(commits)

    if verbose:
        print 'commits after pruning:',
        print sum([1 for c in commits.values() if not c['tombstone']])

    return commits


#
# Top Level Routines
#


def build_git_commits(project, update=True, include_patch=False,
                      include_loc=False):
    """Top level routine to generate commit data """

    repo_name = get_repo_name(project)
    repo = Repo(repo_name)
    assert repo.bare is False

    if update:
        try:
            # Commits will include pruned entries
            commits = load_git_commits(project, prune=False)
        except Exception:
            update = False

    if not update:
        commits = collections.defaultdict(list)

    commits = process_commits(repo, commits, get_filter_config(project))
    print
    print 'total commits:', len(commits)
    if include_patch:
        print 'Augment Git data with patch info'
        commits = update_commits_with_patch_data(commits, project)
        print

    print 'Consolidate Git Merge related commits'
    commits = consolidate_merge_commits(commits,
                                        get_git_master_commit(repo_name))

    print 'Augment Git data with ordering info'
    git_annotate_order(commits, repo_name)

    if include_loc:
        print
        print 'Augment Git data with lines-of-code changed'
        annotate_commit_loc(commits, repo_name, get_filter_config(project))

    jdump(commits, project_to_fname(project))


def load_git_commits(project, prune=True):
    """Top level routine to load commit data, returns dict indexed by cid"""

    name = project_to_fname(project)
    result = jload(name)

    print '  total git_commits:', len(result)
    pruned = sum([1 for v in result.values() if v['tombstone']])
    print '  actual commits:', len(result) - pruned
    print '  pruned commits:', pruned
    print '  bug fix commits:', sum([1 for x in result.values()
                                     if x and 'bugs' in x])
    print '  commits with change_id:', sum([1 for x in result.values()
                                            if x and 'change_id' in x])
    print '  bug fix with change_id:', sum([1 for x in result.values()
                                            if x and 'change_id' in x
                                            and 'bugs' in x])
    return result


#
# Processing of Git Blame
#
# for changing reference point see:
#              http://stackoverflow.com/questions/5098256/git-blame-prior-commits


def get_blame(cid, path, repo_name,
              child_cid='', ranges=[],
              include_cc=False, warn=False):
    """Compute per-line blame for individual file for a commit """

    result = []
    entry = {}
    first = True      # identifies first line in each porcelin group

    # Build git blame command string
    cmd = 'cd ' + repo_name + ' ; ' + 'git blame --line-porcelain '
    for r in ranges:         # Add line number ranges
        cmd += '-L ' + str(r[0]) + ',' + str(r[1]) + ' '
    if include_cc:           # Optionally identiified copied or moved sequences
        cmd += '-C -C '
    cmd += cid + ' -- ' + path   # specify commit and filename

    # Now execute git blame command in subprocess and parse results
    with Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT).stdout as f:
        for line in f:
            try:
                line = str(line)
                # line = line.encode('ascii', errors='ignore')
            except Exception, e:
                print e
                print type(line), line
                raise Exception

            if 'fatal: no such path' in line:
                if warn:
                    print line,
                    print 'child cid:', child_cid
                    print 'parent cid:', cid
                return False

            if first:
                line = line[:-1]    # strip newline
                v = line.split(' ')
                entry['commit'] = v[0]
                entry['orig_lineno'] = v[1]
                entry['lineno'] = v[2]
                first = False
            elif line[0] == '\t':
                entry['text'] = line[1:].strip()
                result.append(entry)
                entry = {}
                first = True
            else:
                try:
                    k, v = line.split(' ', 1)
                    v = v[:-1]
                    # print k, v
                    entry[k] = v
                except Exception, e:
                    pass
    if len(result) == 0:
        if warn:
            print 'Warning -- Empty Blame for ', cid, path
        return False
    return result


def find_diff_ranges(entries):
    """Extracts range of relevant line numbers from parse_diff result"""
    ranges = []
    start = -1
    end = -1

    for e in entries:
        if e['lineno'] != end + 1:  # find discontinuity
            if start != -1:         # skip if first entry
                ranges.append([start, end])
            start = e['lineno']
        end = e['lineno']

    ranges.append([start, end])     # include final entry
    return ranges


def assign_blame(path, diff_text, p_cid, repo_name, child_cid):
    """Combine diff with blame for a file"""
    try:
        result = parse_diff(diff_text)
        ranges = find_diff_ranges(result)

        # now annotate with blame information
        blame = dict([(int(x['lineno']), x)
                      for x in get_blame(p_cid, path, repo_name,
                                         child_cid=child_cid,
                                         ranges=ranges)])
        result = [dict(x.items() + [['commit', blame[x['lineno']]['commit']]])
                  for x in result if x['lineno'] in blame]

        return [path, result]

    except Exception:
        return [path, False]


def process_commit_details(cid, repo, repo_name, filter_config):
    """Process individual commit, computing diff and identifying blame.
       Exclude non-source files
    """

    c = repo.commit(cid)
    pool = Pool()

    # blame = [assign_blame(d.b_blob.path, d.diff, p.hexsha,
    #                       repo_name, cid)
    output = []
    if len(c.parents) > 0:
        p = c.parents[0]
        output = [pool.apply_async(assign_blame,
                                   args=(d.b_blob.path,
                                         d.diff, p.hexsha,
                                         repo_name, cid))
                  # for p in c.parents    # iterate through each parent
                  for d in c.diff(p, create_patch=True).iter_change_type('M')
                  if (d.a_blob and d.b_blob
                      and filter_file(d.b_blob.path, filter_config)
                      and str(d.a_blob) != git.objects.blob.Blob.NULL_HEX_SHA
                      and str(d.b_blob) != git.objects.blob.Blob.NULL_HEX_SHA
                      and d.a_blob.size != 0
                      and d.b_blob.size != 0)
                  ]
    else:
        output = []

    blame = [p.get() for p in output]
    pool.close()
    pool.join()
    return dict(blame)


def compute_all_blame(bug_fix_commits, repo, repo_name, filter_config,
                      start=0, limit=1000000,
                      keep=set(['lineno', 'orig_lineno', 'commit', 'text'])):
    """Top level iterator for computing diff & blame for a list of commits"""
    progress = 0
    all_blame = []

    for cid in bug_fix_commits[start:start+limit]:
        all_blame.append(
            {'cid': cid,
             'blame': process_commit_details(cid, repo,
                                             repo_name, filter_config)})

        progress += 1
        # if progress % 100 == 0:
        if progress % 10 == 0:
            print '.',
        if progress % 1000 == 0:
            print progress

    return all_blame


def prune_huge_blame(entry, threshold=3000):
    total = 0
    for fdat in entry['blame'].values():
        if fdat:
            x = len(fdat)
            total += x
    if total >= threshold:
        entry['blame'] = {}


IMPORTANCE_VALUES = {
    'crit': ['Critical'],
    'high+': ['Critical', 'High'],
    'med+': ['Critical', 'High', 'Medium'],
    'low+': ['Critical', 'High', 'Medium', 'Low'],
    'all': ['Critical', 'High', 'Medium', 'Low', 'Wishlist',
            'Unknown', 'Undecided'],
}

STATUS_VALUES = {
    'committed': ['Fix Committed', ],
    'fixed': ['Fix Committed', 'Fix Released', ],
    'active': ['New', 'Confirmed', 'Triaged', 'In Progress', ],
    'all': ['Fix Committed', 'Fix Released', 'New', 'Incomplete', 'Opinion',
            'Invalid', "Won't Fix", 'Expired', 'Confirmed', 'Triaged',
            'In Progress', 'Incomplete', ]
}


def filter_bug_fix_combined_commits(v, importance='low+', status='fixed'):
    """Filter function for combined_commits based on type of bug"""
    importance = IMPORTANCE_VALUES[importance]
    status = STATUS_VALUES[status]

    return('lp:importance' in v and 'lp:status' in v
           and v['lp:importance'] and v['lp:importance'] in importance
           and v['lp:status'] and v['lp:status'] in status
           )


def filter_bug_fix_commits(v, importance='low+', status='fixed'):
    """Filter function for commits based on type of bug"""
    importance = IMPORTANCE_VALUES[importance]
    status = STATUS_VALUES[status]

    return('importance' in v and 'status' in v
           and v['importance'] and v['importance'] in importance
           and v['status'] and v['status'] in status
           )


def build_all_blame(project, combined_commits, update=True,
                    filt=filter_bug_fix_combined_commits):
    """Top level routine to generate or update blame data
    Parameters:
       project - name of project (used as prefix for all files)
       combined_commits - git-relared data
       update - determines whether full or incremental rebuild
       filt - function used to idnetify bugs
    """

    repo_name = get_repo_name(project)
    repo = Repo(repo_name)
    exclude_file_prefix = GET_FILE_EXCLUDE_PREFIXES(project)
    bug_fix_commits = set([k for k, v in combined_commits.items()
                           if filt(v)])
    print 'bug fix commits:', len(bug_fix_commits)

    if update:
        try:
            known_blame = set([x['cid'] for x in load_all_blame(project)])
            new_blame = bug_fix_commits.difference(known_blame)
            print 'new blame to be computed:', len(new_blame)
            if len(new_blame) > 0:
                new_blame = list(new_blame)
            else:
                return
        except Exception:
            update = False

    if not update:
        new_blame = list(bug_fix_commits)

    all_blame = compute_all_blame(new_blame, repo, repo_name,
                                  get_filter_config(project))
    # prune huge entries
    for x in all_blame:
        prune_huge_blame(x)

    print 'saving'
    if update:
        all_blame = load_all_blame(project) + all_blame

    jdump(all_blame, project_to_fname(project, blame=True))


def load_all_blame(project):
    """Top level routine to load blame data"""
    return jload(project_to_fname(project, blame=True))


#
# Routines for extracting additional metadata from Jenkins for Jenkins
# authored commits
#

def identify_jenkins_commit(commits):
    """Finds commits where Jenkins is author """
    return [c['cid'] for c in commits.values()
            if c and 'jenkins@review.openstack.org' in c['author']]


def get_patch_data(cid, project):
    """Downloads supplementary patch data from OpenStack cgit """
    # base = 'http://git.openstack.org/cgit/openstack/nova/patch/?id='
    template = 'http://git.openstack.org/cgit/openstack/{0}/patch/?id={1}'
    try:
        # f = urllib2.urlopen(urllib2.Request(base+cid))
        f = urllib2.urlopen(urllib2.Request(template.format(project, cid)))
        result = f.read()
        f.close()
        return result.split('diff --git')[0]
    except Exception:
        return False


def load_patch_data(jenkins_commits, project, incremental=True):
    """Downloads supplementary patch data for a set of commits """
    name = project_to_fname(project, patches=True)
    count = 0
    if incremental:
        try:
            patch_data = jload(name)
        except Exception:
            patch_data = []
    else:
        patch_data = []

    existing_commits = [x['cid'] for x in patch_data]
    delta_commits = set(jenkins_commits).difference(set(existing_commits))
    print 'known patches:', len(existing_commits)
    print 'requested patches:', len(jenkins_commits)
    print 'new patches to be fetched:', len(delta_commits)

    if len(delta_commits) == 0:
        return patch_data

    for cid in delta_commits:
        patch_data.append({'cid': cid, 'patch': get_patch_data(cid, project)})

        count += 1
        if count % 10 == 0:
            print '.',
        if count % 100 == 0:
            print count,
        if count % 1000 == 0:
            pdump(patch_data, name)

    jdump(patch_data, name)
    return patch_data


def update_commits_with_patch_data(commits, project):
    """Highest level routine - identifies all commits with jenkins author and
       augments metadata with cgit patch data
    """
    name = project_to_fname(project, patches=True)
    # fetch appropriate data from patches
    jenkins_commits = identify_jenkins_commit(commits)
    patch_commits = dict([(x['cid'], parse_msg(x['patch'], patch=True))
                          for x in load_patch_data(jenkins_commits,
                                                   project,
                                                   incremental=True)])

    # merge information into commits
    for cid in jenkins_commits:
        patch = patch_commits[cid]
        c = commits[cid]

        if 'bug' in patch and 'bug' not in c:
            commits[cid]['bug'] = patch['bug']

        if 'change_id' in patch and 'change_id' not in c:
            commits[cid]['change_id'] = patch['change_id']
            pass

        if 'pAuth' in patch:
            commits[cid]['author2'] = commits[cid]['author']
            commits[cid]['author'] = patch['pAuth']
            commits[cid]['committer'] = patch['pAuth']

        if 'pSummary' in patch and 'msg' not in c:
            commits[cid]['msg'] = patch['pSummary']

    return commits


#
# Routines to annotate commits by order of change
#


def git_annotate_author_order(commits):
    """Order of change by author - author maturity"""
    author_commits = collections.defaultdict(list)

    for k, c in commits.items():
        if not c['tombstone']:
            author_commits[c['author']].append((c['order'], k))

    for author, val in author_commits.items():
        for i, (order, c) in enumerate(sorted(val, key=lambda x: x[0])):
            commits[c]['author_order'] = i + 1


def git_annotate_file_order(commits):
    """Order of change by file - file maturity"""
    file_commits = collections.defaultdict(list)

    for k, c in commits.items():
        if not c['tombstone']:
            for fname in c['files']:
                file_commits[fname].append((c['order'], k))
            c['file_order'] = {}    # Use this as oppty to track on new field

    for fname, val in file_commits.items():
        for i, (order, c) in enumerate(sorted(val, key=lambda x: x[0])):
            commits[c]['file_order'][fname] = i + 1


def git_annotate_file_order_by_author(commits):
    """Order of change by file by author - author/file maturity"""
    file_commits_by_author = collections.defaultdict(
        lambda: collections.defaultdict(list))

    for k, c in commits.items():
        if not c['tombstone']:
            for fname in c['files']:
                file_commits_by_author[fname][c['author']].append((c['order'],
                                                                   k))
                # Use this as opportunity to tack on new field
            c['file_order_for_author'] = {}

    for fname, entry in file_commits_by_author.items():
        for author, val in entry.items():
            for i, (order, c) in enumerate(sorted(val, key=lambda x: x[0])):
                commits[c]['file_order_for_author'][fname] = i + 1


def git_annotate_order(commits, repo_name):
    """ Annotates commits with ordering information
        - Overall commit order   [handled in consolidate_merge_commits()]
        - Order of commit on a per-file basis
        - Order of commits by author
        - Order of commits by author on per-file basis
    """
    git_annotate_author_order(commits)
    git_annotate_file_order(commits)
    git_annotate_file_order_by_author(commits)


def annotate_commit_loc(commits, repo_name, filter_config):
    """Computes lines of code changed """

    repo = git.Repo(repo_name)
    total_operations = 0
    for commit in commits.values():
        if not commit['tombstone'] and 'loc_add' not in commit:
            # print commit['cid']
            c = repo.commit(commit['cid'])
            loc_add = 0
            loc_change = 0
            detail = {}
            if len(c.parents) > 0:
                p = c.parents[0]

                # for p in c.parents:    # iterate through each parent
                for d in c.diff(p, create_patch=True):
                    if d.a_blob and filter_file(d.a_blob.path,
                                                filter_config):
                        fname = d.a_blob.path
                        adds = sum([1 for txt in d.diff.splitlines()
                                    if txt.startswith('+')]) - 1
                        removes = sum([1 for txt in d.diff.splitlines()
                                       if txt.startswith('-')]) - 1
                        changes = max(adds, removes)
                        detail[fname] = {'add': adds, 'changes': changes}
                        loc_add += adds
                        loc_change += changes

            commit['loc_add'] = loc_add
            commit['loc_change'] = loc_change
            commit['loc_detail'] = detail

            total_operations += 1
            if total_operations % 100 == 0:
                    print '.',
            if total_operations % 1000 == 0:
                    print total_operations,

#
# ------ Other Helper Routes, some not currently used --------
#


def get_git_master_commit(repo_name):
    """Returns ID of most recent commit"""
    repo = Repo(repo_name)
    return repo.heads.master.commit.hexsha


def get_authors_and_files(commits):
    """Returns all unique authors and all unique files (naieve to rename) """
    authors = {}
    files = {}
    for c in commits.values():
        if not c['tombstone']:
            authors[c['author']] = 1
            for fn in c['files']:
                files[fn] = 1
    return authors.keys(), files.keys()


def extract_master_commit(commits):
    """ Extracts Master commit cid from commits datastructure"""
    master_cid = False
    for k, c in commits.items():
        if 'is_master_commit' in c and c['is_master_commit']:
            master_cid = k
            break
    assert(master_cid)
    return master_cid


def extract_origin_commit(commits):
    """ Extracts Origin commit cid from commits datastructure"""
    # Find current master commit
    master_cid = extract_master_commit(commits)

    # find_origin_commit
    origin_commit = master_cid
    while commits[origin_commit]['parents']:
        if commits[origin_commit]['parents'][0] in commits:
            origin_commit = commits[origin_commit]['parents'][0]
    assert (origin_commit != master_cid)
    return origin_commit


def get_all_files_from_tree(git_tree, depth=0):
    """Recursively walks Git tree to enumerate files reachable from commit"""
    all_files = []
    for subtree in git_tree.trees:
        all_files += get_all_files_from_tree(subtree, depth=depth+1)
    all_files += [blob.path for blob in git_tree.blobs]
    return all_files


def get_all_files_from_commit(cid, repo, filter_config, verbose=False):
    """Get all files reachable from specified commit,
       qualified by filtering rules
    """
    c = repo.commit(cid)
    git_tree = c.tree
    raw_files = get_all_files_from_tree(git_tree)
    if verbose:
        print 'Total raw files:', len(raw_files)
    subset_files = [f for f in raw_files
                    if filter_file(f, filter_config)]
    if verbose:
        print 'Total selected files:', len(subset_files)
    return subset_files
