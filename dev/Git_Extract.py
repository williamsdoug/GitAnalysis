#
# Git_Extract.py - Code to extract Git commit data, metadata, diff
#                       and blame.
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently being tested using OpenStack (Nova, Swift, Glance, Cinder, Heat)
#
# Last updated 4/26/2015
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
# - 3/4/15  - Added new file suffix to project_to_fname().  Used for
#             reachable commit data.
# - 3/5/15 -  Clean-up obsolete code and add new msg annotation code
# - 3/6/15 -  Compute order range for non-legacy commits.
# - 3/6/15 -  Moved find_legacy_cutoff from BugFixWorkflow to Git_Extract
#             to avoid circular import dependencies
# - 3/6/15 -  Updated annotate_commit_loc() to align with orther global changes
# - 3/9/15 -  Added caching support to LOC computation
# - 3/11/15 - Added compute_git_actor_dedupe().
# - 3/11/15 - Incorporate actor dedupe into annotate_author_order() and
#             annotate_file_order_by_author()
# - 4/23/15 - Fix change file identification in process_commit_files_unfiltered
# - 4/26/15 - Integrate PyDiff and and process_commit_diff feature extraction
#             from language_feature
#
#
# Top Level Routines:
#    from Git_Extract import build_git_commits, load_git_commits
#
#    from Git_Extract import get_git_master_commit, get_authors_and_files
#    from Git_Extract import filter_bug_fix_commits
#    from Git_Extract import filter_bug_fix_combined_commits
#
#     from Git_Extract import extract_master_commit, extract_origin_commit
#     from Git_Extract import get_all_files_from_commit
#     from Git_Extract import author_commiter_same
#
#     from Git_Extract import assign_blame, get_blame
#     from Git_Extract import project_to_fname
#     from Git_Extract import process_commit_files_unfiltered, filter_file
#     from Git_Extract import git_annotate_order
#     from Git_Extract import get_commit_ordering_min_max
#     from Git_Extract import find_legacy_cutoff
#     from Git_Extract import compute_git_actor_dedupe()
#

import git
from pprint import pprint
import collections
import re
import datetime
from datetime import datetime as dt
import sys

from jp_load_dump import convert_to_builtin_type, jload, jdump
from subprocess import Popen, PIPE, STDOUT
from multiprocessing import Pool

from git_analysis_config import get_filter_config
from git_analysis_config import get_repo_name, get_corpus_dir

from language_feature import process_commit_diff


#
# Configuration Variables
#

INSANELY_HUGE_BRANCH_DISTANCE = 1000000000


#
# Helper
#


def project_to_fname(project, patches=False, combined=False,
                     blame=False, reachable=False, loc=False):
    prefix = get_corpus_dir(project)
    if patches:
        return prefix + project + "_patch_data.jsonz"
    elif combined:
        return prefix + project + "_combined_commits.jsonz"
    elif blame:
        return prefix + project + "_all_blame.jsonz"
    elif reachable:
        return prefix + project + "_reachable.jsonz"
    elif loc:
        return prefix + project + "_loc.jsonz"
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


def parse_git_actor(txt):
    name = txt.split('"')[1].split('<')[0].strip()
    email = txt.split('<')[2].split('>')[0].strip()
    return name, email


def author_commiter_same(commit, relaxed=True):
    if not relaxed:
        return(commit['author'].lower() == commit['committer'].lower())

    # Relaxed rules - any of below:
    #  match on email address
    # match on text name
    author_name, author_email = parse_git_actor(commit['author'].lower())
    committer_name, \
        committer_email = parse_git_actor(commit['committer'].lower())
    return (author_name == committer_name or author_email == committer_email)


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


RE_BLUEPRINT_TEMPLATE1 = re.compile('blueprint:?\s*(\S+)$')
RE_BLUEPRINT_TEMPLATE2 = re.compile('bp:?\s*(\S+)$')


def get_commit_blueprint(msg):
    blueprint = False
    if 'blueprint' in msg.lower():
        for line in msg.lower().splitlines():
            if 'blueprint' in line:
                if '//blueprints.launchpad.net' in line:
                    blueprint = line.split('/')[-1]
                    return blueprint
                else:
                    m = RE_BLUEPRINT_TEMPLATE1.search(line)
                    if m:
                        blueprint = m.group(1)
                        if blueprint.endswith('.'):
                            blueprint = blueprint[:-1]
                        return blueprint
    if 'bp' in msg.lower():
        for line in msg.lower().splitlines():
            if 'bp' in line:
                m = RE_BLUEPRINT_TEMPLATE2.search(line)
                if m:
                    blueprint = m.group(1)
                    if blueprint.endswith('.'):
                        blueprint = blueprint[:-1]
                    return blueprint
    return False


def test_get_commit_blueprint():
    """Test code for above"""
    print get_commit_blueprint('bp:pci-passthrough-base')
    print get_commit_blueprint('bp: pci-passthrough-base')
    print get_commit_blueprint('blueprint:pci-passthrough-base')
    print get_commit_blueprint('blueprint: pci-passthrough-base')
    print get_commit_blueprint('booprint:pci-passthrough-base')
    print get_commit_blueprint('booprint: pci-passthrough-base')


def annotate_blueprints(commits):
    """Parses all commits referencing blueprints"""
    for c in commits.values():
        c['blueprint'] = False

    for c in commits.values():
        if c['on_master_branch']:
            blueprint = get_commit_blueprint(c['msg'])
            if blueprint:
                c['blueprint'] = blueprint


RE_CHERRY_TEMPLATE = re.compile('\(cherry picked from commit (([a-f0-9]){40})')


def annotate_cherry_pick(commits):
    """Parses all commit messages containing cherry-pick messages """
    for k, c in commits.items():
        if c['on_master_branch']:
            # m = RE_CHERRY_TEMPLATE.search(c['msg'])
            if 'cherry' in c['msg']:
                m = RE_CHERRY_TEMPLATE.search(c['msg'])
                if m:
                    picked_from = m.group(1)
                    if picked_from in commits:
                        c['cherry_picked_from'] = picked_from
                        if 'cherry_picked_to' not in commits[picked_from]:
                            commits[picked_from]['cherry_picked_to'] = []
                        commits[picked_from]['cherry_picked_to'].append(k)


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

def process_commits(repo, commits, filter_config,
                    skip=-1, limit=-1, verbose=False, use_pydiff=False):
    """Extracts all commit from git repo, subject to limit"""
    total_operations = 0
    total_errors = 0

    for h in repo.heads:
        for c in repo.iter_commits(h):
            if skip > 0:
                skip -= 1
                continue

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
            if use_pydiff:
                diff_result = process_commit_diff(c, filter_config,
                                                  verbose=verbose)
                commits[cid].update(diff_result)

            total_operations += 1
            if total_operations % 100 == 0:
                print '.',
                sys.stdout.flush()
            if total_operations % 1000 == 0:
                print total_operations,
                sys.stdout.flush()

            limit -= 1
            if limit == 0:
                return commits

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


def isValidBlob(blob):
    """Helper function to validate non-null blob"""
    return (blob and str(blob) != git.objects.blob.Blob.NULL_HEX_SHA)


def get_all_blob_paths(t):
    """Iterator recursively walks tree, returning blobs"""
    for b in t.blobs:
        yield(b.path)
    for subtree in t.trees:
        for b in get_all_blob_paths(subtree):
            yield(b)


def get_commit_diff_blob_paths(c, verbose=False):
    """Determine A and B blobs associated with commit pair"""

    if len(c.parents) > 0:
        p = c.parents[0]
        return [{'a_path': d.a_blob.path if isValidBlob(d.a_blob) else None,
                 'b_path': d.b_blob.path if isValidBlob(d.b_blob) else None}
                for d in c.diff(p, create_patch=False)]

    elif len(c.parents) == 0:
        # inaugural commit, so can't use diff
        return [{'a_path': b, 'b_path': None}
                for b in get_all_blob_paths(c.tree)]


def process_commit_files_unfiltered(c, verbose=False):
    """Determine files associated with an individual commit"""

    files = []
    # for p in c.parents:    # iterate through each parent
    if len(c.parents) > 0:
        p = c.parents[0]
        for d in c.diff(p, create_patch=False):
            if False:  # verbose:
                print
                print 'A:', d.a_blob,
                if isValidBlob(d.a_blob):
                    print d.a_blob.path
                print 'B:', d.b_blob,
                if isValidBlob(d.b_blob):
                    print d.b_blob.path
                sys.stdout.flush()
            if not isValidBlob(d.a_blob):
                if verbose:
                    print 'Delete'
                continue
            elif not isValidBlob(d.b_blob):
                if verbose:
                    print 'Add A'
                files.append(d.a_blob.path)
            elif (isValidBlob(d.a_blob) and isValidBlob(d.b_blob)
                  and d.b_blob.path.endswith('.py')):
                    files.append(d.b_blob.path)
    elif len(c.parents) == 0:
        # inaugural commit, so can't use diff
        files = [b for b in get_all_blob_paths(c.tree)]
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
    skip = False
    for line in diff_text.split('\n'):
        # print line
        if line.startswith('@@'):
            skip = False
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

                if p_len == 0:  # Special case for null files
                    skip = True
                    continue

                line_range += range(p_start, p_start+p_len)
            else:
                print 'not found', line
        elif skip:
            continue
        elif line.startswith('--- a/'):
            delete_flag = False
            pass
        elif line.startswith('+++ b/'):
            delete_flag = False
            pass
        elif line.startswith(' '):
            delete_flag = False
            p_pos += 1
            n_pos += 1
        elif line.startswith('+'):
            if delete_flag:
                changes.append(p_pos-1)
            else:
                changes.append(p_pos-1)
                changes.append(p_pos)
            delete_flag = False
            p_pos += 1
        elif line.startswith('-'):
            changes.append(p_pos)
            delete_flag = True
            n_pos += 1
        else:
            # Typically blank line or ''\ No newline at end of file'
            # print 'unused line:', line
            pass

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


#
# annotation code
#


def annotate_children(commits):
    for k, c in commits.items():
        c['children'] = []

    for k, c in commits.items():
        if c['on_master_branch']:
            for p in c['parents']:
                commits[p]['children'].append(k)


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


#
# Top Level Routines
#


def build_git_commits(project, update=True, include_patch=False,
                      use_pydiff=True, skip=-1, limit=-1, verbose=False):
    """Top level routine to generate commit data """

    repo_name = get_repo_name(project)
    repo = git.Repo(repo_name)
    assert repo.bare is False

    if update:
        try:
            # Commits will include pruned entries
            commits = load_git_commits(project)
        except Exception:
            update = False

    if not update:
        commits = collections.defaultdict(list)

    commits = process_commits(repo, commits, get_filter_config(project),
                              use_pydiff=use_pydiff, skip=skip, limit=limit,
                              verbose=verbose)
    print
    print 'total commits:', len(commits)
    """
    if include_patch:
        print 'Augment Git data with patch info'
        commits = update_commits_with_patch_data(commits, project)
        print
    """
    print 'Other commit message post-processing'
    print '  Blueprints'
    annotate_blueprints(commits)
    print '  Identifying cherry-pick commits'
    annotate_cherry_pick(commits)

    master_commit = get_git_master_commit(repo_name)
    print 'Master Commit:', master_commit
    commits = annotate_mainline(commits, master_commit)

    """
    print 'Consolidate Git Merge related commits'
    commits = consolidate_merge_commits(commits,
                                        get_git_master_commit(repo_name))
    """

    annotate_children(commits)   # note parent/child relationships
    jdump(commits, project_to_fname(project))


def load_git_commits(project):
    """Top level routine to load commit data, returns dict indexed by cid"""

    name = project_to_fname(project)
    result = jload(name)

    print '  total git_commits:', len(result)
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

    # print 'Calling get_blame', ranges, path
    warn = True

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

    # print cmd

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

            if line.startswith('fatal:'):
                if warn:
                    print 'Warning --', line,
                    print 'child cid:', child_cid
                    print 'parent cid:', cid
                    print 'command:', cmd
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

#
# Development Version
#

from language_feature import processDiffForBlame


limit = 2
def assign_blame2(d, path, diff_text, p_cid, repo_name,
                  child_cid, use_pydiff=True):
        """Combine diff with blame for a file"""
        global limit
        """try:"""
        if '+++ /dev/null' in diff_text:
            return [path, False]

        print
        print '+'*80
        print
        print 'Child CID:', child_cid
        print 'Parent CID:', p_cid
        print path

        result = parse_diff(diff_text)

        if use_pydiff and path.endswith('.py'):
            # print 'parse_diff:'
            # pprint(result)
            # print
            processDiffForBlame(d)
            print

        if not result:  # skip if empty
            return [path, False]

        ranges = find_diff_ranges(result)
        # print 'ranges'
        # pprint(ranges)

        limit -= 1
        if limit == 0:
            assert False

        # now annotate with blame information
        blame = dict([(int(x['lineno']), x)
                      for x in get_blame(p_cid, path, repo_name,
                                         child_cid=child_cid,
                                         ranges=ranges)])
        # print 'Blame:'
        # pprint(blame)
        result = [dict(x.items() + [['commit', blame[x['lineno']]['commit']]])
                  for x in result if x['lineno'] in blame]

        return [path, result]

        """except Exception:
        print 'Exception'
        assert False
        return [path, False] """


#
# Stable Version
#

def assign_blame(path, diff_text, p_cid, repo_name, child_cid):
    """Combine diff with blame for a file"""
    try:
        if '+++ /dev/null' in diff_text:
            return [path, False]

        result = parse_diff(diff_text)

        if not result:  # skip if empty
            return [path, False]

        ranges = find_diff_ranges(result)
        # print 'ranges'
        # pprint(ranges)

        # now annotate with blame information
        blame = dict([(int(x['lineno']), x)
                      for x in get_blame(p_cid, path, repo_name,
                                         child_cid=child_cid,
                                         ranges=ranges)])
        # print 'Blame:'
        # pprint(blame)
        result = [dict(x.items() + [['commit', blame[x['lineno']]['commit']]])
                  for x in result if x['lineno'] in blame]

        return [path, result]

    except Exception:
        print 'Exception'
        assert False
        return [path, False]

#
# Note: process_commit_details appears to be obsolete, replaced with
# BugFixWorkflow / compute_all_blame.  However, assign_blame appears to
# still be used
#

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

#
# Routines for extracting additional metadata from Jenkins for Jenkins
# authored commits
#


def identify_jenkins_commit(commits):
    """Finds commits where Jenkins is author """
    return [c['cid'] for c in commits.values()
            if c and 'jenkins@review.openstack.org' in c['author']]


#
# Determine date threshold for non-legacy commits
# Legacy commits are considerered commits propr to incorporation of
# Gerrit into tools chain -- ignored due tom complexity of extracting
# valid data from these commits since no Change_Id field in commit message.
#

def find_legacy_cutoff(commits, verbose=False):
    """Identifies commit timstamp for cut-over from legacy change
    tracking to use of Gerrit
    """
    last_without = False
    last_without_date = 0
    first_with = False
    first_with_date = 2e9

    for k, c in commits.items():
            if c['on_master_branch'] and c['on_mainline']:
                commit_date = int(c['date'])
                if ('review@openstack.org' in c['committer'] or
                    'openstack-infra@' in c['committer'] or
                    'change_id' in c and c['change_id']):
                        if commit_date < first_with_date:
                            first_with_date = commit_date
                            first_with = k
                else:
                    if commit_date > last_without_date:
                        last_without_date = commit_date
                        last_without = k

    # sanity check results
    if False:
        print '  First_with:', first_with
        print '  Last_without:', last_without

    transition_delta = (commits[first_with]['date']
                        - commits[last_without]['date'])
    if transition_delta > 0:
        if verbose:
            print '  Transition interval:',
            print str(datetime.timedelta(seconds=int(transition_delta)))
            print '  Parent of first_with:', commits[first_with]['parents'][0]
        assert(commits[first_with]['parents'][0] == last_without)
        if verbose:
            print '  Setting cutoff to:',
            print dt.fromtimestamp(first_with_date - 1).strftime("%d/%m/%Y")
        return first_with_date - 1
    else:
        if verbose:
            print '  Warning: Transition Overlap:',
            print str(datetime.timedelta(seconds=int(-transition_delta)))
            print '  Setting cutoff to:',
            print dt.fromtimestamp(last_without_date).strftime("%d/%m/%Y")
        return last_without_date

#
# Routines to annotate commits by order of change
#


def get_commit_ordering_min_max(commits):
    """Compute range of non-legacy ordered commits"""
    legacy_cutoff = find_legacy_cutoff(commits)
    all_orders = [c['order'] for c in commits.values()
                  if c['reachable'] and c['date'] > legacy_cutoff]
    min_order = min(all_orders)
    max_order = max(all_orders)
    return min_order, max_order


def git_annotate_commit_order(commits):
    """Asserts global ordering of commits based on commit date"""
    # Remove any prior ordering
    for k, c in commits.items():
        if 'order' in c:
            del commits[k]['order']

    commit_order = [[c['cid'], c['date']] for c in commits.values()
                    if c['reachable']]
    commit_order = sorted(commit_order, key=lambda z: z[1])
    for i, (cid, _) in enumerate(commit_order):
        commits[cid]['order'] = i + 1


def git_annotate_author_order(commits, git_actor_dedupe_table):
    """Order of change by author - author maturity"""
    author_commits = collections.defaultdict(list)

    for k, c in commits.items():
        if 'order' in c:
            author = git_actor_dedupe_table[c['author']]['standard_actor']
            author_commits[author].append((c['order'], k))

    for author, val in author_commits.items():
        for i, (order, c) in enumerate(sorted(val, key=lambda x: x[0])):
            commits[c]['author_order'] = i + 1


def git_annotate_file_order(commits):
    """Order of change by file - file maturity"""
    file_commits = collections.defaultdict(list)

    for k, c in commits.items():
        if 'order' in c:
            for fname in c['files']:
                file_commits[fname].append((c['order'], k))
            c['file_order'] = {}    # Use this as oppty to track on new field

    for fname, val in file_commits.items():
        for i, (order, c) in enumerate(sorted(val, key=lambda x: x[0])):
            commits[c]['file_order'][fname] = i + 1


def git_annotate_file_order_by_author(commits, git_actor_dedupe_table):
    """Order of change by file by author - author/file maturity"""
    file_commits_by_author = collections.defaultdict(
        lambda: collections.defaultdict(list))

    for k, c in commits.items():
        if 'order' in c:
            for fname in c['files']:
                author = git_actor_dedupe_table[c['author']]['standard_actor']
                file_commits_by_author[fname][author].append((c['order'], k))
                # Use this as opportunity to tack on new field
            c['file_order_for_author'] = {}

    for fname, entry in file_commits_by_author.items():
        for author, val in entry.items():
            for i, (order, c) in enumerate(sorted(val, key=lambda x: x[0])):
                commits[c]['file_order_for_author'][fname] = i + 1


def git_annotate_order(commits):
    """ Annotates commits with ordering information
        - Overall commit order
        - Order of commit on a per-file basis
        - Order of commits by author
        - Order of commits by author on per-file basis
    """
    git_actor_dedupe_table = compute_git_actor_dedupe(commits)
    git_annotate_commit_order(commits)
    git_annotate_author_order(commits, git_actor_dedupe_table)
    git_annotate_file_order(commits)
    git_annotate_file_order_by_author(commits, git_actor_dedupe_table)


#
# New Version
#
def annotate_commit_loc(commits, project, clear_cache=False):
    """Computes lines of code changed """
    print 'Annotating lines of code changed'
    cache = {}
    if not clear_cache:
        try:
            cache = jload(project_to_fname(project, loc=True))
            # Hack to remove artifacts left by jdump,
            # also remove any empty entries
            """
            for k, entry in cache.items():
                if entry:
                    if 'json_key' in entry:
                        del cache[k]['json_key']
                else:
                    del cache[k]
            """
            print '  Loaded Lines of Code Changed cache'

        except Exception:
            print '  Failed to load Lines of Code Changed cache'
            cache = {}
            pass

    cache_initial_size = len(cache)
    print '  Initial Lines of Code Changed cache size:', cache_initial_size

    repo_name = get_repo_name(project)
    filter_config = get_filter_config(project)
    repo = git.Repo(repo_name)
    total_operations = 0
    for k, commit in commits.items():
        if commit['reachable'] and 'loc_add' not in commit:
            if k not in cache:
                # print commit['cid']
                c = repo.commit(commit['cid'])
                loc_add = 0
                loc_change = 0
                detail = {}
                if len(c.parents) > 0:
                    p = c.parents[0]

                    files = process_commit_files_unfiltered(c)
                    subset_files = [f for f in files
                                    if filter_file(f, filter_config)]
                    for path in subset_files:
                        # print 'Getting diff object for path:', path
                        d = c.diff(p, create_patch=True, paths=path)
                        diff_text = d[0].diff
                        # print diff_text

                        adds = sum([1 for txt in diff_text.splitlines()
                                    if txt.startswith('+')]) - 1
                        removes = sum([1 for txt in diff_text.splitlines()
                                       if txt.startswith('-')]) - 1
                        changes = max(adds, removes)
                        detail[path] = {'add': adds, 'changes': changes}
                        loc_add += adds
                        loc_change += changes

                    cache[k] = {'loc_add': loc_add,
                                'loc_change': loc_change,
                                'loc_detail': detail}
                else:
                    cache[k] = {'loc_add': 0,
                                'loc_change': 0,
                                'loc_detail': {}}

            commit['loc_add'] = cache[k]['loc_add']
            commit['loc_change'] = cache[k]['loc_change']
            commit['loc_detail'] = cache[k]['loc_detail']

            total_operations += 1
            if total_operations % 100 == 0:
                    print '.',
            if total_operations % 1000 == 0:
                    print total_operations,
    print

    if len(cache) > cache_initial_size:
        print
        print '  Saving updated Lines of Code Changed Cache'
        jdump(cache, project_to_fname(project, loc=True))
        """
        # Hack to remove artifacts left by jdump
        for k in blame_cache.keys():   # remove key artifact from jload
            if 'json_key' in blame_cache[k]:
                del blame_cache[k]['json_key']
        """

#
# ------ Other Helper Routes, some not currently used --------
#


def get_git_master_commit(repo_name):
    """Returns ID of most recent commit"""
    repo = git.Repo(repo_name)
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


def compute_git_actor_dedupe(commits, runaway=10, verbose=False):
    """Cleanse Author and email data"""
    if verbose:
        print 'Deduplicate Git Actors'

    # Identify all unique actors, and timestamp for most recent activity
    git_actor_timestamps = collections.defaultdict(int)
    for c in commits.values():
        git_actor_timestamps[c['author']] = \
            max(c['date'], git_actor_timestamps[c['author']])
        git_actor_timestamps[c['committer']] = \
            max(c['date'], git_actor_timestamps[c['committer']])
    all_git_actors = git_actor_timestamps.keys()
    if verbose:
        print '  Total actors:', len(all_git_actors)

    # Build raw alias data
    name_to_actor = collections.defaultdict(list)
    name_to_email = collections.defaultdict(list)
    email_to_actor = collections.defaultdict(list)
    email_to_name = collections.defaultdict(list)

    for actor in all_git_actors:
        name, email = parse_git_actor(actor)
        name_to_actor[name].append(actor)
        name_to_email[name].append(email)
        email_to_actor[email].append(actor)
        email_to_name[email].append(name)

    if '' in name_to_email:   # handle special cas where blank
        del name_to_email['']
    if '' in email_to_name:
        del email_to_name['']

    # Select names and emails with the potential to be duplicates
    names_with_multi_email = \
        [{'names': set([k]), 'email': set(v)}
         for k, v in name_to_email.items() if len(v) > 1]
    email_with_multi_names = \
        [{'email': set([k]), 'names': set(v)}
         for k, v in email_to_name.items() if len(v) > 1]
    all_alias = names_with_multi_email + email_with_multi_names

    if verbose:
        print '  Names with multiple emails:', len(names_with_multi_email)
        print '  Email with multiple names:', len(email_with_multi_names)

    # Iteratively merge alias groups until converged on final set
    last_number_of_alias = -1   # Jump start refinement
    while len(all_alias) != last_number_of_alias and runaway > 0:
        runaway -= 1
        last_number_of_alias = len(all_alias)
        new_alias = []
        for this_alias in all_alias:
            merged = False
            # See if it overlaps with any already selected alias
            for i, current in enumerate(new_alias):
                if (current['names'].intersection(this_alias['names'])
                    or current['email'].intersection(this_alias['email'])):
                        merged = True
                        new_alias[i]['names'] = \
                            current['names'].union(this_alias['names'])
                        new_alias[i]['email'] = \
                            current['email'].union(this_alias['email'])
            if not merged:
                new_alias.append(this_alias)

        all_alias = new_alias
    if verbose:
        print '  Total Alias:', len(all_alias)

    # Select first amon equals for name, email and git_actor
    for alias in all_alias:
        alias['actors'] = set([actor for n in alias['names']
                               for actor in name_to_actor[n]]
                              + [actor for e in alias['email']
                                 for actor in email_to_actor[e]])

        actors_by_date = [[actor, git_actor_timestamps[actor]]
                          for actor in alias['actors']]
        actors_by_date = sorted(actors_by_date, key=lambda x: x[1],
                                reverse=True)
        standard_actor = actors_by_date[0][0]
        name, email = parse_git_actor(standard_actor)

        alias['standard_name'] = name
        alias['standard_email'] = email
        alias['standard_actor'] = standard_actor

    git_actor_dedupe_table = {}
    for actor in all_git_actors:
        name, email = parse_git_actor(actor)
        git_actor_dedupe_table[actor] = {'standard_name': name,
                                         'standard_email': email,
                                         'standard_actor': actor}

    # Incorporate alias data into lookup table
    for alias in all_alias:
        for actor in alias['actors']:
            git_actor_dedupe_table[actor] = alias

    return git_actor_dedupe_table
