
#
# BugFixWorkflow.py - Code to handle identification of bug fixes and
#                     merge processing
#
# Author:  Doug Williams - Copyright 2015
#
# Currently being tested using OpenStack (Nova, Swift, Glance, Cinder, Heat)
#
# Last updated 3/3/2015
#
# ###Issues:
# - Use of global variable ALL_BUG is a hack, needs to be cleaned up
# - Currently ignoring legacy commits
# - How should we handle blueprint commits?
# - Should we treat different A/C as a feature?, may call it sponsor(c)
#
# History:
# - 3/3/15: Initial version based on BugWorkflow3 notebook
# - 3/3/15: Relax collect_all_bug_fix_commits() assertions.  Detect
#           merge reverts as special case with merge commit has change_id
# - 3/4/15: Added annotate_commit_reachability() and suporting code
# - 3/4/15: Caching support for above
# - 3/5/15: Cosmetic changes.
#
# Top level routines:
# from BugFixWorkflow import import_all_bugs
# from BugFixWorkflow import build_all_guilt
# from BugFixWorkflow import annotate_guilt
# from BugFixWorkflow import annotate_commit_reachability


import pprint as pp
import re

# import collections
import datetime
from datetime import date

from commit_analysis import load_all_analysis_data
from commit_analysis import rebuild_all_analysis_data
from Git_Extract import assign_blame, get_blame
from commit_analysis import blame_compute_normalized_guilt
from Git_Extract import project_to_fname
from Git_Extract import process_commit_files_unfiltered, filter_file
from Git_Extract import extract_master_commit
from Git_Extract import get_all_files_from_commit

from jp_load_dump import jload, jdump
from jp_load_dump import pload, pdump
import git
from git import Repo

from git_analysis_config import get_filter_config
from git_analysis_config import get_repo_name
# from git_analysis_config import get_corpus_dir

#
# Global Variables
#

global ALL_BUGS
ALL_BUGS = False

from Git_Extract import IMPORTANCE_VALUES, STATUS_VALUES

BUG_WEIGHT_VALUES = {
    'Critical': 4,
    'High': 3,
    'Medium': 2,
    'Low': 1,
    'Wishlist': 0,
    'Unknown': 0,
    'Undecided': 0,
}


#
# Code
#


def dedupe_list(x):
    "Deduplicates list"
    return list(set(x))


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
                        # print 'Test 1'
                        if commit_date < first_with_date:
                            first_with_date = commit_date
                            first_with = k
                else:
                    if commit_date > last_without_date:
                        last_without_date = commit_date
                        last_without = k

    # sanity check results
    if verbose:
        print 'First_with:', first_with
        print 'Last_without:', last_without

    transition_delta = (commits[first_with]['date']
                        - commits[last_without]['date'])
    if transition_delta > 0:
        if verbose:
            print 'Transition interval:',
            print str(datetime.timedelta(seconds=int(transition_delta)))
            print 'Parent of first_with:', commits[first_with]['parents'][0]
        assert(commits[first_with]['parents'][0] == last_without)
        if verbose:
            print 'Setting cutoff to:',
            print datetime.datetime.fromtimestamp(first_with_date - 1).strftime("%d/%m/%Y")
        return first_with_date - 1
    else:
        if verbose:
            print 'Warning: Transition Overlap:',
            print str(datetime.timedelta(seconds=int(-transition_delta)))
            print 'Setting cutoff to:',
            print datetime.datetime.fromtimestamp(last_without_date).strftime("%d/%m/%Y")
        return last_without_date


#
# Bug Filtering (based on importance)
#

def import_all_bugs(all_bugs):
    """Hack to import bug data into namespace for use by filter routines"""
    global ALL_BUGS
    ALL_BUGS = all_bugs


def filter_bug(bugno, importance='low+', status='fixed', heat=-1):
    global IMPORTANCE_VALUES
    global STATUS_VALUES
    global ALL_BUGS

    if bugno not in ALL_BUGS:
        return False

    bug = ALL_BUGS[bugno]

    importance_values = IMPORTANCE_VALUES[importance]
    status_values = STATUS_VALUES[status]
    if not bug['status'] or bug['status'] not in status_values:
            return False

    if bug['importance'] and bug['importance'] in importance_values:
            return True

    if heat > 0 and int(bug['heat']) >= heat:
            return True

    return False


def filter_bugs(bugs, importance='low+', status='fixed', heat=-1):
        return [bug for bug in bugs
                if filter_bug(bug, importance, status, heat)]


def is_bug_fix(c, importance='low+', status='fixed', heat=-1):
    """Filter function for combined_based on type of bug"""
    if 'bug_details' not in c or len(c['bug_details']) == 0:
        return False

    return [len(filter_bugs(c['bug_details'].keys(),
            importance, status, heat=-1)) > 0]


#
# Parsing and Annotation Routines
#

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
    for c in commits.values():
        c['blueprint'] = False

    for c in commits.values():
        if c['on_master_branch']:
            blueprint = get_commit_blueprint(c['msg'])
            if blueprint:
                c['blueprint'] = blueprint


def find_blueprint_in_bugs(all_bugs, limit=-1, stats=True):
    result = []
    for bugno, bug in all_bugs.items():
        if 'blueprint' in bug['description'].lower():
            # print 'Blueprint found in description for bug:',  bugno
            # print bug['description']
            # print
            result.append(bugno)
            continue
        for msg in bug['messages']:
            if 'blueprint' in msg['content'].lower():
                # print 'Blueprint found in content for bug:',  bugno
                # print msg['content']
                # print
                result.append(bugno)
                break

        limit -= 1
        if limit == 0:
            break

    if stats:
        count = len(result)
        print 'Found blueprint refereces in:', count,
        print 'out of :', len(all_bugs),
        print '   (', 100.0*float(count)/float(len(all_bugs)), '%)'

    return result


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


RE_CHERRY_TEMPLATE = re.compile('\(cherry picked from commit (([a-f0-9]){40})')


def annotate_cherry_pick(commits):
    for k, c in commits.items():
        if c['on_master_branch']:
            # m = RE_CHERRY_TEMPLATE.search(c['msg'])
            if 'cherry' in c['msg']:
                m = RE_CHERRY_TEMPLATE.search(c['msg'])
                if m:
                    cherry_picked_from = m.group(1)
                    if cherry_picked_from in commits:
                        c['cherry_picked_from'] = cherry_picked_from
                        if 'cherry_picked_to' not in commits[cherry_picked_from]:
                            commits[cherry_picked_from]['cherry_picked_to'] = []
                        commits[cherry_picked_from]['cherry_picked_to'].append(k)


def annotate_children(commits):
    for k, c in commits.items():
        c['children'] = []

    for k, c in commits.items():
        if c['on_master_branch']:
            for p in c['parents']:
                commits[p]['children'].append(k)


#
#
# Merge and Bug Fix Processing
#
#

#
# Individual Change merge-type specific helper functions (currently unused)
#

def process_ffc_change(c, verbose=False):
    if verbose:
        print 'FFC:', c['cid']
    c['is_tracked_change'] = True
    if not author_commiter_same(c):
        # print '    different author and committer'
        pass
    pass


def process_simple_merge_change(c, commits, verbose=False):
    if verbose:
        print 'Simple Merge:', c['cid']

    # USE PARENT COMMIT FOR CHANGE_ID AND READ_AUTHOR
    if not author_commiter_same(commits[c['parents'][1]]):
        # print '    different author and committer'
        pass
    pass


def process_complex_merge_change(c, commits, verbose=False):
    if verbose:
        print 'Complex Merge:', c['cid']
    pass


#
# Individual Bug Fix merge-type specific helper functions
#

def process_ffc_bug_fix(c, importance, verbose=False):
    if verbose:
        print 'FFC Bug Fix:', c['cid']
    diff_commit = c['cid']
    blame_commit = c['parents'][0]

    relevant_bugs = []  # NEED TO COMPUTE

    return {'cid': diff_commit, 'diff_commit': diff_commit,
            'blame_commit': blame_commit,
            'bugs': filter_bugs(c['bug_details'].keys(), importance),
            'type': 'fast_forward', 'merge_commit': c['cid']}


def process_simple_merge_bug_fix(c, commits, sub_branch_data,
                                 verbose=False, runaway=100000):
    diff_commit = c['parents'][1]
    blame_commit = c['parents'][1]  # find first ancestor on mainline
    while runaway > 0 and not commits[blame_commit]['on_mainline']:
        blame_commit = commits[blame_commit]['parents'][0]
        runaway -= 1

    if verbose:
        print 'Simple Merge Bug Fix:', c['cid']
        print '    ', sub_branch_data['unique_bugs']
        print '  cid for blame:', blame_commit

    return {'cid': diff_commit, 'diff_commit': diff_commit,
            'blame_commit': blame_commit,
            'bugs': sub_branch_data['unique_bugs'],
            'type': 'simple_merge', 'merge_commit': c['cid']}
    pass


def process_complex_merge_bug_fix(c, commits, sub_branch_data, verbose=False):
    if verbose:
        print 'Complex Merge Bug Fix:', c['cid']

    results = []
    unique_bugs = set(sub_branch_data['unique_bugs'])
    for change_cid in sub_branch_data['commits_with_changes']:
        change_commit = commits[change_cid]
        if 'bug_details' in change_commit:
            current_bugs = set(change_commit['bug_details'].keys())
            relevant_bugs = current_bugs.intersection(unique_bugs)
            if len(relevant_bugs) == 0:
                continue

            if verbose:
                print 'Selected commit:', change_cid,
                print 'for bug', list(relevant_bugs)

            # Method 1:  First commit in branch
            if verbose:
                print '   Parent', change_commit['parents'][0]
            if commits[change_commit['parents'][0]]['on_mainline']:
                diff_commit = change_cid
                blame_commit = change_commit['parents'][0]
                results.append({'cid': change_cid, 'diff_commit': diff_commit,
                                'blame_commit': blame_commit,
                                'bugs': relevant_bugs,
                                'type': 'complex_merge_1',
                                'merge_commit': c['cid']})
                if False:
                    print '   Diff:', diff_commit
                    print '  Blame:', blame_commit
                    print
                continue

            # Method 2:  See if any children on mainline
            # print '  Children', change_commit['children']
            mainline_children = [cid for cid in change_commit['children']
                                 if commits[cid]['on_mainline']]
            if mainline_children:
                diff_commit = mainline_children[0]
                blame_commit = commits[diff_commit]['parents'][0]
                results.append({'cid': change_cid, 'diff_commit': diff_commit,
                                'blame_commit': blame_commit,
                                'bugs': relevant_bugs,
                                'type': 'complex_merge_2',
                                'merge_commit': c['cid']})
                if False:
                    print '   Diff:', diff_commit
                    print '  Blame:', blame_commit
                    print
                continue

            # Method 3: Probably stable branch, try to compare
            # with predecessor commit
            if len(change_commit['parents']) == 1:
                if verbose:
                    print 'Method 3 complex commit:', change_cid
                    if 'cherry_picked_to' in change_commit:
                        print 'Commit:', change_cid
                        print '  Cherry picked to:',
                        print change_commit['cherry_picked_to']
                    if 'cherry_picked_from' in change_commit:
                        print 'Commit:', change_cid
                        print '  Cherry picked from:',
                        print change_commit['cherry_picked_from']

                diff_commit = change_cid
                blame_commit = change_commit['parents'][0]
                results.append({'cid': change_cid, 'diff_commit': diff_commit,
                                'blame_commit': blame_commit,
                                'bugs': relevant_bugs,
                                'type': 'complex_merge_3',
                                'merge_commit': c['cid']})
                if False:
                    print '   Diff:', diff_commit
                    print '  Blame:', blame_commit
                    print
                continue

            print '** Unable to identify diff commit'
            assert (diff_commit)

    if verbose:
        pp.pprint(results)
        print

    return results


#
# Organize commits by branches
#


def get_sub_branch_data(commits, k, importance='low+', depth=1,
                        limit_depth=10000):
    # check for runaway - mabe be able to delete
    if depth > limit_depth:
        raise Exception

    max_depth = depth
    c = commits[k]
    authors = [c['author']]
    committers = [c['committer']]
    commits_with_changes = []

    if 'bug_details' in c and c['bug_details']:
        bugs = filter_bugs(c['bug_details'].keys(), importance)
    else:
        bugs = []

    if 'change_id' in c and c['change_id']:
        changes = [c['change_id']]
        commits_with_changes.append(k)
    else:
        changes = []

    max_parents = len(c['parents'])

    for p in c['parents']:
        if not commits[p]['on_mainline']:
            p_authors, p_committers, p_bugs, p_changes, \
                p_commits_with_changes, p_parents, \
                p_depth = get_sub_branch_data(commits, p, importance,
                                              depth + 1)
            authors = authors + p_authors
            committers = committers + p_committers
            bugs = bugs + p_bugs
            changes = changes + p_changes
            commits_with_changes = (commits_with_changes +
                                    p_commits_with_changes)
            max_parents = max(max_parents, p_parents)
            max_depth = max(max_depth, p_depth)

    return [dedupe_list(authors), dedupe_list(committers),
            dedupe_list(bugs), dedupe_list(changes),
            dedupe_list(commits_with_changes), max_parents, max_depth]


def get_all_branch_data(commits, importance='low+', limit=-1):
    sub_branch_data = {}
    for k, c in commits.items():
        if (c['on_master_branch'] and c['on_mainline'] and
            len(c['parents']) > 1):
                authors, committers, bugs, changes, \
                    commits_with_changes, parents, \
                    depth = get_sub_branch_data(commits, c['parents'][1],
                                                importance)
                if depth > 0:
                    sub_branch_data[k] = {'branch_cid': c['parents'][1],
                                          'branch_depth': depth,
                                          'unique_bugs': bugs,
                                          'unique_changes': changes,
                                          'commits_with_changes':
                                              commits_with_changes,
                                          'unique_authors': authors,
                                          'unique_committers': committers,
                                          'max_parents': parents
                                          }
                    limit -= 1
                    if limit == 0:
                        break
        else:
            sub_branch_data[k] = False
    return sub_branch_data


#
# Top level bug fix and merge processing
#

def collect_all_bug_fix_commits(commits, importance,
                                legacy_cutoff, limit=-1):
    """Identified commits needed to compute guilt.  Also basic
    merge processing
    """

    legacy_ignored = 0
    total_mainline = 0

    guilt_data = []

    all_sub_branches = get_all_branch_data(commits, importance=importance)
    for k, c in commits.items():
        # Identified commit to be included during feature extraction
        c['is_tracked_change'] = False

    for k, c in commits.items():
        if not c['on_master_branch'] or not c['on_mainline']:
            continue

        total_mainline += 1
        # skip legacy for now
        if c['date'] <= legacy_cutoff:
            legacy_ignored += 1
            continue

        limit -= 1
        if limit == 0:
            break

        # Special case for origin commit:
        if len(c['parents']) == 0:
            c['is_tracked_change'] = True
            continue

        # Handle Fast-Forward Commits
        if len(c['parents']) == 1:    # Fast Forward Commit
            if is_bug_fix(c, importance):
                # Bug Fix commit
                guilt_data.append(process_ffc_bug_fix(c, importance))
                process_ffc_change(c)
            elif c['change_id']:
                process_ffc_change(c)
            else:                 # Down't exist for Glance
                print 'collect_all_bug_fix_commits: FFC with no changeid:', k
                raise Exception

        else:
            sub_branch_data = all_sub_branches[k]

            # Correctness checks
            if not sub_branch_data:
                print 'Merge Commit missing sub-branch_data:', k
                raise Exception

            if len(sub_branch_data['unique_changes']) == 0:
                print 'Warning: Merge commit ', k, 'has no changes'
                print '  Bugs:', len(sub_branch_data['unique_bugs'])
                print '  Time relative to legacy cut-off:',
                print int(c['date']) - legacy_cutoff
                if len(sub_branch_data['unique_bugs']) == 0:
                    print '   Ignoring sub-branch'
                    continue
            """
            if c['date'] > legacy_cutoff:
                assert(len(sub_branch_data['unique_changes']) > 0)
                    # changes must always have change_id
            """

            # Simple merge commit
            if len(sub_branch_data['unique_changes']) == 1:
                # check for reverts
                if ('Revert' in c['msg'].split('\n')[0]
                    and 'Revert' in
                    commits[c['parents'][1]]['msg'].split('\n')[0]
                        and 'change_id' not in commits[c['parents'][1]]):
                            print 'Ignoring revert:', c['cid']
                            continue

                # change_id should not be on merge_commit
                if 'change_id' in c:
                    print 'Unexpected change_id on simple merge, ignored',
                    print c['cid']
                    if len(c['files']) == 0:
                        continue
                    else:
                        assert ('change_id' not in c)

                # change_id always attached to parent
                assert (commits[c['parents'][1]]['change_id'])

                process_simple_merge_change(c, commits)
                if len(sub_branch_data['unique_bugs']) > 0:
                    guilt_data.append(process_simple_merge_bug_fix(c, commits, sub_branch_data))
                pass
            else:
                process_complex_merge_change(c, commits)
                if len(sub_branch_data['unique_bugs']) > 0:
                    guilt_data += process_complex_merge_bug_fix(c, commits, sub_branch_data)
                pass

        pass
    print 'Mainline Commits ignored due to legacy:',
    print legacy_ignored, ' out of:', total_mainline
    print 'Total commite requiring blame computation:', len(guilt_data)
    return guilt_data


#
# Blame and Guilt Processing
#

def compute_all_blame(project, guilt_data, combined_commits,
                      clear_cache=False):
    """Computes blame data for commits identified in guilt_data"""

    repo_name = get_repo_name(project)
    repo = Repo(repo_name)
    filter_config = get_filter_config(project)

    blame_cache = {}
    try:
        if not clear_cache:
            blame_cache = jload(project_to_fname(project, blame=True))
            # Hack to remove artifacts left by jdump,
            # also remove any empty entries
            for k, entry in blame_cache.items():
                if entry:
                    if 'json_key' in entry:
                        del blame_cache[k]['json_key']
                else:
                    del blame_cache[k]
            print 'Loaded blame'

    except Exception:
        print 'Failed to load blame'
        blame_cache = {}
        pass

    blame_cache_initial_size = len(blame_cache)
    print 'Initial Blame cache size:', blame_cache_initial_size

    print 'bug fix commits:', len(guilt_data)

    progress = 0
    for be in guilt_data:
        # pp.pprint(be)
        # {'cid': change_cid, 'diff_commit': diff_commit,
        #  'blame_commit': blame_commit, 'bugs':relevant_bugs,
        #  'type': 'complex_merge', 'merge_commit': c['cid']}
        c = repo.commit(be['diff_commit'])
        p = repo.commit(be['blame_commit'])

        bc_key = '__'.join([be['diff_commit'], be['blame_commit']])

        if bc_key not in blame_cache:
            commit_blame = {}
            files = process_commit_files_unfiltered(c)
            subset_files = [f for f in files
                            if filter_file(f, filter_config)]
            for path in subset_files:
                # print 'Getting diff object for path:', path
                d = c.diff(p, create_patch=True, paths=path)
                diff_text = d[0].diff
                # print diff_text
                fname, blame_data = assign_blame(path, diff_text,
                                                 be['blame_commit'],
                                                 repo_name,
                                                 be['diff_commit'])
                commit_blame[fname] = blame_data
            blame_cache[bc_key] = commit_blame    # Now populate cache

        be['blame'] = blame_cache[bc_key]

        # pp.pprint(be)
        # print be['diff_commit'], be['blame'].keys()

        progress += 1
        # if progress % 100 == 0:
        if progress % 10 == 0:
            print '.',
        if progress % 100 == 0:
            print progress,

    # prune huge entries
    if False:
        for x in all_blame:
            prune_huge_blame(x)

    if len(blame_cache) > blame_cache_initial_size:
        print
        print 'Saving updated Blame Cache'
        jdump(blame_cache, project_to_fname(project, blame=True))
        # Hack to remove artifacts left by jdump
        for k in blame_cache.keys():   # remove key artifact from jload
            if 'json_key' in blame_cache[k]:
                del blame_cache[k]['json_key']

#
# Identify Bug Fixes along Mainline
#


def find_missing_guilt_data(guilt_data):
    return [entry for entry in guilt_data if not entry['blame']]


def annotate_guilt(guilt_data, commits, limit=-1):
    """ """
    for k, c in commits.items():  # Initialize
        c['guilt'] = 0.0

    for entry in guilt_data:
        if not entry['blame']:   # skip empty entries
            continue
        limit -= 1
        if limit == 0:
            break

        for commit_key, guilt in blame_compute_normalized_guilt(entry).items():
            commits[commit_key]['guilt'] += guilt


def build_all_guilt(project, combined_commits,
                    clear_cache=False, apply_guilt=False,
                    importance='low+'):
    """Top level routine for Merge processing and guilt annotation"""

    print 'Determining legacy cut-off'
    legacy_cutoff = find_legacy_cutoff(combined_commits, verbose=True)

    # annotate_blueprints(commits)
    # foo = find_blueprint_in_bugs(all_bugs, limit=100000)
    print
    print 'Determining commit parent/child relationships'
    annotate_children(combined_commits)
    print
    print 'Identifying cherry-pick commits'
    annotate_cherry_pick(combined_commits)
    print
    print 'Collecting data on commits with bug fixes'
    guilt_data = collect_all_bug_fix_commits(combined_commits,
                                             importance,
                                             legacy_cutoff,
                                             limit=-1)
    print
    print 'Computing Blame'
    compute_all_blame(project, guilt_data,
                      combined_commits, clear_cache=clear_cache)
    print
    missing_guilt_data = find_missing_guilt_data(guilt_data)
    if len(missing_guilt_data) > 0:
        print
        print 'Warning: Entries with missing guilt data:',
        print len(missing_guilt_data)
    if apply_guilt:
        print
        print 'Annotating Guilt'
        annotate_guilt(guilt_data, combined_commits)
    return guilt_data


def top_level(rebuild=False, rebuild_with_download=False, rebuild_incr=True):
    if rebuild:    # Only rebuilds combined_commits
        rebuild_all_analysis_data(PROJECT, update=rebuild_incr,
                                  download=rebuild_with_download)

    all_bugs, all_changes, all_change_details, \
        commits, combined_commits, all_blame = load_all_analysis_data(PROJECT)

    import_all_bugs(all_bugs)

    guilt_data = build_all_guilt(PROJECT, combined_commits, clear_cache=True)


#
# Code for determining commit reachability
#


def sample_master_branch_commits(commits, master_cid,
                                 sampling_freq=100, legacy_cutoff=0):
    """ Extracts commits cid along master branch at regular intervals
        Always includes master head commit and first commit after cut-off
    """
    samples = [master_cid]

    current = master_cid
    last = False
    count = 0
    while (commits[current]['parents']
           and commits[current]['date'] > legacy_cutoff):
        if commits[current]['parents'][0] in commits:
            last = current
            current = commits[current]['parents'][0]
            count += 1
            if (count % sampling_freq) == 0:
                samples.append(current)
        else:
            break

    # Include first commit after legacy cut-off
    if last and last != samples[-1:]:
        samples.append(last)

    return samples


def identify_reachable_commits(project, commits, legacy_cutoff=0,
                               sampling_freq=25, clear_cache=False):
    """Sampling along master branch, using git, determine
    commit reachability
    """
    print 'Identify reachable commits'
    master_cid = extract_master_commit(commits)
    repo_name = get_repo_name(project)
    repo = Repo(repo_name)
    filter_config = get_filter_config(project)
    all_reachable_commits = set([])

    # Load cache, if available
    #
    # Cache format: {sampling_freq,
    #                min_date,
    #                max_date,
    #                reachability_sets: {cid:{date:,
    #                                    commits:[list of reachable_commits]}
    #
    reachable_cache = False
    try:
        if not clear_cache:
            reachable_cache = pload(project_to_fname(project, reachable=True))
            if sampling_freq != reachable_cache['sampling_freq']:
                print '****Reachable cache sampling frequency mis-match'
                raise Exception
            print '    Loaded cached reachability data'

    except Exception:
        print 'Failed to load reachability data'
        clear_cache = True

    if clear_cache:
        reachable_cache = {'sampling_freq': sampling_freq,
                           'min_date': False,
                           'max_date': False,
                           'reachability_sets': {}
                           }

    cache_initial_size = len(reachable_cache['reachability_sets'])
    print '    Initial Reachable cache size:', cache_initial_size

    # Sample commits at regular intervals along the master branch
    sample_cids = sample_master_branch_commits(commits, master_cid,
                                               sampling_freq, legacy_cutoff)
    print '    Samples:', len(sample_cids)
    all_reachable = set([])
    if cache_initial_size > 0:
        sample_max_date = max([commits[cid]['date'] for cid in sample_cids])
        sample_min_date = min([commits[cid]['date'] for cid in sample_cids])
        assert (sample_max_date >= sample_min_date)
        cache_max_date = reachable_cache['max_date']
        cache_min_date = reachable_cache['min_date']
        assert (cache_max_date >= cache_min_date)

        # include relevant entries from cache
        for x in reachable_cache['reachability_sets'].values():
            if (x['date'] <= sample_max_date
                    and x['date'] >= sample_min_date):
                        all_reachable = all_reachable.union(set(x['commits']))

        # determine additional data to be collected
        sample_cids = [cid for cid in sample_cids
                       if cid not in reachable_cache['reachability_sets']
                       and (commits[cid]['date'] > cache_max_date
                            or commits[cid]['date'] < cache_min_date)]
        print '    Revised samples:', len(sample_cids)

    # fetch incremental data
    for cid in sample_cids:
        commit_reachable = set([])
        files = get_all_files_from_commit(cid, repo, filter_config)
        for path in files:
            blame = get_blame(cid, path, repo_name)
            if blame:
                reachable = set([b['commit'] for b in blame])
                commit_reachable = commit_reachable.union(reachable)
        all_reachable = all_reachable.union(commit_reachable)
        # add to cache as well
        reachable_cache['reachability_sets'][cid] = \
            {'date': commits[cid]['date'],
             'commits': list(commit_reachable)}
        print '.',
    print
    print 'Reachable commits:', len(all_reachable)

    # Save updated cache, if needed
    if len(reachable_cache['reachability_sets']) > cache_initial_size:
        print
        print 'Saving updated Reachable Cache'
        reachable_cache['min_date'] = \
            min([r['date']
                for r in reachable_cache['reachability_sets'].values()])
        reachable_cache['max_date'] = \
            max([r['date']
                for r in reachable_cache['reachability_sets'].values()])
        pdump(reachable_cache, project_to_fname(project, reachable=True))

    return list(all_reachable)


def annotate_commit_reachability(project, commits, sampling_freq=25):
    """Updates commit datastructure (typically combined_commits)
    based on commit reachability derived from git blame.  Don't include
    any commits that are obsolete as of leyacy cut-off"""
    legacy_cutoff = find_legacy_cutoff(commits)

    # Initially mark all commits as unreachable
    for c in commits.values():
        c['reachable'] = False

    for cid in identify_reachable_commits(project, commits,
                                          legacy_cutoff, sampling_freq):
        commits[cid]['reachable'] = True
    return
