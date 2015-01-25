#
# Git_Extract_Join.py - Code to extract Git commit data, metadata, diff
#                       and blame.
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently configured for OpenStack, tested with Nova.
#
# Last updated 1/24/2014
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
# - 9/3/14:  Filter out huge blame entries (default is >3000 total lines per
#            commit
# - 9/9/14:  Collect all file names
# - 9/10/14: Add exception handler to work-around unknown encoding error for
#            process_commits
# - 9/11/14: Fix committer field in update_commits_with_patch_data
# - 1/23/15: replace depricated os.popen() is subprocess.Popen()
# - 1/24/15: Changes to get_blame() to address fatal errors due to /dev/null
#            files.
# - 1/25/15: PEP-8 clean-up
#
# Issues:
# - None
#
# To Do:
# - get_patch_data(cid) hard-coded to nova, need to generalize for
#   other projects
# - remember huge commits to avoid recomputation during update
#
# Other:
# - Installation: pip install -U gitpython==0.3.2.RC1
# - see fix for blame:
# https://github.com/vitalif/GitPython/commit/104524267b616934983fff89ff112970febac68e
#
# Top Level Routines:
#    from Git_Extract_Join import build_git_commits, load_git_commits
#    from Git_Extract_Join import build_joined_LP_Gerrit_git,
#    from Git_Extract_Join import load_combined_commits
#    from Git_Extract_Join import build_all_blame, load_all_blame
#

from git import *
import git
import time
import pprint as pp
import collections
from collections import defaultdict
# import cPickle as pickle
# import os
import re
# import gzip
# import json
import urllib2

from jp_load_dump import convert_to_builtin_type, pload, pdump, jload, jdump
from subprocess import Popen, PIPE, STDOUT


#
# Globals
#

global commits
commits = collections.defaultdict(list)

global repo
repo = False


#
# Helper
#

def project_to_fname(project, patches=False, combined=False,
                     blame=False, prefix='./Corpus/'):
    if patches:
        return prefix + project + "_patch_data.jsonz"
    elif combined:
        return prefix + project + "_combined_commits.jsonz"
    elif blame:
        return prefix + project + "_all_blame.jsonz"
    else:
        return prefix + project + "_commits.jsonz"


#
# Parse Commit Messages
#

change_template = re.compile('Change-Id:\s*(\S+)', re.IGNORECASE)


def parse_change(txt):
    """Extracts and fixes case in Change-Id """
    txt = txt.lower()
    m = change_template.search(txt)
    if m:
        val = m.group(1)
        if val.startswith('i'):
            val = 'I' + val[1:]
        return {'change_id': str(val)}
    else:
        # print txt
        return {}


bug_template = re.compile('bug[s:\s/-]*(?:lp|lp:|lp:#|lp:)*[\s#]*(\d+)',
                          re.IGNORECASE)


def parse_bug(txt):
    """Extracts bug id """
    txt = txt.lower()
    m = bug_template.search(txt)
    if m:
        return {'bug': m.group(1)}
    else:
        return {}


def parse_msg(msg, patch=False):
    """
    Overall routine for processing commit messages, also used for patch msgs
    """
    result = {}
    if not msg:
        return {}
    for line in msg.split('\n'):
        lline = line.lower()
        # if 'bug' in yy or 'change' in yy:
        if 'change' in lline:
            result.update(parse_change(line))
        elif 'bug' in lline:
            result.update(parse_bug(line))
        if patch:
            if line.startswith('From: '):
                try:
                    # '<git.Actor "'
                    # + obj.name.encode('ascii', 'ignore')
                    # +' <'+ obj.email+'>">'
                    result.update({'pAuth': '<git.Actor "'
                                            + line[len('From: '):]+'">'})
                except Exception:
                    pass
            elif line.startswith('Subject: '):
                try:
                    result.update({'pSummary': line[len('Subject: '):]})
                except Exception:
                    pass

    return result


#
# Basic Commit Processing
#

def process_commits(repo_name="/Users/doug/SW_Dev/nova", max_count=False):
    """Extracts all commit from git repo, subject to max_count limit"""
    total_operations = 0
    total_errors = 0
    global repo
    repo = Repo(repo_name)
    # repo = Repo(repo_name, odbt=GitCmdObjectDB)
    assert repo.bare is False
    global commits
    commits = collections.defaultdict(list)

    for c in repo.iter_commits('master', max_count=max_count):
        cid = c.hexsha
        # commits[cid] = {'author':c.author, 'date': c.committed_date,
        #                 'cid':c.hexsha,
        #                'committer':c.committer, 'msg':c.message,
        #                'files':[], 'parents':[p.hexsha for p in c.parents]}

        try:
            commits[cid] = {'author': convert_to_builtin_type(c.author),
                            'date': c.committed_date,
                            'cid': c.hexsha,
                            'committer': convert_to_builtin_type(c.committer),
                            'msg': c.message.encode('ascii', 'ignore'),
                            'files': process_commit_files(c),
                            'parents': [p.hexsha for p in c.parents]}

            commits[cid].update(parse_msg(c.message))
            total_operations += 1
            if total_operations % 100 == 0:
                print '.',
            if total_operations % 1000 == 0:
                print total_operations,

        except Exception:
            print 'x',
            total_errors += 1
        # print '.',

    if total_errors > 0:
        print
        print 'Commits skipped due to error:', total_errors

    return commits


def process_commit_files(c, filt=['py', 'sh', 'js', 'c', 'go', 'sh', 'conf']):
    """Determine files associated with an individual commit"""

    files = []

    for p in c.parents:    # iterate through each parent
        i = c.diff(p, create_patch=False)

        for d in i.iter_change_type('A'):
            if d.b_blob and d.b_blob.path.split('.')[-1] in filt:
                files.append(d.b_blob.path)

        for d in i.iter_change_type('D'):
            if d.a_blob and d.a_blob.path.split('.')[-1] in filt:
                files.append(d.a_blob.path)

        for d in i.iter_change_type('R'):
            if d.a_blob and d.a_blob.path.split('.')[-1] in filt:
                files.append(d.a_blob.path)
            if d.b_blob and d.b_blob.path.split('.')[-1] in filt:
                files.append(d.b_blob.path)

        for d in i.iter_change_type('M'):
            if d.b_blob and d.b_blob.path.split('.')[-1] in filt:
                files.append(d.b_blob.path)

    return files


hdr_re = re.compile('@@\s+-(?P<nstart>\d+)(,(?P<nlen>\d+))?\s+\+'
                    + '(?P<pstart>\d+)(,(?P<plen>\d+))?\s+@@')


def parse_diff(d, proximity_limit=4):
    """
    Parses individual git diff str, returns range and line proximity to changes
    """
    p_pos = 0
    n_pos = 0
    changes = []
    line_range = []
    for line in d.diff.split('\n'):
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


def build_git_commits(project, repo_name=''):
    """Top level routine to generate commit data """
    commits = process_commits(repo_name=repo_name)
    print
    print 'total commits:', len(commits)
    print 'Augment GIT data with patch info'
    update_commits_with_patch_data(commits, project=project)


def load_git_commits(project='nova'):
    """Top level routine to load commit data, returns dict indexed by cid"""
    name = project_to_fname(project)

    result = jload(name)
    # normalize change_id to begin with upper case I
    # for k, v in result.items():
    #    if 'Change-Id' in v and  v['Change-Id'].startswith('i'):
    #        result[k]['Change-Id'] = 'I' + v['Change-Id'][1:]

    print 'Object type:', type(result)
    print 'total git_commits:', len(result)
    print 'bug fix commits:', len([x for x in result.values() if 'bug' in x])
    print 'commits with change_id:', len([x for x in result.values()
                                          if 'change_id' in x])
    print 'bug fix with change_id:', len([x for x in result.values()
                                          if 'change_id' in x and 'bug' in x])
    return result


#
# Processing of Git Blame
#
# for changing reference point see:
#              http://stackoverflow.com/questions/5098256/git-blame-prior-commits


def get_blame(cid, d, repo_name='/Users/doug/SW_Dev/nova',
              child_cid='', warn=True):
    """Compute per-line blame for individual file for a commit """

    # skip null files
    try:
        if (str(d.a_blob) == git.objects.blob.Blob.NULL_HEX_SHA
                or str(d.b_blob) == git.objects.blob.Blob.NULL_HEX_SHA
                or d.a_blob.size == 0 or d.b_blob.size == 0):
            return False
    except Exception:
        return False

    path = d.b_blob.path
    result = []
    entry = {}
    first = True      # identifies first line in each porcelin group
    cmd = 'cd ' + repo_name
    # cmd += ' ; ' + 'git reset --soft ' + cid
    # cmd += ' ; ' + 'git blame --line-porcelain ' + path
    # cmd += ' ; ' + 'git blame --line-porcelain ' + cid + ' -- ' + path

    cmd += ' ; ' + 'git blame --line-porcelain --follow ' + cid + ' -- ' + path
    # cmd += ' ; ' + 'git blame --line-porcelain -C -C  ' + cid + ' -- ' + path

    # with os.popen(cmd) as f:
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
                    if d.b_blob.path != d.a_blob.path:
                        print 'b:', d.b_blob.path
                        print 'a:', d.a_blob.path
                return False

            # line = line.encode('utf-8', errors='ignore')
            # print line
            if first:
                line = line[:-1]    # strip newline
                v = line.split(' ')
                entry['commit'] = v[0]
                entry['orig_lineno'] = v[1]
                entry['lineno'] = v[2]
                # print 'first: ', v
                first = False
            elif line[0] == '\t':
                # print entry['lineno']
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
        # print
        print 'Warning -- Empty Blame for ', cid, path
        # with os.popen(cmd) as f:
        #    for line in f:
        #        print line
        return False
    return result


def assign_blame(d, p_cid, repo_name='', child_cid=''):
    """Combine diff with blame for a file"""
    try:
        result = parse_diff(d)
        # now annotate with blame information

        # blame = dict([(int(x['lineno']), x)
        #              for x in get_blame(p_cid, d.b_blob.path,
        #                                 repo_name=repo_name)])

        blame = dict([(int(x['lineno']), x)
                      for x in get_blame(p_cid, d, repo_name=repo_name,
                                         child_cid=child_cid)])
        result = [dict(x.items() + [['commit', blame[x['lineno']]['commit']]])
                  for x in result if x['lineno'] in blame]

        return [d.b_blob.path, result]

    except Exception:
        return [d.b_blob.path, False]


def process_commit_details(cid, repo_name='',
                           filt=['py', 'sh', 'js', 'c', 'go', 'sh', 'conf']):
    """Process individual commit, computing diff and identifying blame.
       Exclude non-source files
    """
    global repo

    c = repo.commit(cid)
    blame = []

    for p in c.parents:    # iterate through each parent
        i = c.diff(p, create_patch=True)

        for d in i.iter_change_type('M'):
            if d.b_blob and d.b_blob.path.split('.')[-1] in filt:
                blame.append(assign_blame(d, p.hexsha,
                                          repo_name=repo_name, child_cid=cid))

        # blame += [assign_blame(d,p.hexsha,repo_name=repo_name)
        #           for d in i.iter_change_type('M') if d.b_blob]

    return dict(blame)


def compute_all_blame(bug_fix_commits, start=0, limit=1000000, repo_name='',
                      keep=set(['lineno', 'orig_lineno', 'commit', 'text'])):
    """Top level iterator for computing diff & blame for a list of commits"""
    progress = 0
    all_blame = []

    # code for debug, remove later
    if repo_name:
        global repo
        repo = Repo(repo_name)

    for cid in bug_fix_commits[start:start+limit]:
        all_blame.append(
            {'cid': cid,
             'blame': process_commit_details(cid, repo_name=repo_name)})

        progress += 1
        # if progress % 100 == 0:
        if progress % 10 == 0:
            print '.',
        if progress % 1000 == 0:
            print progress

    return all_blame


def filter_huge_blame(entry, threshold=3000):
    total = 0
    for fdat in entry['blame'].values():
        if fdat:
            x = len(fdat)
            total += x
    return total < threshold


def build_all_blame(project, combined_commits, update=True, repo_name=''):
    """Top level routine to generate or update blame data"""
    global repo
    repo = Repo(repo_name)

    bug_fix_commits = set([k for k, v in combined_commits.items()
                           if 'lp:id' in v])
    print 'bug fix commits:', len(bug_fix_commits)

    if update:
        known_blame = set([x['cid'] for x in load_all_blame(project)])
        new_blame = bug_fix_commits.difference(known_blame)
        print 'new blame to be computed:', len(new_blame)

        if len(new_blame) > 0:
            new_blame = list(new_blame)
            print
            all_blame = compute_all_blame(new_blame, repo_name=repo_name)
            # prune huge entries
            all_blame = [x for x in all_blame if filter_huge_blame(x)]
            print 'saving'
            jdump(load_all_blame(project) + all_blame,
                  project_to_fname(project, blame=True))

    else:
        all_blame = compute_all_blame(list(bug_fix_commits),
                                      repo_name=repo_name)
        # prune huge entries
        all_blame = [x for x in all_blame if filter_huge_blame(x)]
        print 'saving'
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
            if 'jenkins@review.openstack.org' in c['author']]


def get_patch_data(cid):
    """Downloads supplementary patch data from OpenStack cgit """
    base = 'http://git.openstack.org/cgit/openstack/nova/patch/?id='
    try:
        f = urllib2.urlopen(urllib2.Request(base+cid))
        result = f.read()
        f.close()
        return result.split('diff --git')[0]
    except Exception:
        return False


def load_patch_data(jenkins_commits, project='nova', incremental=True):
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
        patch_data.append({'cid': cid, 'patch': get_patch_data(cid)})

        count += 1
        if count % 10 == 0:
            print '.',
        if count % 100 == 0:
            print count,
        if count % 1000 == 0:
            pdump(patch_data, name)

    jdump(patch_data, name)
    return patch_data


def update_commits_with_patch_data(commits, project='nova'):
    """Highest level routine - identifies all commits with jenkins author and
       augments metadata with cgit patch data
    """
    name = project_to_fname(project, patches=True)
    # fetch appropriate data from patches
    jenkins_commits = identify_jenkins_commit(commits)
    patch_commits = dict([(x['cid'], parse_msg(x['patch'], patch=True))
                          for x in load_patch_data(jenkins_commits,
                                                   project=project,
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

    jdump(commits, project_to_fname(project))


#
# Join Code - produces combined_commits
#             joins LP Bugs & Gerrit with Commit data
#

def build_joined_LP_Gerrit_git(project, commits, downloaded_bugs,
                               all_change_details):
    """Top level routine to compute combined_commits by joining bugs & Gerrit
       data with commit entries.
    """
    results = {}

    total = 0

    # create indices
    idx_cid_to_bugs = dict([[v['cid'], k]
                            for k, v in downloaded_bugs.items()
                            if 'cid' in v])
    # print len(idxcid_to_bugs)
    idx_changeid_to_bugs = dict([[v['change_id'], k]
                                 for k, v in downloaded_bugs.items()
                                 if 'change_id' in v])
    # print len(idx_changeid_to_bugs)

    # convert gerrit data into dict indexed by change_id
    d_gerrit_by_changeid = dict([[v['change_id'], v]
                                 for v in all_change_details
                                 if 'change_id' in v])
    print len(d_gerrit_by_changeid)

    for cid, commit in commits.items():
        lp_val = []
        gerrit_val = []
        git_val = [[k, v] for k, v in commit.items()]

        # join with LP data
        bugno = False
        if cid in idx_cid_to_bugs:
            bugno = idx_cid_to_bugs[cid]
        elif 'bug' in commit and commit['bug'] in downloaded_bugs:
            bugno = commit['bug']
        elif ('change_id' in commit
                and commit['change_id'] in idx_changeid_to_bugs):
            bugno = idx_changeid_to_bugs[commit['change_id']]

        if bugno:
            lp_val = [['lp:'+k, v] for k, v in downloaded_bugs[bugno].items()]

        # join gerrit data using change_id
        if ('change_id' in commit
                and commit['change_id'] in d_gerrit_by_changeid):
            gerrit_val = [['g:'+k, v]
                          for k, v
                          in d_gerrit_by_changeid[commit['change_id']].items()]

        results[cid] = dict(lp_val+gerrit_val+git_val)

        total += 1
        if total % 100 == 0:
            print '.',
        if total % 1000 == 0:
            print total,

    jdump(results, project_to_fname(project, combined=True))


def load_combined_commits(project):
    """Loads combined_commit data from disk"""
    return jload(project_to_fname(project, combined=True))
