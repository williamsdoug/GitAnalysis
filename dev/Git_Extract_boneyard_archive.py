#
# Git_Extract_boneyard_archive.py - Obsolete code from Git_Extract
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently being tested using OpenStack (Nova, Swift, Glance, Cinder, Heat)
#
# Last updated 3/5/2015
#


#
#
# Boneyard
#
#

def _consolidate_merge_commits(commits, master_commit,
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

    # Reset branch-related values
    for v in commits.values():
        v['tombstone'] = False
        v['merge_commit'] = False

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


def _aggregate_merge_bugs_and_changes(commits):
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


def _combine_merge_commit(c, commits):
    """Promotes relevant information from second parent into
    merge commit
    """

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

def _load_all_blame(project):
    """Top level routine to load blame data"""
    return jload(project_to_fname(project, blame=True))


def _build_all_blame(project, combined_commits, update=True,
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


def _compute_all_blame(bug_fix_commits, repo, repo_name, filter_config,
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


def _prune_huge_blame(entry, threshold=3000):
    total = 0
    for fdat in entry['blame'].values():
        if fdat:
            x = len(fdat)
            total += x
    if total >= threshold:
        entry['blame'] = {}


def _get_patch_data(cid, project):
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


def _load_patch_data(jenkins_commits, project, incremental=True):
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


def _update_commits_with_patch_data(commits, project):
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
