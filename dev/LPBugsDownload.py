#
# LPBugsDownload.py - Code to download a set of bugs from Launchpad
#                     for a given project
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Currently configured for OpenStack, tested with Nova.
#
# Last updated 2/19/2015
#
# History:
# - 9/1/2014:  Converted from iPython notebook, added callable interfaces
# - 9/10/2014: Fixed debug comments
# - 1/25/2015: PEP-8 clean-up
# - 2/5/2015:  removed references to project='nova'
# - 2/5/2015:  created annotate_bug_status().  Only download fixed bugs
# - 2/18/2015: Change lp_parse_messages() to parse multiple git cid and
#              Change-Id sets.  Values now stored in ['commits'] of
#              bug entry.
# - 2/19/2015: Extended lp_parse_messages to identify commits prefixed by
#              https://git.openstack.org/cgit/openstack/glance/commit/?id=
#
# Top Level Routines:
#    from LPBugsDownload import build_lp_bugs, load_lp_bugs
#

#
# Includes
#

from launchpadlib.launchpad import Launchpad
import pprint as pp
import httplib2
import os

from jp_load_dump import pdump, pload, jdump, jload

#
# Globals
#

global lp_excludes
lp_excludes = {'https://api.launchpad.net/1.0/#bug':
               ['web_link', 'resource_type_link', 'http_etag',
                'can_expire', 'date_last_message', 'date_last_updated',
                'date_made_private', 'latest_patch_uploaded',
                'number_of_duplicates',
                'other_users_affected_count_with_dupes',
                'private', 'users_affected_count',
                'users_affected_count_with_dupes', 'users_unaffected_count',
                'bug_watches', 'duplicates', 'linked_branches',
                'subscriptions', 'users_affected', 'users_affected_with_dupes',
                'users_unaffected', 'who_made_private'],

               'https://api.launchpad.net/1.0/#bug_task':
               ['web_link', 'resource_type_link', 'http_etag', 'date_closed',
                'date_confirmed', 'date_created', 'date_fix_committed',
                'date_fix_released', 'date_in_progress', 'date_incomplete',
                'date_left_closed', 'date_left_new', 'date_triaged',
                'related_tasks', 'bug_watch', 'milestone'],

               'https://api.launchpad.net/1.0/#bugs': ['resource_type_link',
                                                       'next', 'prev'],

               'https://api.launchpad.net/1.0/#message':
               ['web_link', 'resource_type_link', 'http_etag',
                'owner', 'parent'],

               'https://api.launchpad.net/1.0/#project':
               ['web_link', 'resource_type_link', 'http_etag',
                'bug_reported_acknowledgement', 'bug_reporting_guidelines',
                'commercial_subscription_is_due',
                'date_next_suggest_packaging',
                'download_url', 'freshmeat_project', 'is_permitted',
                'license_info', 'licenses', 'private', 'private_bugs',
                'programming_language', 'project_reviewed',
                'qualifies_for_free_hosting', 'remote_product',
                'reviewer_whiteboard', 'screenshots_url', 'security_contact',
                'sourceforge_project', 'wiki_url', 'active_milestones',
                'all_milestones', 'recipes', 'series', 'brand',
                'bug_supervisor', 'bug_tracker', 'commercial_subscription',
                'development_focus', 'driver', 'icon', 'logo', 'owner',
                'registrant', 'translation_focus'],

               'https://api.launchpad.net/1.0/#project-page-resource':
               ['resource_type_link', 'total_size', 'start',
                'entries', 'next', 'prev', 'entry_'],

               'https://api.launchpad.net/1.0/#project_group':
               ['web_link', 'resource_type_link', 'http_etag',
                'bug_reported_acknowledgement', 'bug_reporting_guidelines',
                'freshmeat_project', 'homepage_content', 'homepage_url',
                'official_bug_tags', 'reviewed', 'sourceforge_project',
                'wiki_url', 'active_milestones', 'all_milestones',
                'bug_tracker', 'driver', 'icon', 'logo', 'mugshot',
                'owner', 'registrant']
               }

#
# Code
#


def project_to_fname(project, prefix='./Corpus/'):
    """Helper routine to create standard filename from project name"""
    return prefix+project + "_lp_bugs.jsonz"


def lp_fetch_object(obj,
                    exclude=[],
                    include=['lp_attributes', 'lp_entries', 'lp_collections'],
                    debug=False):
    """Downloads individual entry from Launchpad """
    global lp_excludes
    if not exclude:
        resource_type_link = obj.resource_type_link
        if resource_type_link in lp_excludes:
            # print 'found'
            exclude = lp_excludes[resource_type_link]
        elif debug:
            print 'Exclusion list not found for:', resource_type_link
    result = {}
    result['resource_type'] = obj.resource_type_link.split('/')[-1]

    if 'lp_attributes' in include:
        for name in obj.lp_attributes:
            if name not in exclude:
                # print 'Fetching', name
                try:
                    if name.startswith("date_"):
                        result[name] = str(getattr(obj, name))
                    else:
                        result[name] = getattr(obj, name)
                except Exception, e:
                    pass
            elif debug:
                print 'Skipping', name
                pass

    if 'lp_entries' in include:
        for name in obj.lp_entries:
            if name not in exclude:
                try:
                    if name.startswith("date_"):
                        result[name] = str(getattr(obj, name))
                    else:
                        result[name] = getattr(obj, name)
                except Exception, e:
                    pass
            elif debug:
                print 'Skipping', name
                pass

    if 'lp_collections' in include:
        for name in obj.lp_collections:
            if name not in exclude:
                try:
                    result[name] = [x for x in getattr(obj, name)]
                except Exception, e:
                    pass
            elif debug:
                print 'Skipping', name
                pass

    return result


def fetch_bug(bugno):
    """Adds bug entry to ALL_BUGS unless already present """
    global ALL_BUGS
    if bugno in ALL_BUGS:
        return
    bug = lp_fetch_object(lp.bugs[bugno])
    # now fetch associated messages
    bug['messages'] = [lp_fetch_object(msg) for msg in bug['messages']]

    ALL_BUGS[bugno] = bug


def fetch_unique_bugno(project_name):
    """Generated list of bug id numbers from Launchpad """
    unique_bugs = {}
    project = lp.distributions[project_name]
    bug_tasks = project.searchTasks(status=['Fix Committed', 'Fix Released'])
    # bug_tasks = project.searchTasks(status=['Confirmed', 'In Progress',
    #                                         'Fix Committed', 'Fix Released'])
    for b in bug_tasks:
        bugno = str(b).split('/')[-1]
        unique_bugs[bugno] = 1
    return unique_bugs


def fetch_all_bugs(project_name, limit=-1, prior={}):
    """Downloads data for any newly posted bugs """
    global ALL_BUGS
    ALL_BUGS = {}

    unique_bugs = fetch_unique_bugno(project_name)
    print 'total bugs:', len(unique_bugs)
    new_bugs = [k for k in unique_bugs.keys() if k not in prior]
    print 'bugs to be downloaded:', len(new_bugs)

    for bugno in new_bugs:
        print '.',
        fetch_bug(bugno)

        limit -= 1
        if limit == 0:
            return


def pickle_clean(foo):
    """Helper routine used by build_lp_bugs"""
    if type(foo) == list:
        return [pickle_clean(i) for i in foo]
    elif type(foo) == dict:
        result = {}
        for k, v in foo.items():
            result[k] = pickle_clean(v)
        return result
    elif type(foo) == set:
        return set([pickle_clean(i) for i in foo])
    elif type(foo) == unicode:
        return foo
    else:
        return str(foo)


def lp_parse_messages(messages):
    """Extracts commit metadata from bug messages """
    all_data = []

    for msg in messages:
        result = {}
        txt = msg['content']

        for line in txt.split('\n'):
            if 'Change-Id: I' in line:
                change_id = line.split(':')[-1].strip()
                result['change_id'] = change_id
            elif 'Committed: http://github.com/openstack/' in line:
                commit = line.split('/')[-1]
                result['cid'] = commit
            elif ('https://git.openstack.org/cgit/openstack/glance/commit/?id='
                  in line):
                commit = line.split('=')[-1]
                result['cid'] = commit

        if 'change_id' in result and 'cid' in result:
            all_data.append(result)
    return all_data


def annotate_bug_status(bugs, project):
    global lp
    lpproject = lp.distributions[project]
    print 'annotating bug entries ...'
    for bug in bugs.values():
        bug['importance'] = False
        bug['status'] = False

    importance = ['Critical', 'High', 'Medium', 'Low', 'Wishlist',
                  'Unknown', 'Undecided']

    status = ['Fix Committed', 'Fix Released',
              # 'New', 'Incomplete', 'Opinion', 'Invalid',
              # "Won't Fix", 'Expired', 'Confirmed', 'Triaged',
              # 'In Progress', 'Incomplete',
              ]

    for field in importance:
        print '     ', field,
        count = 0
        count2 = 0
        for bugno in [str(b).split('/')[-1]
                      for b in lpproject.searchTasks(importance=field,
                                                     status=status)]:
            if bugno in bugs:
                bugs[bugno]['importance'] = field
                count += 1
            count2 += 1
        print count, '/', count2

    print
    for field in status:
        print '     ', field,
        count = 0
        count2 = 0
        for bugno in [str(b).split('/')[-1]
                      for b in lpproject.searchTasks(status=field)]:
            if bugno in bugs:
                bugs[bugno]['status'] = field
                count += 1
            count2 += 1
        print count, '/', count2


#
# Top Level Routines
#

def build_lp_bugs(project, update=True, limit=-1, cachedir=''):
    """Top level routine to download bug related data from Launchpad"""
    global ALL_BUGS
    ALL_BUGS = {}
    global lp
    lp = Launchpad.login_anonymously('just testing', 'production', cachedir)

    pname = project_to_fname(project)
    prior = {}
    # see if prior version exists
    if update:
        try:
            prior = jload(pname)
            os.rename(pname, pname+'.old')
            print 'Total prior bugs:', len(prior)
        except Exception, e:
            prior = {}

    # fetch incremental bugs
    fetch_all_bugs(project, limit=limit, prior=prior)

    # clean-up for pickling
    for k in ALL_BUGS.keys():
        del ALL_BUGS[k]['bug_tasks']
        del ALL_BUGS[k]['owner']

    ALL_BUGS = pickle_clean(ALL_BUGS)

    print 'New bugs downloaded:', len(ALL_BUGS)

    print 'Prior bugs downloaded:', len(prior)

    # merge prior results
    for k, v in prior.items():
        ALL_BUGS[k] = v
    print 'Final bugs:', len(ALL_BUGS)

    annotate_bug_status(ALL_BUGS, project)

    # save
    jdump(ALL_BUGS, pname)


def load_lp_bugs(project):
    """
    Top level routine to load bug info from disk, includes port processing
    """
    pname = project_to_fname(project)
    x = jload(pname)

    # now annotate bug entry with additional content, if available
    found = 0
    for k, bug in x.items():
        x[k]['commits'] = lp_parse_messages(bug['messages'])
        if len(x[k]['commits']) > 0:
            found += 1

    # print 'Object type:', type(x)
    print 'total LP bugs:', len(x)
    print 'Entries annotated:', found

    return x
