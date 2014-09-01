#
# GerritDownloader.py - Code to extract Gerrit review data for an OpenStack Project
#
# Author:  Doug Williams - Copyright 2014 
# 
# Currently configured for OpenStack, tested with Nova.
# 
# Last updated 9/1/2014
# 
# History:
# 0. 9/1/14: Converted from iPython notebook, added callable interfaces
#
# Issues:
# -  None
# 
# To Do:
# -  None
#
# Top Level Routines:
#    from GerritDownload import build_gerrit_data
#    from GerritDownload import load_gerrit_changes, load_gerrit_change_details
#

import pprint as pp
import pygerrit
from pygerrit.rest import GerritRestAPI
import time

from jp_load_dump import pdump, pload, jdump, jload

#
# Background Information:
#
#     source: https://pypi.python.org/pypi/pygerrit/0.2.1
#     and https://gerrit-review.googlesource.com/Documentation/rest-api.html
#
#     sample gerrit query: http://review.gluster.org/#/q/project:glusterfs+branch:master+topic:bug-1100144,n,z
#
#     Documentation for v 2.8 REST API: https://gerrit-documentation.storage.googleapis.com/Documentation/2.8/rest-api-changes.html#list-changes

#
# Globals
#
rest = GerritRestAPI(url='http://review.openstack.org')

#
# Code
#

def project_to_fname(project, details=False):
    """Helper function - converts project name to standardized file names """
    if details:
        return project + "_change_details.jsonz"
    else:
        return project + "_changes.jsonz"

def get_version():
    """Gets REST API version - not currently used """
    global rest
    proj = rest.get("/config/server/version")


def get_projects(filter='openstack/'):
    """Gets list of projects from Gerrit - not currently used """
    global rest
    proj = rest.get("/projects/?")
    if filter:
        return [x for x in proj.keys() if filter in x]
    else:
        return proj.keys()


def get_all_changes(project="nova", prefix='openstack/', limit=1000000000):
    """Gets list of changes for a project """
    global rest
    result = []
    total = 0
    sortkey = ''
    count = 1
    query = "/changes/?q=project:{" + prefix + project + "}+is:merged"
    
    while count > 0 and total < limit:
        if sortkey:
            suffix = '+resume_sortkey:'+sortkey
        else:
            suffix = ''
            
        #print query+suffix
        changes = rest.get(query+suffix)
        
        count = len(changes)
        if count > 0:
            result += changes
            sortkey = changes[-1]['_sortkey']
        total += count
        #print 'returned:', len(changes), 'total:', total
        print '.',
    
    print
    print 'total changes:', total
    return result



def get_change(changeno):
    """Gets summary change data """
    global rest
    query = "/changes/" + str(changeno)
    return rest.get(query)


def compress_vote_set(d):
    result = {}
    if 'all' in d:
        result['all'] = [{'_account_id': z['_account_id'],'name':z['name'],'value':z['value']} for z in d['all'] if z['value'] != 0]
    if 'approved' in d:
        result['approved'] = {'_account_id': d['approved']['_account_id'],'name':d['approved']['name']} 
    if 'recommended' in d:
        result['recommended'] = d['recommended']
    if 'value' in d:
        result['value'] = d['value']
    return result


def compress_votes(d):
    """Extracts voting data """
    result = {}
    if 'Workflow' in d:
            result['Workflow'] = compress_vote_set(d['Workflow'])
    if 'Verified' in d:
            result['Verified'] = compress_vote_set(d['Verified'])
    if 'Code-Review' in d:
            result['Code-Review'] = compress_vote_set(d['Code-Review'])
        
    return result


def compress_messages(messages):
    """Gets only messages for most recent change """
    standard =     [msg for msg in messages if '_revision_number' in msg]
    non_standard = [msg for msg in messages if '_revision_number' not in msg]
    return [msg for msg in standard
            if msg['_revision_number'] == standard[-1]['_revision_number']] + non_standard 


def get_change_detail(changeno, _prune=True):
    """Get full detail for a change """
    global rest
    query = "/changes/" + str(changeno) + '/detail'
    x = rest.get(query)
    
    if _prune:
        x = prune(x)

    return x 


def prune(x):
    """Trim extraneous data from Gerrit history """
    #print x['_number']
    if '_sortkey' in x:
        del x['_sortkey'] 
    if 'id' in x:   
        del x['id'] 
    if 'kind' in x:   
        del x['kind']    
    if 'removable_reviewers' in x:
        del x['removable_reviewers']  
    if 'permitted_labels' in x: 
        del x['permitted_labels'] 
        
    owner = {}
    if '_account_id' in x['owner']:
        owner['_account_id'] = x['owner']['_account_id']
    if 'name' in x['owner']:
        owner['name'] = x['owner']['name']
    x['owner']  = owner
    
    x['labels'] = compress_votes(x['labels'])
    
    try:
        x['messages'] = compress_messages(x['messages'])
    except Exception, e:
        print e
        print  x['_number']
        print type(x['messages'])
        pp.pprint(x['messages'])
        raise Exception
    
    return x 


def load_gerrit_changes(project='nova'):
    """Top level routine to load summary change data from disk"""
    name = project_to_fname(project)
    x = jload(name)
    print 'Object type:', type(x)
    print 'total gerrit changes:', len(x)
    return x
    
def load_gerrit_change_details(project='nova'):
    """Top level routine to load detailed change data from disk"""
    name = project_to_fname(project, details=True)
    x = jload(name)
    print 'Object type:', type(x)
    print 'total gerrit changes with detail:', len(x)
    return x


def build_all_changes(project='nova', update=False):
    """Top level routine to download summary change data """
    all_changes = get_all_changes(project='nova')
    print len(all_changes)
    name = project_to_fname(project)
    jdump(all_changes, name)
    return all_changes
    

def build_all_change_details(project='nova', update=False):
    """Top level routine to download detailed change data.
        Supports incremental updates.
    """
    if update:
        all_changes = load_gerrit_changes(project)
        print 'all_changes:', len(all_changes)
        all_change_details = load_gerrit_change_details(project)
        print 'all_change_details:', len(all_change_details)
        found = dict([[x['change_id'], 1] for x in all_change_details])
        missing = [str(x['_number']) for x in all_changes if x['change_id'] not in found]
        
        print 'Missing:',len(missing)
        all_change_details_plus = []
        i = 0
        skipped = 0
        for changeno in missing:
            try:
                #time.sleep(0.03)
                all_change_details_plus.append(get_change_detail(changeno, _prune=True))
            except Exception, e:
                skipped += 1
            if i % 100 == 0:
                print '.',
            i += 1
        print 'Downloaded:', len(all_change_details_plus)
        print 'Skipped:', skipped

        all_change_details = load_gerrit_change_details(project)
        all_change_details = all_change_details + all_change_details_plus
        name = project_to_fname(project, details=True)
        jdump(all_change_details, name)
        return all_change_details
    else:
        all_changes = load_gerrit_changes('nova')
        all_change_details = []
        for changeno in [x['change_id'] for x in all_changes]:
            try:
                all_change_details.append(get_change_detail(changeno))
            except Exception, e:
                pass
        print len(all_change_details)
        name = project_to_fname(project, details=True)
        jdump(all_change_details, name)
        return all_change_details

    
def build_gerrit_data(project='nova', update=True):
    """Top level routine for downloading or updating both summary change data
        and detailed change data.
    """
    build_all_changes(project, update=update)
    build_all_change_details(project, update=update)
