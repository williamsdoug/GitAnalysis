
# coding: utf-8

# #Corpus of git commit related data for OpenStack Nova
# 
# Copyright Doug Williams - 2014
# 
# ###Updated: 9/2/2014
# 
# ###Sources:
# - Launchpad (https://launchpad.net/nova)
# - Gerrit (https://review.openstack.org/#/q/status:open,n,z)
# - Git (https://github.com/openstack/nova)
# 
# ###Structure of Corpus:
# - combined_commits.jsonz : Basic record for each commit, integrating data from Launchpad, Gerrit and Git
# - all_blame.jsonz :  Applies GIT BLAME to the parent commit of every bug fix related commit.  Only includes those lines included within the window for diff.  Proximity is distance of line within blame for a changed line in a bug fix commit, where proximity=1 for changed lines or lines adjacent to inserts
# 
# Note:  All .jsonz files are gzipped json, with individual json entries for each record.  Apply json.loads() to each individual line.
# 
# ###Examples:
# - Sample entry from combined_commits.jsonz
# - Sample entry from all_blame.jsonz
# - Example of blame normalization
# 
# ###History
# - 9/2/2014:  Updated corpus, used additional patch information from Jenkins to annotate commits with author jenkins@review.openstack.org.  Updated normalize_blame to handle null entries
# - 9/2/2014: split nova_all_blame.jsonz into nova_all_blame_1.jsonz and nova_all_blame_2.jsonz to avoid gtithub size limit

## Code to load corpus

# In[1]:

import gzip
import json
import pprint as pp


# In[2]:

def jload_helper(f):
    #restore the object
    result = []
    dict_result = {}
    for line in f:
            v = json.loads(line)
            if 'json_key' in v:
                #individual entries are items in a dict:
                k = v['json_key']
                del v['json_key']
                dict_result[k] = v
                pass
            else:
                result.append(v)
    if dict_result:
        print 'returning dict'
        return dict_result
    elif len(result) > 1:
        print 'returning list'
        return result
    elif len(result) == 1:
        print 'returning singleton'
        return result[0]
    else:
        return False


# In[3]:

def jload(name):
    if name.endswith('z'):
        print 'gzipped'
        with gzip.open(name,'rb') as f:
            return jload_helper(f)
    else:
        with open(name,'rb') as f:
            return jload_helper(f)


## Sample entry from nova_combined_commits.jsonz

# In[4]:

combined_commits = jload('nova_combined_commits.jsonz')


####### Basic commit entry (without corresponsing Launchpad info)

# In[5]:

pp.pprint(combined_commits.items()[0])


# Entry associated with bug fix.  'g:' prefix is data from gerrit, 'lp:' prefix is data from launchpad.  All other fields from git commit

# In[6]:

bug_fix_commits = [k for k,v in combined_commits.items() if 'lp:id' in v]
print 'commits associated with big fixes:', len(bug_fix_commits)
print bug_fix_commits[0]
print
pp.pprint (combined_commits[bug_fix_commits[0]] )


## Sample entry from all_blame.jsonz

# In[7]:

all_blame = jload('nova_all_blame_1.jsonz') + jload('nova_all_blame_2.jsonz')


# In[18]:

pp.pprint(all_blame[2])


# #Normalize Blame Entry
# 
# Weight commits based on proximity to change.  Normalize to that overall weight is 0.  Normalization can either be reciprical of distance or exponential

# In[9]:

from collections import defaultdict


# In[10]:

def normalize_blame(blameset, exp_weighting=True):
    """returns list of commits with weighted blame"""
    result = defaultdict(float)
    total = 0
    for per_file in blameset['blame'].values():
        if per_file:       #validate not null entry
            for per_line in per_file:
                total += 1
                #print per_line
                if exp_weighting:
                    result[per_line['commit']] += 1.0/(2.0**(per_line['proximity']-1))
                else:
                    result[per_line['commit']] += 1.0/float(per_line['proximity'])
                
    return dict([[k, v/total] for k, v in result.items()]) 


# In[16]:

normalize_blame(all_blame[2], exp_weighting=True)


# In[17]:

normalize_blame(all_blame[2], exp_weighting=False)


# In[ ]:



