
# coding: utf-8

# #Corpus of git commit related data for OpenStack Nova
# 
# Copyright Doug Williams - 2014
# 
# ###Updated: 9/3/2014
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
# - 9/3/2014:  Update nova_all_blame, filtering out all entries > 3000 diff lines that were contributing to huge file size
# - 9/3/2014:  Corrected normalize_blame_by_commit computation, added normalize_blame_by_file

## Code

# In[1]:

from pprint import pprint
from collections import defaultdict
import sys
sys.path.append('./dev')
from jp_load_dump import jload


# In[2]:

def normalize_blame_by_commit(blameset, exp_weighting=True):
    """returns list of commits with weighted blame based on proximity to changes"""
    result = defaultdict(float)
    total = 0
    for per_file in blameset['blame'].values():
        if per_file:       #validate not null entry
            for per_line in per_file:
                if exp_weighting:
                    weight = 1.0/(2.0**(per_line['proximity']-1))
                else:
                    weight = 1.0/float(per_line['proximity'])
                result[per_line['commit']] += weight
                total += weight
                
    return dict([[k, v/total] for k, v in result.items()]) 


# In[3]:

def normalize_blame_by_file(blameset, exp_weighting=True):
    """returns list of files with weighted blame"""
    result = defaultdict(float)
    total = 0.0
    for fname, per_file in blameset['blame'].items():
        if per_file:       #validate not null entry
            weight = 0.0
            for per_line in per_file:
                if exp_weighting:
                    weight +=  1.0/(2.0**(per_line['proximity']-1))
                else:
                    weight += 1.0/float(per_line['proximity'])
            result[fname] = weight
            total += weight
                
    return dict([[k, v/total] for k, v in result.items()]) 


## Sample entry from nova_combined_commits.jsonz

# In[4]:

combined_commits = jload('corpus/nova_combined_commits.jsonz')


####### Basic commit entry (without corresponsing Launchpad info)

# In[5]:

pprint(combined_commits.items()[0])


# Entry associated with bug fix.  'g:' prefix is data from gerrit, 'lp:' prefix is data from launchpad.  All other fields from git commit

# In[6]:

bug_fix_commits = [k for k,v in combined_commits.items() if 'lp:id' in v]
print 'commits associated with big fixes:', len(bug_fix_commits)
print bug_fix_commits[0]
print
pprint (combined_commits[bug_fix_commits[0]] )


## Sample entry from all_blame.jsonz

# In[7]:

all_blame = jload('corpus/nova_all_blame.jsonz')


# In[8]:

pprint(all_blame[1])


# #Normalize Blame Entry

# In[9]:

normalize_blame_by_commit(all_blame[1], exp_weighting=True)


# In[10]:

normalize_blame_by_commit(all_blame[1], exp_weighting=False)


# In[11]:

normalize_blame_by_file(all_blame[1], exp_weighting=True)


# In[12]:

normalize_blame_by_file(all_blame[1], exp_weighting=False)

