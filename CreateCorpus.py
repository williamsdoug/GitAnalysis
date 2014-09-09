
# coding: utf-8

# #Master file to create Corpus
# 
# Author:  Doug Williams, Copyright 2014
# 
# Last Updated: 9/9/2014
# 
# Can either rebuild corpus from scratch, or apply updates incrementally.   Since some of the cata is downloaded from the web, there are a set of intermediate corpus that are also maintained.  These are later joined to produce the finl result
# 
# Intermediate Corpus:
# - nova_change_details.jsonz
# - nova_changes.jsonz
# - nova_commits.jsonz
# - nova_lp_bugs.jsonz
# - nova_patch_data.jsonz
# 
# 
# Output Corpus:
# - nova_all_blame.jsonz
# - nova_combined_commits.jsonz

# In[1]:

import pprint as pp
import sys
sys.path.append('./dev')

from jp_load_dump import pdump, pload, jdump, jload

from LPBugsDownload import build_lp_bugs, load_lp_bugs

from GerritDownload import build_gerrit_data
from GerritDownload import load_gerrit_changes, load_gerrit_change_details

from Git_Extract_Join import build_git_commits, load_git_commits
from Git_Extract_Join import build_joined_LP_Gerrit_git, load_combined_commits
from Git_Extract_Join import build_all_blame, load_all_blame


#### Parameters

# In[2]:

PROJECT = 'nova'
REPO_NAME ='/Users/doug/SW_Dev/nova'
CACHEDIR = './cache/'
UPDATE_CORPUS = False


#### Update base data for Corpus

# In[3]:

if UPDATE_CORPUS:
    print 'updating bug data'
    build_lp_bugs(PROJECT, update=True, cachedir=CACHEDIR)
    print 'rebuilding Gerrit data'
    build_gerrit_data(PROJECT, update=True)

    print 'rebuilding Git data'
    build_git_commits(project=PROJECT, repo_name=REPO_NAME)


#### Get Launchpad, Gerrit and Git Data

# In[4]:

print
print 'loading bug data'
downloaded_bugs = load_lp_bugs(PROJECT)
print 'loading change data'    
all_changes = load_gerrit_changes(PROJECT)
print 'all_changes:', len(all_changes)
all_change_details = load_gerrit_change_details(PROJECT)
print 'all_change_details:', len(all_change_details)
print 'loading Git commit data'
commits = load_git_commits(PROJECT)


####### Sanity check data

# In[5]:

total_bugs = 0
total_changes = 0
total_both = 0
for x in commits.values():
    if 'bug' in x:
        total_bugs += 1
        
print 'commits:', len(commits)
print 'bugs:', total_bugs

sample_change = [x for x in commits.values() if 'change_id' in x]
print 'changes:', len(sample_change)
sample_both = [x for x in commits.values() if 'change_id' in x and 'bug' in x]
print 'bug fix changes:', len(sample_both)

#print
#pp.pprint(commits.values()[0:2])


#### Join LP, Gerrit and Git data and generate consolidated corpus

# In[6]:

if UPDATE_CORPUS:
    build_joined_LP_Gerrit_git(PROJECT, commits, downloaded_bugs, all_change_details)


# In[7]:

combined_commits = load_combined_commits(PROJECT)

print len(combined_commits)


#### Compute Diff and Blame

                if UPDATE_CORPUS:
    build_all_blame(PROJECT, combined_commits, update=True, repo_name=REPO_NAME)
                
# In[8]:

build_all_blame(PROJECT, combined_commits, update=False, repo_name=REPO_NAME)


# In[9]:

all_blame = load_all_blame(PROJECT)
print len(all_blame)


# In[10]:

#pp.pprint(all_blame[0])


# In[10]:



