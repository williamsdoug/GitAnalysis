Code for GitAnalysis
====================

Major files:
- Git_Extract.py - code related to parsing and processing of Git data
- LPBugsDownload.py - code related to Launchpad Bug Tracker
- GerritDownload.py - code related to Gerrit code review system
- commit_analysis.py - Code to join git, lp and gerrit data
- BugFixWorkflow.py - code related to guilt computation and feature extraction
- jp_load_dump.py - low level routines for json and pickle data

Configuration File:
- git_analysis_config.py

Notes:
- Code still mostly hard-coded to OpenStack tools chain Git+LaunchPad+Gerrit.  Needs t ve extended for other development toolchains as well.  