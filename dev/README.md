Code for GitAnalysis
====================

Major files:
- Git_Extract.py - code related to parsing and processing of Git data
- LPBugsDownload.py - code related to Launchpad Bug Tracker
- GerritDownload.py - code related to Gerrit code review system
- commit_analysis.py - Code to join git, lp and gerrit data
- BugFixWorkflow.py - code related to guilt computation and feature extraction
- jp_load_dump.py - low level routines for json and pickle data
- PythonDiff.py - Language aware diff function
- python_introspection.py - Python complexity metrics

Configuration File:
- git_analysis_config.py

Other files:
- ml_plot.py - helper routines for analytics spreadsheets

Notes:
- Code still mostly hard-coded to OpenStack toolchain Git+LaunchPad+Gerrit.  Needs to be extended for other development toolchains as well.  
