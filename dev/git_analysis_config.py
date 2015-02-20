#
# git_analysis_config.py - Consolidates all configuration-specific information
#
# Author:  Doug Williams - Copyright 2015
#
# Currently configured for OpenStack, tested with Nova and Swift.
#
# Last updated 2/20/2014
#
# History:
# - 2/20/15: Initial version.  Includes configuration data for file filtering,
#            git repo, lp cache and corpus.  New routines get_filter_config(),
#            get_corpus_dir(), get_repo_dir(), get_lpcache_dir()
#
# Top Level Routines:
#    from git_analysis_config import get_filter_config
#    from git_analysis_config import get_corpus_dir
#    from git_analysis_config import get_repo_name
#    from git_analysis_config import get_lpcache_dir
#

DEFAULT_REPO_PATH = '/Users/doug/SW_Dev/'
DEFAULT_WORKING_DIRECTORY = '/Users/doug/iPython/git_analysis2'
DEFAULT_CORPUS_PATH = DEFAULT_WORKING_DIRECTORY + '/Corpus/'
DEFAULT_LPCACHE_PATH = DEFAULT_WORKING_DIRECTORY + '/cache/'

DEFAULT_FILTER_INCLUDE_FILE_SUFFIX = ['.py', '.sh', '.js', '.c', '.go', '.sh']
DEFAULT_FILTER_EXCLUDE_FILE_PREFIX = ['tools/']

['' + '/tests/', 'tools/']

PROJECT_CONFIG = {
    'nova': {'git_repo': DEFAULT_REPO_PATH + 'nova',
             'lp_cache': DEFAULT_LPCACHE_PATH + 'nova' + '/',
             'corpus_location': DEFAULT_CORPUS_PATH,
             'file_include_suffix': DEFAULT_FILTER_INCLUDE_FILE_SUFFIX,
             'file_exclude_prefix': ['nova/tests/', 'tools/'],
             'toolchain': ['launchpad', 'gerrit', 'git'],
             'toolchain_configuration': 'OpenStack',
             },
    'swift': {'git_repo': DEFAULT_REPO_PATH + 'swift',
              'lp_cache': DEFAULT_LPCACHE_PATH + 'swift' + '/',
              'corpus_location': DEFAULT_CORPUS_PATH,
              'file_include_suffix': DEFAULT_FILTER_INCLUDE_FILE_SUFFIX,
              'file_exclude_prefix': ['swift/tests/', 'tools/'],
              'toolchain': ['launchpad', 'gerrit', 'git'],
              'toolchain_configuration': 'OpenStack',
              },
    'cinder': {'git_repo': DEFAULT_REPO_PATH + 'cinder',
               'lp_cache': DEFAULT_LPCACHE_PATH + 'cinder' + '/',
               'corpus_location': DEFAULT_CORPUS_PATH,
               'file_include_suffix': DEFAULT_FILTER_INCLUDE_FILE_SUFFIX,
               'file_exclude_prefix': ['cinder/tests/', 'tools/'],
               'toolchain': ['launchpad', 'gerrit', 'git'],
               'toolchain_configuration': 'OpenStack',
               },
    'heat': {'git_repo': DEFAULT_REPO_PATH + 'heat',
             'lp_cache': DEFAULT_LPCACHE_PATH + 'heat' + '/',
             'corpus_location': DEFAULT_CORPUS_PATH,
             'file_include_suffix': DEFAULT_FILTER_INCLUDE_FILE_SUFFIX,
             'file_exclude_prefix': ['heat/tests/', 'tools/'],
             'toolchain': ['launchpad', 'gerrit', 'git'],
             'toolchain_configuration': 'OpenStack',
             },
    'glance': {'git_repo': DEFAULT_REPO_PATH + 'glance',
               'lp_cache': DEFAULT_LPCACHE_PATH + 'glance' + '/',
               'corpus_location': DEFAULT_CORPUS_PATH,
               'file_include_suffix': DEFAULT_FILTER_INCLUDE_FILE_SUFFIX,
               'file_exclude_prefix': ['glance/tests/', 'tools/'],
               'toolchain': ['launchpad', 'gerrit', 'git'],
               'toolchain_configuration': 'OpenStack',
               },
    'kubernetes': {'git_repo': DEFAULT_REPO_PATH + 'kubernetes',
                   'corpus_location': DEFAULT_CORPUS_PATH,
                   'file_include_suffix': DEFAULT_FILTER_INCLUDE_FILE_SUFFIX,
                   'file_exclude_prefix': DEFAULT_FILTER_EXCLUDE_FILE_PREFIX,
                   'toolchain': ['git'],
                   'toolchain_configuration': 'Kubernetes',
                   },
}


def get_filter_config(project):
    if project not in PROJECT_CONFIG:
        raise Exception('get_filter_config: unknown project config ' + project)
    else:
        return {'include_suffix':
                PROJECT_CONFIG[project]['file_include_suffix'],
                'exclude_prefix':
                PROJECT_CONFIG[project]['file_exclude_prefix'],
                }


def get_corpus_dir(project):
    if project not in PROJECT_CONFIG:
        raise Exception('get_corpus_dir: unknown project config ' + project)
    else:
        return PROJECT_CONFIG[project]['corpus_location']


def get_repo_name(project):
    if project not in PROJECT_CONFIG:
        raise Exception('get_repo_dir: unknown project config ' + project)
    else:
        return PROJECT_CONFIG[project]['git_repo']


def get_lpcache_dir(project):
    if project not in PROJECT_CONFIG:
        raise Exception('get_lpcache_dir: unknown project config ' + project)
    else:
        return PROJECT_CONFIG[project]['lp_cache']
