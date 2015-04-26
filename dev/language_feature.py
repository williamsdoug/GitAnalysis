#
# language_feature.py - Language-specific processing for Python, including
#                       change detection and language_related feature
#                       extraction.  Primary integration with Git_Extract.
#
# Author:  Doug Williams - Copyright 2015
#
# Last updated 4/26/2015
#
# History:
# - 4/21/15 - Initial version of file
# - 4/26/15 - Integrate PyDiff and and process_commit_diff feature extraction
#             from language_feature.
# - 4/26/15 - Enrich test-related metrics
#
#
# Top Level Routines:
#
# from language_feature import process_commit_diff
#

import ast
from pprint import pprint
import collections
import re
from git import Repo
import git
import sys

import radon
import radon.visitors
import radon.complexity
import radon.raw
import radon.metrics

from git_analysis_config import get_repo_name, get_filter_config
from python_introspection import get_total_nodes, get_stats
from python_introspection import getDepth
from PythonDiff import pythonDiff


#
# WARNING HACK: Clones of functions from Git_Extract duplicatee to avoid
# interdependent imports.  To Do: Clean-up
#


# from Git_Extract import isValidBlob
def isValidBlob(blob):
    """Helper function to validate non-null blob"""
    return (blob and str(blob) != git.objects.blob.Blob.NULL_HEX_SHA)


# from Git_Extract import filter_file
def filter_file(fname, filter_config):
    for prefix in filter_config['exclude_prefix']:
        if fname.startswith(prefix):
            return False
    for suffix in filter_config['include_suffix']:
        if fname.endswith(suffix):
            return True
    return False


def printSubtree(subtrees, level=0, indent=2):
    """Debug routine to print diff subtree"""
    for tree in subtrees:
        if tree['full_match']:
            continue
        if tree['pair']:
            if tree['node_match']:
                has_pair = 'P'
            else:
                has_pair = 'X'
        else:
            has_pair = '-'
        if 'lineno' in tree:
            print ' '*(level*indent), has_pair, tree['type'],
            print '[line', tree['lineno'], ']',
            if tree['pair'] and 'lineno' in tree['pair']:
                print '[pair line', tree['pair']['lineno'], ']',
            else:
                # print,
                pass
        else:
            print ' '*(level*indent), has_pair, tree['type'],

        # print ' '*(level*indent+10), tree['hash'],
        print tree['name'] if tree['name'] else '',
        print tree['target'] if tree['target'] else '',
        print tree['sig']

        if 'subtrees' in tree and tree['subtrees']:
            printSubtree(tree['subtrees'], level=level+1)


def showRanges(line_ranges, verbose=False):
    """Debug Function"""
    if verbose:
        print 'ranges:'
        for r in line_ranges:
            for c in r['changes']:
                print c, r['text'][c-r['start']]
        print


def showDiffResult(subtreeA, subtreeB, verbose=True):
    """debug function"""
    if verbose:
            printSubtree(subtreeA)
            print '-'*20
            printSubtree(subtreeB)
            print
            print '+'*40


def get_st_from_blob(blob, verbose=False):
    """Extracts Syntax Tree (AST) from git blob"""
    txt = blob.data_stream.read()
    st = ast.parse(txt, filename=blob.path)
    return st, txt.split('\n')


def get_line_ranges(st, txt):
    """Extract line number ranges from Abstract Syntax Tree"""
    assert isinstance(st, ast.Module)
    lineCount = len(txt)
    lines = [{'start': stmt.lineno, 'st': stmt, 'changes': []}
             for stmt in st.body]
    if len(lines) == 0:
        return []

    for i in range(0, len(lines) - 1):
        lines[i]['end'] = max(lines[i]['start'], lines[i+1]['start'] - 1)

    lines[-1]['end'] = lineCount
    for line in lines:
        line['text'] = txt[line['start']-1:line['end']]
    return lines

minus_re = re.compile('\-(\d+)')
plus_re = re.compile('\+(\d+)')


def parse_diff_txt(txt, verbose=False):
    """Parses git diff, returning line numbers containing changes """
    lineA = -1
    lineB = -1
    changesA = []
    changesB = []

    lines = txt.split('\n')
    for line in lines:
        if line.startswith('@@'):   # Start of diff hunk
            range_info = line.split('@@')[1]
            match = re.search(minus_re, range_info)  # -start, len
            if match:
                txt = match.group()
                lineA = int(txt[1:])
            match = re.search(plus_re, range_info)   # +start, len
            if match:
                txt = match.group()
                lineB = int(txt[1:])
            continue
        elif line.startswith('---'):
            continue
        elif line.startswith('+++'):
            continue
        elif line.startswith('-'):
            changesA.append(lineA)
            if verbose:
                print 'A', lineA, line
            lineA += 1
        elif line.startswith('+'):
            changesB.append(lineB)
            if verbose:
                print 'B', lineB, line
            lineB += 1
        elif line.startswith(' '):
            # To be conservative, include both (needed for inserts)
            changesA.append(lineA)
            changesB.append(lineB)
            if verbose:
                print 'A/B', line
            lineA += 1
            lineB += 1
        # print line
    return changesA, changesB


def annotate_changes(line_ranges, changes):
    """Annotates parser output with change information for git diff """
    for lineno in changes:
        for r in line_ranges:
            if lineno >= r['start'] and lineno <= r['end']:
                r['changes'].append(lineno)
                continue


def trimAST(line_ranges):
    """Removes subtrees not flagged by git diff"""
    return ast.Module(body=[r['st'] for r in line_ranges if r['changes']])


def addRangeInfoToSubtree(line_ranges, subtree):
    """Merge line_range info into tree structure
    Note:  This info may be useful during decomposition and metric calculation
    """
    for tree in subtree:
        found = False
        for r in line_ranges:
            if tree['ast'] == r['st']:
                # print 'Foung match'
                tree['range_info'] = r
                found = True
                continue
        if not found:
            # print 'Match not found'
            pass


def ASTwrapper(st):
    if isinstance(st, ast.ClassDef) or isinstance(st, ast.FunctionDef):
        return st
    result = ast.FunctionDef(name='***dummy***',
                             body=[st])
    try:
        result.lineno = st.lineno
        result.col_offset = st.col_offset
    except Exception:
        result.lineno = 1
        result.col_offset = 1
    return result


def get_cc(st):
    """Wraps CC, inserting dummy function if needed"""

    if (isinstance(st, ast.FunctionDef) or isinstance(st, ast.FunctionDef)
        or isinstance(st, ast.Module)):
            this_st = st
            wrapped = False
    else:
        this_st = ASTwrapper(st)
        wrapped = True
    result = radon.complexity.cc_visit_ast(this_st)

    if len(result) == 1:
        if wrapped:
            return result[0].complexity - 1
        else:
            return result[0].complexity
    else:
        return sum([r.complexity for r in result])


def getMetrics(blob, result, verbose=True):
    st, source = get_st_from_blob(blob)
    source = '\n'.join(source)
    raw = radon.raw.analyze(source)
    result['lloc'] = raw.lloc
    result['sloc'] = raw.sloc
    result['mi'] = radon.metrics.mi_visit(source, multi=True)
    result['hal'] = radon.metrics.h_visit_ast(st)
    result['nodes'] = get_total_nodes(st)
    result['complexity'] = get_cc(st)
    result['depth'] = getDepth(ASTwrapper(st))[0]


def aggregateMetrics(entries):
    results = {}
    results['cc'] = sum([e['cc'] for e in entries
                         if 'test' not in e['name']])
    results['test_cc'] = sum([e['cc'] for e in entries
                              if 'test' in e['name']])
    results['changes'] = sum([e['changes'] for e in entries
                              if 'test' not in e['name']])
    results['test_changes'] = sum([e['changes'] for e in entries
                                   if 'test' in e['name']])
    results['complexity'] = sum([e['complexity'] for e in entries
                                 if 'test' not in e['name']])
    results['test_complexity'] = sum([e['complexity'] for e in entries
                                      if 'test' in e['name']])
    results['new_functions'] = sum([e['new_functions'] for e in entries
                                    if 'test' not in e['name']])
    results['test_new_functions'] = sum([e['new_functions'] for e in entries
                                         if 'test' in e['name']])
    results['new_classes'] = sum([e['new_classes'] for e in entries
                                  if 'test' not in e['name']])
    results['test_new_classes'] = sum([e['new_classes'] for e in entries
                                       if 'test' in e['name']])
    results['lloc'] = sum([e['lloc'] for e in entries
                           if 'test' not in e['name']])
    results['test_lloc'] = sum([e['lloc'] for e in entries
                                if 'test' in e['name']])
    results['nodes'] = sum([e['nodes'] for e in entries
                            if 'test' not in e['name']])
    results['test_nodes'] = sum([e['nodes'] for e in entries
                                 if 'test' in e['name']])

    results['avg_expr_depth'] = \
        (float(sum([e['depth']['expr']['sum_of_depth']
                    for e in entries if 'test' not in e['name']] + [0]))
         / max(float(sum([e['depth']['expr']['instances']
                          for e in entries
                          if 'test' not in e['name']] + [0])), 1))

    results['avg_node_depth'] = \
        (float(sum([e['depth']['node']['sum_of_depth']
                    for e in entries if 'test' not in e['name']] + [0]))
         / max(float(sum([e['depth']['node']['instances']
                          for e in entries
                          if 'test' not in e['name']] + [0])), 1))

    results['avg_stmt_depth'] = \
        (float(sum([e['depth']['stmt']['sum_of_depth']
                    for e in entries if 'test' not in e['name']] + [0]))
         / max(float(sum([e['depth']['stmt']['instances']
                          for e in entries
                          if 'test' not in e['name']] + [0])), 1))

    results['max_expr_depth'] = max([e['depth']['expr']['max_depth']
                                     for e in entries
                                     if 'test' not in e['name']] + [0])

    results['max_node_depth'] = max([e['depth']['node']['max_depth']
                                     for e in entries
                                     if 'test' not in e['name']] + [0])

    results['max_stmt_depth'] = max([e['depth']['stmt']['max_depth']
                                     for e in entries
                                     if 'test' not in e['name']] + [0])
    return results


DECISION_OPS = ['If', 'For', 'While', 'With', 'ExceptHandler',
                'Lambda', 'Assert', 'BoolOp', 'ListComp', 'SetComp',
                'GeneratorExp', 'DictComp']


def computeChanges(tree):
    """Computes node changes and cyclomatic complex for a set of ast edits"""
    changes = 0
    complexity = 0
    new_functions = 0
    new_classes = 0

    if tree['full_match']:
        return changes, complexity, new_functions, new_classes
    if tree['pair']:
        if not tree['node_match']:
            changes += 1

            node_type = type(tree['ast']).__name__.split('.')[-1]
            if node_type in DECISION_OPS:
                complexity += 1

        if 'subtrees' in tree:
            for subtree in tree['subtrees']:
                (this_change, this_cc,
                 this_functions, this_classes) = computeChanges(subtree)
                changes += this_change
                complexity += this_cc
                new_functions += this_functions
                new_classes += this_classes
    else:
        changes += get_total_nodes(tree['ast'])
        complexity += get_cc(tree['ast'])

        if (isinstance(tree['ast'], ast.FunctionDef)
            or isinstance(tree['ast'], ast.FunctionDef)
            or isinstance(tree['ast'], ast.Module)
            ):
                stats = get_stats(tree['ast'])
                new_functions += stats['total_functions']
                new_classes += stats['total_classes']

    return changes, complexity, new_functions, new_classes


def process_commit_add(d, verbose=False):
    """Determines changes associated with a file add"""

    st, txt = get_st_from_blob(d.a_blob)
    changes = get_total_nodes(st)
    complexity = get_cc(st)
    stats = get_stats(st)
    return {'name': d.a_blob.path,
            'changes': changes,
            'cc': complexity,
            'new_functions': stats['total_functions'],
            'new_classes': stats['total_classes']}


def process_commit_diff(c, filter_config, verbose=False):
    """Apply language sensitive diff to each changed file"""
    # global START
    cid = c.hexsha
    if verbose:
        print
        print 'CID:', cid
    sys.stdout.flush()

    files = []
    results = []
    # for p in c.parents:    # iterate through each parent
    if len(c.parents) > 0:
        p = c.parents[0]
        i = c.diff(p, create_patch=True)

        #
        # Design note:  These iterators are non-exclusive.
        # An add appears both as an add and a modify with null blob_a

        for d in i:
            if verbose:  # verbose:
                print
                print 'A:', d.a_blob,
                if isValidBlob(d.a_blob):
                    print d.a_blob.path
                print 'B:', d.b_blob,
                if isValidBlob(d.b_blob):
                    print d.b_blob.path
                sys.stdout.flush()

            if ((not d.a_blob and not d.b_blob)
                or
                (isValidBlob(d.a_blob)
                 and (not filter_file(d.a_blob.path, filter_config)
                      or not d.a_blob.path.endswith('.py')))
                or
                (isValidBlob(d.b_blob)
                 and (not filter_file(d.b_blob.path, filter_config)
                      or not d.b_blob.path.endswith('.py')))):
                if verbose:
                    print 'skipping'
                continue

            if not isValidBlob(d.a_blob):
                if verbose:
                    print 'Delete'
                continue
            elif not isValidBlob(d.b_blob):
                if verbose:
                    print 'Add A'
                try:
                    result = process_commit_add(d, verbose=verbose)
                    if result['changes'] > 0:
                        getMetrics(d.a_blob, result, verbose=True)
                        results.append(result)
                        if verbose:
                            pprint(result)
                except SyntaxError:
                    pass
            elif (isValidBlob(d.a_blob) and isValidBlob(d.b_blob)
                  and d.b_blob.path.endswith('.py')):
                    try:
                        result = processDiff(d, verbose=verbose)
                        # print '*',
                        # sys.stdout.flush()
                        if result['changes'] > 0:
                            getMetrics(d.b_blob, result, verbose=True)
                            results.append(result)
                            if verbose:
                                pprint(result)
                    except SyntaxError:
                        pass
            else:
                raise Exception('Unknown change format')

    ret = {}
    if len(results) > 0:
        ret = {'individual': {x['name']: x for x in results},
               'aggregate': aggregateMetrics(results)}
        if verbose:
            print 'Metrics:'
            pprint(ret)
            print
    return ret


def processDiff(d, verbose=False):
    """Process individual diff pair """
    # print d.a_blob.path, d.b_blob.path,
    if verbose:
        print '+'*40

    # get diff text and identify ranges
    changesA, changesB = parse_diff_txt(d.diff, verbose=verbose)

    st_a, txt_a = get_st_from_blob(d.a_blob)

    line_rangesA = get_line_ranges(st_a, txt_a)
    annotate_changes(line_rangesA, changesA)
    if verbose:
        showRanges(line_rangesA)

    st_b, txt_b = get_st_from_blob(d.b_blob)
    line_rangesB = get_line_ranges(st_b, txt_b)
    annotate_changes(line_rangesB, changesB)
    if verbose:
        showRanges(line_rangesB)

    # When performing PythonDiff, limit diff to identified ranges
    prunedA = trimAST(line_rangesA)
    prunedB = trimAST(line_rangesB)
    subtreeA, subtreeB = pythonDiff(prunedA, prunedB, verbose=verbose)
    if verbose:
        showDiffResult(subtreeA, subtreeB)
    addRangeInfoToSubtree(line_rangesA, subtreeA)
    addRangeInfoToSubtree(line_rangesB, subtreeB)

    total_changes = 0
    total_complexity = 0
    total_new_functions = 0
    total_new_classes = 0
    for tree in subtreeB:
        changes, cc, new_functions, new_classes = computeChanges(tree)
        total_changes += changes
        total_complexity += cc
        total_new_functions += new_functions
        total_new_classes += new_classes
        # print 'Changes:', changes, 'CC:', cc
    if verbose:
        print 'Change nodes:', total_changes,
        print 'Change complexity:', total_complexity
        print 'New functions', total_new_functions
        print 'New classes', total_new_classes
        print

    return {'name': d.b_blob.path,
            'changes': total_changes,
            'new_functions': total_new_functions,
            'new_classes': total_new_classes,
            'cc': total_complexity}


#
# Debug code, should be deleted once no longer needed
#

def test_all_git_commits(project, verbose=False, limit=-1, skip=-1):
    """Top level routine to generate commit data """

    repo_name = get_repo_name(project)
    repo = Repo(repo_name)
    filter_config = get_filter_config(project)
    assert repo.bare is False

    commits = collections.defaultdict(list)
    new_process_commits(repo, commits, filter_config,
                        skip=skip, verbose=verbose, limit=limit)


def new_process_commits(repo, commits, filter_config,
                        skip=-1, limit=-1, verbose=False):
    """Extracts all commit from git repo, subject to max_count limit"""
    total_operations = 0
    total_errors = 0

    for h in repo.heads:
        for c in repo.iter_commits(h):

            if skip > 0:
                skip -= 1
                continue

            # for c in repo.iter_commits('master', max_count=max_count):
            cid = c.hexsha
            if cid in commits:
                continue
            # try:
            commits[cid] = {}
            diff_result = process_commit_diff(c, filter_config,
                                              verbose=verbose)
            commits[cid].update(diff_result)

            total_operations += 1
            if total_operations % 10 == 0:
                print '.',
            if total_operations % 100 == 0:
                print total_operations,
            limit -= 1
            if limit == 0:
                return commits
    return commits
