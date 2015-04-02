#
# python_introspection.py - Language aware processing of python source files
#
# Author:  Doug Williams - Copyright 2015
#
# Last updated 3/31/2015
#
# Nomenclaure:
# - AST, ST - Abstract Syntax Tree  (from Python standatd library)
# - HT - Hash Tree - AST annotates with signatures of each included element
#
# History:
# - 3/31/15 - Initial version, based on expderimental work in iPython notebook
# - 4/1/15  - Adds support for language aware difference: pyDiff
#
# Top Level Routines:
#  - pyDiff():  Compares two AST for differences
#
#  - show_st(): Display routine for AST.  Wrapper for DisplayNodes
#
#  - get_all_files():  Recursively walks over source pool, returning file paths
#
#  - get_st_from_file():  generates ast parse tree from file
#
#  - get_total_nodes(): Computes node count for ast sub-tree or list of
#                       sub-trees.  Wrapper for CountTotalNodes
#
#  - get_stats(): Computes raw data for various code complexity stats for
#                 and ast tree or list of trees. Wrapper for ComputeComplexity
#
#  - get_delta_stats(): Compute difference between two sets of stats
#
#  - combine_stats(): Aggregates a list of stats sets
#
#  - get_unique_classes():  Debug/development routine to identify _ast classes
#                           within a pool of source code.  Wrapper for
#                           FindUniqueClasses
#
# from python_introspection import pyDiff
# from python_introspection import get_all_files, get_st_from_file, show_st
# from python_introspection import get_total_nodes
# from python_introspection import get_stats, get_delta_stats, combine_stats
# from python_introspection import get_unique_classes

import ast
from pprint import pprint
import collections
import os
import fnmatch
import hashlib


def get_all_files(starting_dir, pattern='*.py'):
    """Iterator over source pool"""
    for root, dirnames, filenames in os.walk(starting_dir):
        for filename in fnmatch.filter(filenames, pattern):
            yield os.path.join(root, filename)


def get_st_from_file(fname):
    """Displays Abstract Tree for python file"""
    with open(fname) as f:
        contents = f.read()
    st = ast.parse(contents)
    return st


#
# Display Routines
#

def show_st(node):
    if isinstance(node, list) or isinstance(node, tuple):
        dn = DisplayNodes()
        for n in node:
            dn.visit(n)
    else:
        DisplayNodes().visit(node)


class DisplayNodes(ast.NodeVisitor):
    level = 0
    indent = 3
    SPACES = ' '*3*30
    filter_fields = True  # currently unused
    ignore_fields = ['body', 'elts', 'keys', 'value',  # Currently unused
                     'values', 'targets', 'ctx',
                     'left', 'op', 'right', 'args',
                     'ops', 'test', 'orelse', 'comparators',
                     'type', 'func', 'slice', 'level', 'names']

    def __init__(self, filter_fields=False):
        self.level = 0
        self.filter_fields = filter_fields
        super(DisplayNodes, self).__init__()

    def visit_If(self, node):
        self.level += 1
        print self.SPACES[0:(self.level)*self.indent] + '**TEST**'
        self.visit(node.test)

        print self.SPACES[0:(self.level)*self.indent] + '**BODY**'
        if isinstance(node.body, list) or isinstance(node.body, tuple):
            for n in node.body:
                self.visit(n)
        else:
            self.visit(node.body)

        if node.orelse:
            print self.SPACES[0:(self.level)*self.indent] + '**ORSELSE**'
            if isinstance(node.orelse, list) or isinstance(node.orelse, tuple):
                for n in node.orelse:
                    self.visit(n)
            else:
                self.visit(node.orelse)
        self.level -= 1

    def visit_For(self, node):
        self.level += 1

        print self.SPACES[0:(self.level)*self.indent] + '**TARGET**'
        if isinstance(node.target, list):
            for n in node.target:
                self.visit(n)
        else:
            self.visit(node.target)

        print self.SPACES[0:(self.level)*self.indent] + '**ITER**'
        if isinstance(node.iter, list) or isinstance(node.iter, tuple):
            for n in node.iter:
                self.visit(n)
        else:
            self.visit(node.iter)

        print self.SPACES[0:(self.level)*self.indent] + '**BODY**'
        if isinstance(node.body, list) or isinstance(node.body, tuple):
            for n in node.body:
                self.visit(n)
        else:
            self.visit(node.body)

        if node.orelse:
            print self.SPACES[0:(self.level)*self.indent] + '**ORSELSE**'
            if isinstance(node.orelse, list) or isinstance(node.orelse, tuple):
                for n in node.orelse:
                    self.visit(n)
            else:
                self.visit(node.orelse)
        self.level -= 1

    def visit_While(self, node):
        self.level += 1
        print self.SPACES[0:(self.level)*self.indent] + '**TEST**'
        self.visit(node.test)

        print self.SPACES[0:(self.level)*self.indent] + '**BODY**'
        if isinstance(node.body, list) or isinstance(node.body, tuple):
            for n in node.body:
                self.visit(n)
        else:
            self.visit(node.body)

        if node.orelse:
            print self.SPACES[0:(self.level)*self.indent] + '**ORSELSE**'
            if isinstance(node.orelse, list) or isinstance(node.orelse, tuple):
                for n in node.orelse:
                    self.visit(n)
            else:
                self.visit(node.orelse)
        self.level -= 1

    def generic_visit(self, node):
        for k, v in ast.iter_fields(node):
            if not v:
                continue
            vv = repr(v)
            if 'object at 0x' not in vv:
                print self.SPACES[0:(self.level+1)*self.indent], k, ' = ', vv

        self.level += 1
        super(DisplayNodes, self).generic_visit(node)
        self.level -= 1

    def visit(self, node):
        print self.SPACES[0:self.level*self.indent], self.level, ':',
        if isinstance(node, ast.expr):
            print 'EXPR', type(node).__name__, [node.lineno, node.col_offset],
        elif isinstance(node, ast.stmt):
            print 'STMT', type(node).__name__, [node.lineno, node.col_offset],
        else:
            print type(node).__name__,
        print
        super(DisplayNodes, self).visit(node)

#
# Walks AST (sub) tree computing node count
#


class CountTotalNodes(ast.NodeVisitor):
    total_nodes = 0

    def __init__(self):
        self.total_nodes = 0
        super(CountTotalNodes, self).__init__()

    def generic_visit(self, node):
        self.total_nodes += 1
        super(CountTotalNodes, self).generic_visit(node)

    def getTotalNodes(self):
        return self.total_nodes


def get_total_nodes(node):
    ctn = CountTotalNodes()
    if isinstance(node, list):
        for n in node:
            ctn.visit(n)
    else:
        ctn.visit(node)
    return ctn.getTotalNodes()

#
# Walks AST (sub)tree computing various code complexity features
#


class ComputeComplexity(ast.NodeVisitor):
    function_depth = 0
    class_depth = 0
    nested_functions = 0
    nested_classes = 0
    func_in_class = 0
    total_statements = 0
    total_expressions = 0
    other_nodes = 0
    total_classes = 0
    total_functions = 0
    total_confitionals = 0
    total_comprehension = 0
    comprehension_with_if = 0
    total_tests = 0
    test_nodes = 0
    leaf_nodes = 0
    total_nodes = 0
    detail = collections.defaultdict(int)

    def __init__(self):
        self.function_depth = 0
        self.class_depth = 0
        self.nested_functions = 0
        self.nested_classes = 0
        self.func_in_class = 0
        self.total_statements = 0
        self.total_expressions = 0
        self.other_nodes = 0
        self.total_classes = 0
        self.total_functions = 0
        self.total_conditionals = 0
        self.total_comprehension = 0
        self.comprehension_with_if = 0
        self.total_tests = 0
        self.test_nodes = 0
        self.leaf_nodes = 0
        self.total_nodes = 0
        self.detail = collections.defaultdict(int)
        super(ComputeComplexity, self).__init__()

    def visit_ClassDef(self, node):
        # print(node.name)
        if self.class_depth > 0:
            self.nested_classes += 1
        self.total_classes += 1
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_FunctionDef(self, node):
        if self.function_depth > 0:
            self.nested_functions += 1
        if self.class_depth > 0:
            self.func_in_class += 1
        # print(node.name)
        self.total_functions += 1
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_For(self, node):
        self.total_conditionals += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.total_conditionals += 1
        self.total_tests += 1
        self.test_nodes += get_total_nodes(node.test)
        self.generic_visit(node)

    def visit_If(self, node):
        self.total_conditionals += 1
        self.total_tests += 1
        self.test_nodes += get_total_nodes(node.test)
        self.generic_visit(node)

    def visit_TryExcept(self, node):
        self.total_conditionals += 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.total_comprehension += 1
        self.comprehension_with_if += len(node.ifs) > 0
        self.total_tests += len(node.ifs)
        self.test_nodes += sum([get_total_nodes(i) for i in node.ifs])
        self.generic_visit(node)

    def generic_visit(self, node):
        self.total_nodes += 1
        node_type = type(node).__name__
        self.detail[node_type] += 1
        if isinstance(node, ast.stmt):
            self.total_statements += 1
        elif isinstance(node, ast.expr):
            self.total_expressions += 1
        else:
            self.other_nodes += 1

        if len([1 for x in ast.iter_child_nodes(node)]) == 0:
            self.leaf_nodes += 1

        super(ComputeComplexity, self).generic_visit(node)

    def getStats(self):
        return {'total_statements': self.total_statements,
                'total_expressions': self.total_expressions,
                'other_nodes': self.other_nodes,
                'total_classes': self.total_classes,
                'total_functions': self.total_functions,
                'total_conditionals': self.total_conditionals,
                'total_comprehension': self.total_comprehension,
                'comprehension_with_if': self.comprehension_with_if,
                'total_tests': self.total_tests,
                'test_nodes': self.test_nodes,
                'leaf_nodes': self.leaf_nodes,
                'total_nodes': self.total_nodes,
                'detail': self.detail,
                'nested_functions': self.nested_functions,
                'nested_classes': self.nested_classes,
                'func_in_class':  self.func_in_class,
                }


def get_stats(node, convert_to_dict=False):
    """Computes code complexity states for an AST tree or list of trees"""
    cc = ComputeComplexity()
    if isinstance(node, list):
        for n in node:
            cc.visit(n)
    else:
        cc.visit(node)

    stats = cc.getStats()
    if convert_to_dict:
        stats['detail'] = dict(stats['detail'])
    return stats


def get_delta_stats(stats1, stats2, convert_to_dict=False):
    """Computes difference between two sets of stats"""
    results = {}
    # aggregate results. details requires special handling
    for k in stats1.keys():
        if k == 'detail':
            continue
        results[k] = stats1[k] - stats2[k]

    # aggregate details, special care required since dicts are sparse
    rdetail = collections.defaultdict(int)
    detail1 = stats1['detail']
    if not isinstance(detail1, collections.defaultdict):
        detail1 = collections.defaultdict(int, detail1)
    detail2 = stats2['detail']
    if not isinstance(detail2, collections.defaultdict):
        detail2 = collections.defaultdict(int, detail2)
    for k in set(detail1.keys()).union(set(detail2.keys())):
        rdetail[k] = detail1[k] - detail2[k]

    if convert_to_dict:
        rdetail = dict(rdetail)
    results['detail'] = rdetail
    return results


def combine_stats(list_of_stats, convert_to_dict=False):
    """Combines a set of stats """
    results = collections.defaultdict(int)
    detail = collections.defaultdict(int)
    for stat in list_of_stats:
        # aggregate results. details requires special handling
        for k in stat.keys():
            if k == 'detail':
                continue
            results[k] += stat[k]

        # aggregate details, special care required since dicts are sparse
        for k, v in stat['detail'].items():
            detail[k] += v

    if convert_to_dict:
        results = dict(results)
        detail = dict(detail)
    results['detail'] = detail
    return results

#
# Identifies Unque classes within AST tree
#


class FindUniqueClasses(ast.NodeVisitor):
    statements = {}
    expressions = {}
    all_classes = {}

    def __init__(self):
        self.statements = {}
        self.expressions = {}
        self.other_classes = {}
        super(FindUniqueClasses, self).__init__()

    def generic_visit(self, node):
        node_type = type(node).__name__
        if isinstance(node, ast.stmt):
            self.statements[node_type] = 1
        if isinstance(node, ast.expr):
            self.expressions[node_type] = 1
        self.all_classes[node_type] = 1

        super(FindUniqueClasses, self).generic_visit(node)

    def getStats(self):
        stmt_and_expr = list(set(self.statements.keys()).intersection(
            set(self.expressions.keys())))
        stmt_or_expr = set(self.statements.keys()).union(
            set(self.expressions.keys()))
        non_stmt_expr = list(set(self.all_classes.keys()).intersection(
            stmt_or_expr))

        return {'statements': self.statements.keys(),
                'expressions': self.expressions.keys(),
                'all_classes': self.all_classes.keys(),
                'stmt_and_expr': stmt_and_expr,
                'non_stmt_expr': non_stmt_expr,
                }


def get_unique_classes(node):
    """Finds unique classes contained in an AST tree or list of trees"""
    fuc = FindUniqueClasses()
    if isinstance(node, list):
        for n in node:
            fuc.visit(n)
    else:
        fuc.visit(node)
    return fuc.getStats()

#
# Python Diff Code
#
# Helper routines for pyDiff
#


def st2hashes(st, parent=None, level=0, max_level=-1):
    """Recursively walk ast, returning hashes for every statement,
    for lower level items, returns string representing subtree"""

    hash_entry = {'stmt_type': None, 'hash': None, 'ast': None,
                  'child_hashes': [], 'parent': None,
                  'lineno': -1, 'lineno_min': -1, 'lineno_max': -1}
    result_string = ''

    # represent current statement
    result_string += type(st).__name__ + '\n'
    for k, v in ast.iter_fields(st):
        vv = repr(v)
        if v and 'object at 0x' not in vv:
            result_string += k + '=' + vv + '\n'

    if isinstance(st, ast.stmt) or isinstance(st, ast.expr):

        hash_entry['ast'] = st
        hash_entry['stmt_type'] = type(st).__name__
        hash_entry['parent'] = parent
        hash_entry['lineno'] = st.lineno
        hash_entry['lineno_min'] = st.lineno
        hash_entry['lineno_max'] = st.lineno
        """
        hash_entry['child_hashes'].append(
            {'stmt_type': 'self',
             'hash': hashlib.sha1(result_string).hexdigest(),
             'ast': st,
             'parent': parent,
             'child_hashes': [],
             'lineno': st.lineno,
             'lineno_min': st.lineno,
             'lineno_max': st.lineno})
        """

    # now process any child statements
    for child_st in ast.iter_child_nodes(st):
        child_hash_entry, child_string = st2hashes(child_st, parent=st,
                                                   level=level+1)
        if child_hash_entry:
            hash_entry['child_hashes'].append(child_hash_entry)
            hash_entry['lineno_min'] = min(hash_entry['lineno_min'],
                                           child_hash_entry['lineno_min'])
            hash_entry['lineno_max'] = min(hash_entry['lineno_max'],
                                           child_hash_entry['lineno_max'])
        elif isinstance(st, ast.stmt) or isinstance(st, ast.expr):
            if isinstance(child_st, ast.expr) or isinstance(child_st,
                                                            ast.expr):
                lineno = child_st.lineno
            else:
                lineno = -1
            hash_entry['child_hashes'].append(
                {'stmt_type': type(child_st).__name__,
                 'hash': hashlib.sha1(child_string).hexdigest(),
                 'ast': child_st,
                 'parent': st,
                 'child_hashes': [],
                 'lineno': lineno,
                 'lineno_min': lineno,
                 'lineno_max': lineno})
        result_string += child_string

    if isinstance(st, ast.Module):
        return hash_entry['child_hashes'], None
    elif isinstance(st, ast.stmt) or isinstance(st, ast.expr):
        # print 'Returning Instance Entry'
        hash_entry['hash'] = hashlib.sha1(result_string).hexdigest()
        return hash_entry, result_string
    else:
        return None, result_string


def clear_pairings(htree):
    """Initializes fields used when aligning two trees"""
    if isinstance(htree, list):
        for x in htree:
            clear_pairings(x)
    else:
        htree.update({'match': None,
                      'match_type': 'Unknown',
                      'differences': []})
        for child in htree['child_hashes']:
            clear_pairings(child)


def get_hashes(htree, depth=0, strip_depth=True):
    """returns list of [hash, htree], can be used to built dict
    results are sorted in breadth-first order, optionally including
    the depth within the tree"""
    result = []
    if isinstance(htree, list):
        for x in htree:
            result.extend(get_hashes(x, depth=depth+1))
    else:
        result.append([htree['hash'], htree, depth])
        for child in htree['child_hashes']:
            result.extend(get_hashes(child, depth=depth+1))

    if depth > 0:
        return result
    else:
        # return
        result = sorted(result, key=lambda x: x[2])
        if strip_depth:
            return [[h, ht] for h, ht, l in result]
        else:
            return result


def print_fields(node):
    """pyDiff helper routine for verbose mode """
    for k, v in ast.iter_fields(node):
        if not v:
                continue
        vv = repr(v)
        if 'object at 0x' not in vv:
            print '    ', k, ' = ', vv


def propagate_matches(htree, verbose=True):
    """In subtrees with unique match, promote all child-trees
    with potential match to unique match """
    changes = 1
    runaway = 100
    # Iteratively resolve
    while changes > 0 and runaway > 0:
        changes = 0
        runaway -= 1
        for _, ht in get_hashes(htree):
            if ht['match_type'] == 'Unique':
                for child in ht['child_hashes']:
                    if child['match_type'] == 'Potential':
                        # print '*',
                        changes += 1
                        child['match_type'] = 'Unique'
                    elif child['match_type'] == 'Unmatched':
                        raise Exception('propagate_matches: match conflict,'
                                        + ' unmatched child for unqiuely '
                                        + 'matched parent')
    unresolved = [[h, ht] for h, ht in get_hashes(htree)
                  if ht['match_type'] == 'Potential']
    unmatched = [[h, ht] for h, ht in get_hashes(htree)
                 if ht['match_type'] == 'Unmatched']
    if verbose:
        print 'propagate_matches:  Unresolved:', len(unresolved),
        print 'Unmatched:', len(unmatched)

    return unresolved, unmatched


def resolve_potential_matches(htree1, htree2, runaway=100, verbose=True):
    changes_found = True
    last_unresolved = -1
    while changes_found and runaway > 0:
        runaway -= 1
        unresolved1, unmatched1 = propagate_matches(htree1, verbose=verbose)
        unresolved2, unmatched2 = propagate_matches(htree2, verbose=verbose)

        # further resolve my matching common instance counts
        unresolved_counts1 = collections.defaultdict(int)
        unresolved_counts2 = collections.defaultdict(int)
        for htree, unresolved_counts in [[htree1, unresolved_counts1],
                                         [htree2, unresolved_counts2]]:
            for h, ht in get_hashes(htree):
                if ht['match_type'] == 'Potential':
                    unresolved_counts[h] += 1

        matched = [h for h in unresolved_counts1.keys()
                   if unresolved_counts1[h] == unresolved_counts2[h]]
        if verbose:
            print
            print 'resolve_potential_matches: Resolved after propagate: ',
            print len(matched)

        for unresolved in [unresolved1, unresolved2]:
            for h, ht in unresolved:
                if h in matched:
                    ht['match_type'] = 'Unique'
        unresolved1 = [[h, ht] for h, ht in unresolved1 if h not in matched]
        unresolved2 = [[h, ht] for h, ht in unresolved2 if h not in matched]

        if verbose:
            print 'resolve_potential_matches: For HT1 -- Unresolved:',
            print len(unresolved1), 'Unmatched:', len(unmatched1)
            print 'resolve_potential_matches: For HT2 -- Unresolved:',
            print len(unresolved2), 'Unmatched:', len(unmatched2)

        all_unresolved = len(unresolved1) + len(unresolved2)
        if verbose:
            print 'resolve_potential_matches: Current unresolved:',
            print all_unresolved, 'Previous unresolved:', last_unresolved
        if all_unresolved == last_unresolved:
            break
        last_unresolved = all_unresolved

    return unresolved1, unmatched1, unresolved2, unmatched2


#
# Main pyDiff routine
#

def pyDiff(diff1_st, diff2_st, verbose=False):
    """Top level diff routine """
    htree1, _ = st2hashes(diff1_st)
    clear_pairings(htree1)          # may want to move into st2hashes
    htree2, _ = st2hashes(diff2_st)
    clear_pairings(htree1)          # may want to move into st2hashes

    # Builds dict of subtrees indexed by hash, note hashes are non-unique
    hashes2htree1 = collections.defaultdict(list)
    hashes2htree2 = collections.defaultdict(list)
    for htree, hashes2htree in [[htree1, hashes2htree1],
                                [htree2, hashes2htree2]]:
        for k, ht in get_hashes(htree):
            hashes2htree[k].append(ht)

    if verbose:
        print 'pyDiff: Unique hashes - HT1:', len(hashes2htree2.keys()),
        print 'HT2:', len(hashes2htree2.keys())

    # now identify shared and differing hashes
    common = set(hashes2htree1.keys()).intersection(set(hashes2htree2.keys()))
    diff1 = set(hashes2htree1.keys()).difference(set(hashes2htree2.keys()))
    diff2 = set(hashes2htree2.keys()).difference(set(hashes2htree1.keys()))
    if verbose:
        print 'pyDiff: Common:', len(common), 'diff1:', len(diff1),
        print 'diff2:', len(diff2)

    # Try to match trees
    for k in common:
        # In instances where only one hash in each map, assume trees correspond
        # note:  May be invalid assumption for lower-level sub-trees
        if len(hashes2htree1[k]) == 1 and len(hashes2htree2[k]) == 1:
            hashes2htree1[k][0]['match'] = hashes2htree2[k][0]
            hashes2htree2[k][0]['match'] = hashes2htree1[k][0]
            hashes2htree1[k][0]['match_type'] = 'Unique'
            hashes2htree2[k][0]['match_type'] = 'Unique'
        else:
            # For non-unique pairs, declare as potential and resolve later
            for hashes2htree in [hashes2htree1, hashes2htree2]:
                for ht in hashes2htree[k]:
                    ht['match_type'] = 'Potential'

    # label everyting else as unmatched
    for diff, hashes2htree in [[diff1, hashes2htree1], [diff2, hashes2htree2]]:
        for k in diff:
            for ht in hashes2htree[k]:
                ht['match_type'] = 'Unmatched'

    # display intermediate results
    if verbose:
        print
        print 'pyDiff: Intermediate results'
        for diff, hashes2htree in [[diff1, hashes2htree1],
                                   [diff2, hashes2htree2]]:
            for k in diff:
                if len(hashes2htree[k]) > 1:
                    print '    ', k, len(hashes2htree[k])
                for ht in hashes2htree[k]:
                    print '    ', type(ht['ast']).__name__, 'Parent:',
                    print type(ht['parent']).__name__, ht['lineno']
                    print_fields(ht['ast'])
            print '   ', '-'*40
        print

    # Iteratively resolve disposition of potential matches
    [unresolved1, unmatched1,
     unresolved2, unmatched2] = resolve_potential_matches(htree1, htree2,
                                                          verbose=verbose)

    if verbose:
        print
        print 'pyDiff: HT1 -- Unresolved:', len(unresolved1),
        print 'Unmatched:', len(unmatched1)
        print 'pyDiff: HT2 -- Unresolved:', len(unresolved2),
        print 'Unmatched:', len(unmatched2)

    return unresolved1, unmatched1, unresolved2, unmatched2
