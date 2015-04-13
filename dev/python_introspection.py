#
# python_introspection.py - Language aware processing of python source files
#
# Author:  Doug Williams - Copyright 2015
#
# Last updated 4/13/2015
#
# Nomenclaure:
# - AST, ST - Abstract Syntax Tree  (from Python standatd library)
# - HT - Hash Tree - AST annotates with signatures of each included element
#
# History:
# - 3/31/15 - Initial version, based on expderimental work in iPython notebook
# - 4/1/15  - Adds support for language aware difference: pyDiff
# - 4/3/15  - Adds support for computing AST depth.  Includes getDepth(),
#             ComputeDepth [nodeVisitor class], and displayDepthResults
# - 4/13/15 - PythonDiff now in standalone file
#
# Top Level Routines:
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
# - getDepth():  Function level and aggregate depth AST depth statistics at
#                statement, expression and node level.  getDepth() is wapper
#                for ComputeDepth AST nodeVisitor class.  displayDepthResults()
#                can be used to print out results.
#
# from python_introspection import get_all_files, get_st_from_file, show_st
# from python_introspection import get_total_nodes
# from python_introspection import get_stats, get_delta_stats, combine_stats
# from python_introspection import get_unique_classes
# from python_introspection import ComputeDepth, getDepth, displayDepthResults

import ast
# from pprint import pprint
import collections
import os
import fnmatch
# import hashlib


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


def print_fields(node):
    """pyDiff helper routine for verbose mode """
    for k, v in ast.iter_fields(node):
        if not v:
            continue
        vv = repr(v)
        if 'object at 0x' not in vv:
            print '    ', k, ' = ', vv


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
# Compute AST Depth statistics
#

class ComputeDepth(ast.NodeVisitor):
    """Collects statistics on tree depth by function, at statement,
    expression and node levels
    """
    results = []  # results for each function
    stats = {}    # per-function stats
    classes = []  # remember naming hierarchy for classes

    def init_stats(self):
        self.stats = {}
        for level in ['stmt', 'expr', 'node']:
            self.stats[level] = {'depth': 0,
                                 'max_depth': 0,
                                 'sum_of_depth': 0,
                                 'instances': 0}

    def __init__(self):
        self.classes = []
        self.results = []
        self.init_stats()
        super(ComputeDepth, self).__init__()

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.generic_visit(node)
        self.classes = self.classes[0:-1]

    def visit_FunctionDef(self, node):
        self.init_stats()
        self.generic_visit(node)

        name = node.name  # compute name, prefix with class name as appropriate
        if self.classes:
            name = '.'.join(self.classes) + '.' + name
        self.stats['name'] = name

        for level in ['stmt', 'expr', 'node']:  # compute per-function averages
            self.stats[level]['avg_depth'] = (
                float(self.stats[level]['sum_of_depth'])
                / float(self.stats[level]['instances']))
        self.results.append(self.stats)

    def generic_visit(self, node):
        # update per-node stats
        self.stats['node']['max_depth'] = max(self.stats['node']['max_depth'],
                                              self.stats['node']['depth'])
        self.stats['node']['sum_of_depth'] += self.stats['node']['depth']
        self.stats['node']['instances'] += 1
        self.stats['node']['depth'] += 1

        if isinstance(node, ast.stmt):  # update statement level stats
            self.stats['stmt']['depth'] += 1
            self.stats['stmt']['max_depth'] = max(
                self.stats['stmt']['max_depth'], self.stats['stmt']['depth'])
            self.stats['stmt']['sum_of_depth'] += self.stats['stmt']['depth']
            self.stats['stmt']['instances'] += 1

        if isinstance(node, ast.expr):  # update expression level stats
            self.stats['expr']['depth'] += 1
            self.stats['expr']['max_depth'] = max(
                self.stats['expr']['max_depth'], self.stats['expr']['depth'])
            self.stats['expr']['sum_of_depth'] += self.stats['expr']['depth']
            self.stats['expr']['instances'] += 1

        super(ComputeDepth, self).generic_visit(node)

        # decrement depth counters
        self.stats['node']['depth'] -= 1
        if isinstance(node, ast.stmt):
            self.stats['stmt']['depth'] -= 1
        if isinstance(node, ast.expr):
            self.stats['expr']['depth'] -= 1

    def visit(self, node):
        super(ComputeDepth, self).visit(node)
        return self.results


def getDepth(node):
    """Wrapper for ComputeDepth, also determines max and average
    depth across all functions for nodes, expressions and statements."""
    results = ComputeDepth().visit(node)
    totals = {}
    for level in ['stmt', 'expr', 'node']:
        totals[level] = {}
        totals[level]['max_depth'] = max([r[level]['max_depth']
                                          for r in results])
        totals[level]['sum_of_depth'] = sum([r[level]['sum_of_depth']
                                             for r in results])
        totals[level]['instances'] = sum([r[level]['instances']
                                          for r in results])
        totals[level]['avg_depth'] = (float(totals[level]['sum_of_depth'])
                                      / float(totals[level]['instances']))
    return totals, results


def displayDepthResults(totals, results):
    """Displays function level and aggregated depth statistics"""
    for r in results:
        print
        print r['name']
        for level in ['stmt', 'expr', 'node']:
            print '    %s -- avg %0.1f  peak %0.1f' % (level,
                                                       r[level]['avg_depth'],
                                                       r[level]['max_depth'])
    print
    print 'Aggregated:'
    for level in ['stmt', 'expr', 'node']:
        print '    %s -- avg %0.1f  peak %0.1f' % (level,
                                                   totals[level]['avg_depth'],
                                                   totals[level]['max_depth'])
