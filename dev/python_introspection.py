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
#
# Top Level Routines:
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
# from python_introspection import get_total_nodes
# from python_introspection import get_stats, get_delta_stats, combine_stats
# from python_introspection import get_unique_classes

import ast
from pprint import pprint
import collections

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
