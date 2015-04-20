#
# PythonDiff.py -- Code to determine differences bertween two python files
#
# Author:  Doug Williams - Copyright 2015
#
# Last Updated: 13-Apr-2015
#
# Nomenclaure:
# - AST, ST - Abstract Syntax Tree  (from Python standatd library)
# - HT - Hash Tree - AST annotates with signatures of each included element
#
# Issues and To Do:
# - Need to reconsole with PythonIntrospection and remove obsolete code.
#
# History:
# - 4/x/15:  Initial version, created from iPython notebook
# - 4/13/15: Various bug fixes after testing against Nova git repo
#
# Top level Routines:
#  - pythonDiff() - Compares two AST sub-trees
#
#
# from PythonDiff import recursiveDiff, pythonDiff, printSubtree
# from PythonDiff import GetSubtrees, GetHash

import ast
# import _ast
from pprint import pprint
# import fnmatch
# import os
import collections
import hashlib
# from python_introspection import show_st


#
# Generatim of Hashes for subtrees
#


class GetHash(ast.NodeVisitor):
    """Generates hash for sub-tree"""

    # only return subtrees for any top level item (classes, functions,
    # statements).  Can call recursvely if needed, to handled class methods,
    # nested functions and nested classes

    text = ''

    def __init__(self):
        self.text = ''
        super(GetHash, self).__init__()

    def generic_visit(self, node):
        self.text += type(node).__name__ + '\n'
        for k, v in ast.iter_fields(node):
            vv = repr(v)
            if v and 'object at 0x' not in vv:
                self.text += k + '=' + vv + '\n'
        super(GetHash, self).generic_visit(node)

    def visit(self, node):
        super(GetHash, self).visit(node)
        return hashlib.sha1(self.text).hexdigest()

#
# Get all subtrees down to the expr level
#


def getTargetName(node):
    # show_st(node)
    """Returns text representing target for an assignment """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        # print getTargetName(node.value) + '__attribute__' + node.attr
        return getTargetName(node.value) + '__attribute__' + node.attr
    elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
        return '__'.join([getTargetName(e) for e in node.elts])
    elif isinstance(node, ast.Subscript):
        # print type(node)
        return (getTargetName(node.value)
                + '[' + getTargetName(node.slice) + ']')
    elif isinstance(node, ast.Index):
        # print type(node)
        return getTargetName(node.value)
    elif isinstance(node, ast.Slice):
        # print type(node)
        result = ''
        if node.lower:
            result += getTargetName(node.lower)
        result += ':'
        if node.upper:
            result += getTargetName(node.upper)
        result += ':'
        if node.step:
            result += ':' + getTargetName(node.step)
        return result
    elif isinstance(node, ast.ExtSlice):
        # print type(node)
        return ','.join([getTargetName(e) for e in node.dims])
    elif isinstance(node, ast.Num):
        return str(node.n)
    elif isinstance(node, ast.Str):
        return str(node.s)
    elif isinstance(node, ast.BinOp):
        return (getTargetName(node.left)
                + type(node.op).__name__
                + getTargetName(node.right))
    else:
        result = type(node).__name__
        params = [repr(k) + '=' + repr(v)
                  for k, v in ast.iter_fields(node)
                  if v and 'object at 0x' not in repr(v)]
        if params:
            result = result + '+' + '+'.join(params)
        print 'getTargetName: default format - ', result
        return result


class GetSubtrees(ast.NodeVisitor):
    """Explore one level of parse tree - currently only processe to statement level
    """

    # only return subtrees for any top level item (classes, functions,
    # statements). Can call recursvely if needed, to handled class methods,
    # nested functions and nested classes

    results = []  # results for each function
    entry = None
    depth = 0   # current depth in tree

    def __init__(self):
        self.results = []
        self.entry = None
        self.depth = 0
        super(GetSubtrees, self).__init__()

    @staticmethod
    def getSignature(node):
        sig = ''
        if not isinstance(node, ast.AST):
            raise Exception('getSignature - Invalid type')
        sig += type(node).__name__
        for k, v in ast.iter_fields(node):
            vv = repr(v)
            if v and 'object at 0x' not in vv:
                sig += ' ' + k + '=' + vv
        return sig

    def visit_ClassDef(self, node):
        self.entry = {'name': node.name,
                      'target': None}
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.entry = {'name': None,
                      'target': '__'.join([getTargetName(no)
                                           for no in node.targets])}
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.entry = {'name': None,
                      'target': getTargetName(node.target)}
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.entry = {'name': node.name,
                      'target': None}
        self.generic_visit(node)

    def generic_visit(self, node):
        if self.depth > 0:
            if not self.entry:
                self.entry = {'name': None,
                              'target': None,
                              'lineno': -1}
            if (isinstance(node, ast.stmt) or isinstance(node, ast.expr)):
                self.entry.update({'lineno': node.lineno})
            self.entry.update({'type': type(node).__name__,
                               'ast': node,
                               'hash': GetHash().visit(node),
                               'pair': None,
                               'sig': self.getSignature(node),
                               'full_match': False
                               })
            self.results.append(self.entry)
            self.entry = None

        # if isinstance(node, ast.Module):
        if self.depth == 0:
            self.depth += 1
            super(GetSubtrees, self).generic_visit(node)

    def visit(self, node):
        super(GetSubtrees, self).visit(node)
        return self.results


#
# Hewper functions for Python Diff
#


def createMap(trees, field='hash'):
    """Indexes a set of trees by slecified field"""
    resultMap = collections.defaultdict(list)
    for t in trees:
        if field in t and t[field]:
            resultMap[t[field]].append(t)
    return resultMap


def getStartLineno(subtrees, verbose=True):
    """Min and max line numbers for a set of subtrees"""
    allLineno = [t['lineno'] for t in subtrees
                 if 'lineno' in t and t['lineno'] != -1]
    if allLineno:
        return min(allLineno)
    else:
        None


def find_remaining(trees, pairs, offset=0, field='hash'):
    """Removes paired subtrees from list of trees.
    offset=0 for treeA, offset=1 for treeB""
    """
    paired = [p[offset][field] for p in pairs]
    return [t for t in trees if t[field] not in paired]


def subHashes(node):
    individualHashes = [GetHash().visit(child)
                        for child in ast.iter_child_nodes(node)]
    return set(individualHashes)

#
# Match routines for Python diff
#


def matchOnCommonality(subtreesA, subtreesB,
                       diffPairs, verbose=True, threshold=0.5):
    """Match based on number of common sub-terms. must 50% of terms
    in common"""
    if not subtreesA or not subtreesB:
        return [], subtreesA, subtreesB
    groupA = [[entry, subHashes(entry['ast'])] for entry in subtreesA]
    groupB = [[entry, subHashes(entry['ast'])] for entry in subtreesB]
    candidates = [[pA, pB, len(shA.intersection(shB))]
                  for pA, shA in groupA for pB, shB in groupB
                  if shA.intersection(shB)
                  and threshold <= (float(len(shA.intersection(shB)))
                                    / float(min(len(pA), len(pB))))]
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)

    [pairs, usedA, usedB] = findBestCandidate(candidates, verbose=verbose)
    unmatchedA = [t for t in subtreesA if t not in usedA]
    unmatchedB = [t for t in subtreesB if t not in usedB]
    diffPairs += pairs

    if verbose:
        print 'matchOnCommonality: starting ', len(subtreesA), len(subtreesB)
        print '                    ending ', len(pairs),
        print len(unmatchedA), len(unmatchedB)

    return diffPairs, unmatchedA, unmatchedB


def findBestCandidate(candidates, verbose=True):
    """Select best among a sorted set of candidates"""
    if verbose:
        print 'Candidates:'
        pprint(candidates)

    pairs = []
    usedA = []
    usedB = []
    for pA, pB, __ in candidates:
        if pA not in usedA and pB not in usedB:
            pairs.append([pA, pB])
            usedA.append(pA)
            usedB.append(pB)
    if verbose:
        print 'findBestCandidate - Pairs:'
        pprint(pairs)
    return pairs, usedA, usedB


def matchOnLineno(subtreesA, subtreesB, offset=None, verbose=True):
    """Attempts pairing based on line number proximity"""
    if verbose:
        print
        print 'calling matchOnLineno   - offset:', offset
        pprint(subtreesA)
        print '-'*10
        pprint(subtreesB)

    # try again, this tiime using line numbers
    withLinenoA = [[t['lineno'], t] for t in subtreesA
                   if 'lineno' in t and t['lineno'] != -1]
    withLinenoB = [[t['lineno'], t] for t in subtreesB
                   if 'lineno' in t and t['lineno'] != -1]

    candidates = [[tA, tB, abs(lineA - lineB - offset)]
                  for lineA, tA in withLinenoA
                  for lineB, tB in withLinenoB]
    candidates = sorted(candidates, key=lambda x: x[2])

    if verbose:
        print 'matchOnLineno: calling findBestCandidate'
    pairs, _, __ = findBestCandidate(candidates, verbose=verbose)
    if verbose:
        print 'matchOnLineno: returned'
        print 'matchOnLineno - pairs:', len(pairs)
    return pairs


def matchOnField(subtreesA, subtreesB,
                 identicalPairs, diffPairs, offset=None,
                 field='hash', verbose=True):
    """Finds sets of sub-trees that uniquely match on specified field
    returns two sets of pairs, those with matching hashes and those
    with differing pairs"""

    if not subtreesA or not subtreesB:  # skip if nothing to match
        return identicalPairs, diffPairs, subtreesA, subtreesB

    if verbose:
        print
        print 'MatchOnField:', field, len(subtreesB), len(subtreesB)
        print 'Side A:'
        pprint(subtreesA)
        print 'Side B:'
        pprint(subtreesB)

    mapA = createMap(subtreesA, field=field)
    mapB = createMap(subtreesB, field=field)
    pairs = [[mapA[k][0], mapB[k][0]]
             for k in mapA.keys()
             if k in mapB
             and len(mapA[k]) == 1 and len(mapB[k]) == 1]
    if verbose:
        print 'Before', len(pairs)
        print 'MapA'
        pprint(dict(mapA))
        print 'MapB'
        pprint(dict(mapB))

    # now handle non-unique cases, use lineno as tie-breaker
    # print 'Handle non-unique cases', offset
    if isinstance(offset, int):
        for k in mapA.keys():
            # print 'Key:', k
            if k in mapB and len(mapA[k]) > 1 and len(mapB[k]) > 1:
                # print '.'
                pairs.extend(matchOnLineno(mapA[k], mapB[k],
                                           offset=offset, verbose=verbose))
    if verbose:
        print 'After:', len(pairs)

    identicalPairs += [[pA, pB] for pA, pB in pairs
                       if pA['hash'] == pB['hash']]
    diffPairs += [[pA, pB] for pA, pB in pairs
                  if pA['hash'] != pB['hash']]

    unmatchedA = find_remaining(subtreesA, pairs, offset=0, field=field)
    unmatchedB = find_remaining(subtreesB, pairs, offset=1, field=field)

    if verbose:
        print
        print 'Results from matching on ', field
        print '    Identical pairs: ', len(identicalPairs),
        print 'Differing pairs:', len(diffPairs)
        print '    Remaining - A:', len(unmatchedA),  'B:', len(unmatchedB)

    return identicalPairs, diffPairs, unmatchedA, unmatchedB


#
# Higher level PythonDiff routines
#


def compareSubtrees(subtreesA, subtreesB, verbose=True):
    """Match two sets of trees based on commonality"""
    startA = getStartLineno(subtreesA, verbose=verbose)
    startB = getStartLineno(subtreesB, verbose=verbose)
    if startA and startB:
        offset = startA - startB
    else:
        offset = None
    # print 'Offset Value:', offset
    identicalPairs = []
    diffPairs = []

    # Initially match on name
    [identicalPairs, diffPairs,
     unmatchedA, unmatchedB] = matchOnField(subtreesA, subtreesB,
                                            identicalPairs, diffPairs,
                                            offset=offset,
                                            field='name', verbose=verbose)

    # match on assignment target
    if unmatchedA and unmatchedB:
        [identicalPairs, diffPairs,
         unmatchedA, unmatchedB] = matchOnField(unmatchedA, unmatchedB,
                                                identicalPairs, diffPairs,
                                                offset=offset,
                                                field='target',
                                                verbose=verbose)

    # match on hash
    if unmatchedA and unmatchedB:
        [identicalPairs, diffPairs,
         unmatchedA, unmatchedB] = matchOnField(unmatchedA, unmatchedB,
                                                identicalPairs, diffPairs,
                                                offset=offset,
                                                field='hash',
                                                verbose=verbose)

    # match based on number of common sub-terms. must 50% of terms in common
    if unmatchedA and unmatchedB:
        [diffPairs,
         unmatchedA, unmatchedB] = matchOnCommonality(unmatchedA, unmatchedB,
                                                      diffPairs,
                                                      threshold=0.5,
                                                      verbose=verbose)

    # match on type
    if unmatchedA and unmatchedB:
        [identicalPairs, diffPairs,
         unmatchedA, unmatchedB] = matchOnField(unmatchedA, unmatchedB,
                                                identicalPairs, diffPairs,
                                                offset=offset,
                                                field='type',
                                                verbose=verbose)

    # Annotate trees with pairing information
    for pA, pB in identicalPairs:
        pA['pair'] = pB
        pB['pair'] = pA
        pA['full_match'] = True
        pB['full_match'] = True
        pA['node_match'] = True
        pB['node_match'] = True

    for pA, pB in diffPairs:
        pA['pair'] = pB
        pB['pair'] = pA
        pA['node_match'] = (pA['sig'] == pB['sig'])
        pB['node_match'] = (pA['sig'] == pB['sig'])

    if verbose:
        print
        print 'Final result - identical pairs', len(identicalPairs),
        print 'differing pairs:', len(diffPairs)
        print '               remaining A:', len(unmatchedA),
        print 'B:', len(unmatchedB)
    if verbose:
        if unmatchedA:
            print
            print 'Remaining Side A:'
            for entry in unmatchedA:
                pprint(entry)

        if unmatchedB:
            print
            print 'Remaining Side B:'
            for entry in unmatchedB:
                pprint(entry)

        print
    return identicalPairs, diffPairs, [unmatchedA, unmatchedB]


def printSubtree(subtrees, level=0, indent=4):
    """Print hierarchical view of a subtree """
    for tree in subtrees:
        has_pair = 'P' if tree['pair'] else '-'
        if tree['full_match']:
            continue
        if 'lineno' in tree:
            print ' '*(level*indent), tree['type'], has_pair,
            print '[line', tree['lineno'], ']'
        else:
            print ' '*(level*indent), tree['type'], has_pair

        print ' '*(level*indent+2), tree['name'], tree['target'], tree['hash']
        if 'subtrees' in tree and tree['subtrees']:
            printSubtree(tree['subtrees'], level=level+1)


def recursiveDiff(subtreesA, subtreesB, verbose=True, level=0):
    """Compare next level of nodes within trees"""
    identicalPairs, diffPairs, unmatched = compareSubtrees(subtreesA,
                                                           subtreesB,
                                                           verbose=verbose)
    for pA, pB in diffPairs:
        if pA['ast'] and pB['ast']:
            # print 'Drilldown'
            pA['subtrees'] = GetSubtrees().visit(pA['ast'])
            pB['subtrees'] = GetSubtrees().visit(pB['ast'])
            recursiveDiff(pA['subtrees'], pB['subtrees'], verbose=verbose,
                          level=level+1)


def pythonDiff(nodeA, nodeB, verbose=True):
    """Tope level routine to compare two AST sub-trees"""
    subtreesA = GetSubtrees().visit(nodeA)
    subtreesB = GetSubtrees().visit(nodeB)

    recursiveDiff(subtreesA, subtreesB, verbose=verbose)
    return subtreesA, subtreesB
