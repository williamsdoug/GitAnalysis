#
# NewDiff.py - Language-specific change detection for Python
#
# Author:  Doug Williams - Copyright 2015
#
# Last updated 5/5/2015
#
# History:
# - 5/5/15 - Initial version of file
#
#
# Top Level Routines:
#
# from NewDiff import process_commit_diff
#


# Yet Another attempt at python diff
#
# Approach:
# - use a technique like hashes to primarily resolve pairing
#  - can use diff for the same
# - Establish statement level affinity based on:
#  - lines in common
#  - proximity to common lines
#
# Steps:
#  1. Parse Each tree [done]
#   - Also associate line ranges
#  2. Identify ranges for each entry [done]
#   - If all entries has pair, then discard
#    - intersection of tokens == union of tokens
#   - If differences, then refine further
#  3. Annotate pairings [partial]
#   - depth-first, associate tokens with sub-trees [partial]
#    - match trees where sub-trees matched (may currently miss
#      if insufficient data in header) [to do]
#   - verify uniqueness
#  4. check for spurrious mis-matches using compare_ast [to do]
#   - spurious mis-matches may be use to comments
#
# Outputs:
#  - Sparse sub-tree of differences
#   - for use in complexity calculation for feature extraction
#  - List of line numbers
#   - for use in blame calculation)
#
#
# ###Proposed Changes to existing code:
#
# 1) associate unique ID with each subtree, and create
#    table mapping ID to subtree  [DONE]
#
# 2) annotate trees [DONE]
#   - Include ID in each subtree
#   - Parent ID in each child
#
# 3) for each partial match, use ast_compare to verify if false mismatch
#   - if so, promote tokens to parent and remove matching subtrees from
#     subtrees list
# - also remember to update token map for each token moved to parent
#
# 4) For unmatched nodes with matching subtrees, use parent of matched nodes
#
# 5). We man need to drill-down below function first level statements.
#     If so, do so only after matching functions,  and do so recursively
#     in pairs.
#

import ast
from pprint import pprint
import collections
import re
import itertools

# sys.path.append('./dev')
# from git_analysis_config import get_repo_name


# from: http://stackoverflow.com/questions/3312989/elegant-way-to-test-python-asts-for-equality-not-reference-or-object-identity

def compare_ast(node1, node2):
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        for k, v in vars(node1).iteritems():
            if k in ('lineno', 'col_offset', 'ctx'):
                continue
            if not compare_ast(v, getattr(node2, k)):
                return False
        return True
    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast,
                                     itertools.izip(node1, node2)))
    else:
        return node1 == node2


def get_st_from_blob(blob, verbose=False):
    """Extracts Syntax Tree (AST) from git blob"""

    st = ast.parse(blob.data_stream.read(), filename=blob.path)
    return st


def get_lines_from_blob(blob):
    """Extracts line count from blob"""

    data = blob.data_stream.read()
    return len(data.splitlines())


minus_re = re.compile('\-(\d+)')
plus_re = re.compile('\+(\d+)')


def matchMakeTokens(match, sideB=False):
    """Convert entries to standard tokens for comparison"""
    match[0] = 'space'

    for i, val in enumerate(match):
        if isinstance(val, int):
            if sideB:
                match[i] = 'A' + str(val) + '_B' + str(i)
            else:
                match[i] = 'A' + str(i) + '_B' + str(val)
    return match


def MatchFlagInsertsHelper(thisMatch, otherMatch, tokenToOtherLine):
    """Flag tokens prior to insert for a single side"""
    for i in range(2, len(thisMatch)):
        # detect line before insert
        if (not thisMatch[i] and thisMatch[i-1]
                and not thisMatch[i-1].endswith('_insert')):
            otherIdx = tokenToOtherLine[thisMatch[i-1]]
            if otherMatch[otherIdx+1]:
                newToken = thisMatch[i-1] + '_insert'
                thisMatch[i-1] = newToken
                otherMatch[otherIdx] = newToken

    return thisMatch, otherMatch


def MatchFlagInserts(matchA, matchB):
    """Flag tokens prior to insert"""
    tokenToLineA = {matchA[i]: i for i in range(1, len(matchA)) if matchA[i]}
    tokenToLineB = {matchB[i]: i for i in range(1, len(matchB)) if matchB[i]}

    matchA, matchB = MatchFlagInsertsHelper(matchA, matchB, tokenToLineB)
    matchB, matchA = MatchFlagInsertsHelper(matchB, matchA, tokenToLineA)

    return matchA, matchB


def matchFlagBlankTokens(match, data):
    # tag blank lines
    for i, line in enumerate(data):
        if len(line.strip()) == 0:
            if match[i+1] and match[i+1].endswith('_insert'):
                match[i+1] = 'blank_insert'
            else:
                match[i+1] = 'blank'
    return match


def makeAllTokens(matchA, dataA, matchB, dataB):
    """Generate match tokens, identifying blank lines and inserts"""
    matchA = matchMakeTokens(matchA, sideB=False)
    matchB = matchMakeTokens(matchB, sideB=True)

    matchA, matchB = MatchFlagInserts(matchA, matchB)

    matchA = matchFlagBlankTokens(matchA, dataA)
    matchB = matchFlagBlankTokens(matchB, dataB)
    return matchA, matchB


def old_matchMakeTokens(match, data, sideB=False):
    """Convert entries to standard tokens for comparison"""
    match[0] = 'space'
    # tag blank lines
    for i, line in enumerate(data):
        if len(line.strip()) == 0:
            match[i+1] = 'blank'
    for i, val in enumerate(match):
        if isinstance(val, int):
            if sideB:
                match[i] = 'A' + str(val) + '_B' + str(i)
            else:
                match[i] = 'A' + str(i) + '_B' + str(val)
    return match


def parse_diff_txt(txt, a_blob, b_blob, verbose=False):
    """Parses git diff, returning line numbers containing changes.
    Per-line values in matchA and matchB:
        None => Mismatch
        -1 => blank
        int => matching lineno
    """

    dataA = a_blob.data_stream.read().splitlines()
    sizeA = len(dataA)

    dataB = b_blob.data_stream.read().splitlines()
    sizeB = len(dataB)

    # 1 based indexing, ignore element 0
    matchA = [None] * (sizeA + 1)
    matchB = [None] * (sizeB + 1)

    lineA = -1
    lineB = -1

    curA = 1
    curB = 1

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

            while curA < lineA:
                matchA[curA] = curB
                matchB[curB] = curA
                curA += 1
                curB += 1
            continue
        elif line.startswith('---'):
            continue
        elif line.startswith('+++'):
            continue
        elif line.startswith('-'):
            changesA.append(lineA)
            lineA += 1
            curA += 1
        elif line.startswith('+'):
            changesB.append(lineB)
            lineB += 1
            curB += 1
        elif line.startswith(' '):
            if verbose:
                print 'A/B', line
            matchA[curA] = curB
            matchB[curB] = curA
            curA += 1
            curB += 1
            lineA += 1
            lineB += 1

    while curA < len(matchA):
            matchA[curA] = curB
            matchB[curB] = curA
            curA += 1
            curB += 1

    computes_changes = [i for i, v in enumerate(matchA) if not v and i != 0]
    if (set(changesA).difference(set(computes_changes))
            or set(computes_changes).difference(set(changesA))):
        print 'Mismatch A!'

    computes_changes = [i for i, v in enumerate(matchB) if not v and i != 0]
    if (set(changesB).difference(set(computes_changes))
            or set(computes_changes).difference(set(changesB))):
        print 'Mismatch B!'

    # now strip out whitespace
    matchA,  matchB = makeAllTokens(matchA, dataA, matchB, dataB)
    return matchA, matchB


def newTree(st, treeIdx, parentTree=None, start=None, end=None):
    """Creates new subtree and inserts into index"""
    result = {'ast': st, 'idxSelf': len(treeIdx)}
    treeIdx.append(result)

    if start:
        result['start'] = start
    if end:
        result['end'] = end
    if parentTree:
        result['idxParent'] = parentTree['idxSelf']
    else:
        result['idxParent'] = -1
    return result


def pruneAST(st, end, match):
    """Removes matching elements from tree"""
    assert isinstance(st, ast.Module)
    treeIdx = []
    treetop = newTree(st, treeIdx, start=1, end=end)
    pruneAST_helper(treetop, match, treeIdx)
    # pruneDetail(treetop, treeIdx)
    return treetop, treeIdx


def pruneAST_helper(tree, match, treeIdx):
    tree['tokens'] = [match[i] for i in range(tree['start'], tree['end']+1)
                      if match[i] and not match[i].startswith('blank')]
    tree['mismatch'] = sum([1 for i in range(tree['start'], tree['end']+1)
                            if not match[i]])
    tree['insert'] = sum([1 for i in range(tree['start'], tree['end']+1)
                          if match[i] and match[i].endswith('_insert')])

    if ((tree['mismatch'] > 0 or tree['insert'] > 0)
        and (isinstance(tree['ast'], ast.Module)
             or isinstance(tree['ast'], ast.ClassDef)
             or isinstance(tree['ast'], ast.FunctionDef))):
        subtrees = [newTree(st, treeIdx, parentTree=tree, start=st.lineno)
                    for st in tree['ast'].body]
        all_start = [x['start'] for x in subtrees] + [tree['end']]

        for i, subtree in enumerate(subtrees):
            subtree['end'] = max(all_start[i], all_start[i+1] - 1)
            pruneAST_helper(subtree, match, treeIdx)

        # tree['subtrees'] = [t for t in subtrees if t['mismatch'] > 0]

        tree['subtreesIdx'] = [t['idxSelf'] for t in subtrees
                               if t['mismatch'] > 0]

        # now compute header:
        firstSubtreeLineno = min([t['start'] for t in subtrees])
        tree['header_tokens'] = [match[i]
                                 for i in range(tree['start'],
                                                firstSubtreeLineno)
                                 if match[i]]
        tree['header_mismatch'] = sum([1
                                       for i in range(tree['start'],
                                                      firstSubtreeLineno)
                                       if not match[i]])


def tokenMapper(tree, tokenMap, idxTree, side='A'):
    for token in tree['tokens']:
        if not token.startswith('blank'):
            tokenMap[token][side] = tree

    if 'subtreesIdx' in tree:
        for i in tree['subtreesIdx']:
            tokenMapper(idxTree[i], tokenMap, idxTree, side=side)


def treeViewer(tree, idxTree, depth=0, indent=4):
    print ' '*depth*indent, type(tree['ast']).__name__, 'ID:', tree['idxSelf'],
    print '[', tree['start'], ',', tree['end'], ']',
    print 'Mismatch:', tree['mismatch'], 'Tokens:', len(tree['tokens']),
    if 'pair' in tree:
        print 'Pair:', tree['pair']
    else:
        print
    if 'header_mismatch' in tree and tree['header_mismatch'] > 0:
        print ' '*(depth+1)*indent, 'Header - Mismatch:',
        print tree['header_mismatch'], 'Tokens:', len(tree['header_tokens'])

    if 'subtreesIdx' in tree:
        # for t in tree['subtrees']:
        for i in tree['subtreesIdx']:
            treeViewer(idxTree[i], idxTree, depth=depth+1)


def computePairs(tree, tokenMap, idxTree, otherIdxTree, thisSide='A'):
    pairs = []

    if thisSide == 'A':
        otherSide = 'B'
    else:
        otherSide = 'A'

    if 'subtreesIdx' in tree:
        tokens = tree['header_tokens']

        for i in tree['subtreesIdx']:
            computePairs(idxTree[i], tokenMap, idxTree, otherIdxTree,
                         thisSide=thisSide)
    else:
        tokens = tree['tokens']

    for tok in tokens:
        if not tok.startswith('blank'):
            this = tokenMap[tok][thisSide]
            match = tokenMap[tok][otherSide]
            if this == tree and match not in pairs:
                pairs.append(match)
            elif this != tree:
                print 'Skipping', tok

    if len(pairs) == 1:
        tree['pair'] = pairs[0]['idxSelf']
        print 'Pairing:', tree['idxSelf'], 'with', tree['pair']

    elif len(pairs) == 0:
        print 'ID:', tree['idxSelf'],
        print 'Currently Unmatched - mismatch count:', tree['mismatch'],
        print 'Tokens:', len(tree['tokens'])
        if 'subtreesIdx' in tree:
            print '    Subtrees:', len(tree['subtreesIdx'])

        # Try to match based on children:
        if 'subtreesIdx' in tree:
            for i in tree['subtreesIdx']:
                if 'pair' in idxTree[i]:
                    print 'subtree match',
                    pprint(otherIdxTree[idxTree[i]])
                    assert False

    else:
        print 'Too many pairs', len(pairs), thisSide

        print
        print '***Tree ***'
        treeViewer(tree, idxTree)
        print '*'*40

        print 'This tree:', tree['idxSelf']
        pprint(tree)
        print
        print 'This tree:', tree['idxSelf']
        print 'Pairs'
        for p in pairs:
            print '----'
            print 'candidate:', p['idxSelf'], 'parent:', p['idxParent'],
            print 'start:', p['start'], 'end:', p['end']
            print 'pair', p['pair']
            print len(p['header_tokens']), len(p['tokens'])
        assert False


def pruneDetail(tree, idxTree):
    """Prune sub-trees when no matching tokens for parent"""
    if 'subtreesIdx' in tree:
        if len(tree['tokens']) == 0:
            del tree['subtreesIdx']
        else:
            for i in tree['subtreesIdx']:
                pruneDetail(idxTree[i], idxTree)


def performDiff(d):
    """Perform diff operation on individual file"""
    if not d.b_blob or not d.b_blob.path.endswith('.py'):
        print 'Error:  Invalid blob for performDiff', d.b_blob
        assert False
    print
    print '+'*60
    print
    print 'Comparing ', d.b_blob.path
    if d.a_blob.path != d.b_blob.path:
        print '    With', d.b_blob.path

    matchA, matchB = parse_diff_txt(d.diff, d.a_blob, d.b_blob)

    st_a = get_st_from_blob(d.a_blob)
    treeA, idxA = pruneAST(st_a, len(matchA) - 1, matchA)
    if False:
        print
        print '***Tree A ***'
        treeViewer(treeA, idxA)
        print '*'*40
        # pprint(treeA)

    st_b = get_st_from_blob(d.b_blob)
    treeB, idxB = pruneAST(st_b, len(matchB) - 1, matchB)
    if False:
        print
        print '***Tree B ***'
        treeViewer(treeB, idxB)

    print
    print 'Token Mapper'
    tokenMap = collections.defaultdict(dict)
    tokenMapper(treeA, tokenMap, idxA, side='A')
    tokenMapper(treeB, tokenMap, idxB, side='B')

    print 'Compute pairings:'
    computePairs(treeA, tokenMap, idxA, idxB, thisSide='A')
    print '-'*20
    computePairs(treeB, tokenMap, idxB, idxA, thisSide='B')

    pruneDetail(treeA, idxA)
    pruneDetail(treeB, idxB)

    print
    print '***Tree A ***'
    treeViewer(treeA, idxA)
    print '*'*40
    print
    print '***Tree B ***'
    treeViewer(treeB, idxB)
