#
# NewDiff.py - Language-specific change detection for Python
#
# Author:  Doug Williams - Copyright 2015
#
# Last updated 5/8/2015
#
# History:
# - 5/5/15 - Initial version of file
# - 5/8/15 - Continued active development
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


def debug_ast_strip_lineno(line):
    """Strips lineno and col_offset from ast.dump output"""
    line = re.sub(r",?\s+lineno=-?\d+", "", line)
    line = re.sub(r",?\s+col_offset=-?\d+", "", line)
    return line


def debugSanityCheckDump(node):
    """Validates that dump is behaving correctly,
    should be removed after debug"""
    if (ast.dump(node, include_attributes=False)
            != debug_ast_strip_lineno(ast.dump(node,
                                      include_attributes=True))):
        string1 = ast.dump(node, include_attributes=False)
        string2 = debug_ast_strip_lineno(ast.dump(node,
                                         include_attributes=True))
        print
        print 'compare_ast mismatch'
        print string1
        print
        print string2
        print len(string1) == len(string2)
        differences = [i for i in range(min(len(string1), len(string2)))
                       if string1[i] != string2[i]]
        print differences
        print [string1[i] for i in differences]
        print
        print [string2[i] for i in differences]
        assert False
    return True


def compare_ast(node1, node2):
    """My version of compare_ast based on ast.dump"""
    if type(node1) is not type(node2):
        return False
    if isinstance(node1, ast.AST):
        debugSanityCheckDump(node1)
        debugSanityCheckDump(node2)
        return (ast.dump(node1, include_attributes=False)
                == ast.dump(node2, include_attributes=False))

    elif isinstance(node1, list):
        return all(itertools.starmap(compare_ast,
                                     itertools.izip(node1, node2)))
    else:
        return node1 == node2


# from: http://stackoverflow.com/questions/3312989/elegant-way-to-test-python-asts-for-equality-not-reference-or-object-identity

def old_compare_ast(node1, node2, debug=True):
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


def treeViewer(tree, idxTree, depth=0, indent=4, trim=False,
               idxOther=False):
    """Displays tree and it's sub-trees, optionally prune matching sub-trees"""
    if trim and tree['mismatch'] == 0:
        if idxOther:  # Check if pair has mis-match
            if ('pair' in tree and tree['pair']
                    and idxOther[tree['pair']]['mismatch'] == 0):
                return
        else:
            return
    print ' '*depth*indent, type(tree['ast']).__name__, 'ID:', tree['idxSelf'],
    print '[', tree['start'], ',', tree['end'], ']',
    print 'Mismatch:', tree['mismatch'], 'Tokens:', len(tree['tokens']),
    print 'Insert:', tree['insert'],
    if 'pair' in tree:
        print 'Pair:', tree['pair']
    else:
        print
    if False:  # tree['tokens']:
        print
        print 'Tokens:', tree['tokens']
        print

    if 'header_mismatch' in tree and tree['header_mismatch'] > 0:
        print ' '*(depth+1)*indent, 'Header - Mismatch:',
        print tree['header_mismatch'], 'Tokens:', len(tree['header_tokens'])

    if 'subtreesIdx' in tree:
        # for t in tree['subtrees']:
        for i in tree['subtreesIdx']:
            treeViewer(idxTree[i], idxTree, depth=depth+1,
                       trim=trim, idxOther=idxOther)


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
            # print i, len(thisMatch), otherIdx, len(otherMatch)
            if (otherIdx < len(otherMatch) - 1
                    and otherMatch[otherIdx+1]):
                newToken = thisMatch[i-1] + '_insert'
                thisMatch[i-1] = newToken
                otherMatch[otherIdx] = newToken

    return thisMatch, otherMatch


def matchFlagInserts(matchA, matchB):
    """Flag tokens prior to insert"""
    tokenToLineA = {matchA[i]: i for i in range(1, len(matchA)) if matchA[i]}
    tokenToLineB = {matchB[i]: i for i in range(1, len(matchB)) if matchB[i]}

    matchA, matchB = MatchFlagInsertsHelper(matchA, matchB, tokenToLineB)
    matchB, matchA = MatchFlagInsertsHelper(matchB, matchA, tokenToLineA)

    return matchA, matchB


def matchFlagBlankTokens(match, data):
    # tag blank lines
    for i, line in enumerate(data):
        if len(line.strip()) == 0 or line.strip()[0] == '#':
            if match[i+1] and match[i+1].endswith('_insert'):
                match[i+1] = 'blank_insert'
            else:
                match[i+1] = 'blank'
    return match


def makeAllTokens(matchA, dataA, matchB, dataB):
    """Generate match tokens, identifying blank lines and inserts"""
    matchA = matchMakeTokens(matchA, sideB=False)
    matchB = matchMakeTokens(matchB, sideB=True)

    matchA, matchB = matchFlagInserts(matchA, matchB)

    matchA = matchFlagBlankTokens(matchA, dataA)
    matchB = matchFlagBlankTokens(matchB, dataB)
    return matchA, matchB


def getBlobData(blob):
    """Reads data from blob and splits lines"""
    return blob.data_stream.read().splitlines()


def parse_diff_txt(txt, a_blob, b_blob, verbose=False):
    """Parses git diff, returning line numbers containing changes.
    Per-line values in matchA and matchB:
        None => Mismatch
        -1 => blank
        int => matching lineno
    """

    dataA = getBlobData(a_blob)
    sizeA = len(dataA)

    dataB = getBlobData(b_blob)
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


def buildTree(st, end, match, text):
    """Builds nested sub-trees from Python AST - top level"""
    assert isinstance(st, ast.Module)
    treeIdx = []
    treetop = newTree(st, treeIdx, start=1, end=end)
    buildTree_helper(treetop, match, treeIdx, text)
    # pruneDetail(treetop, treeIdx)
    return treetop, treeIdx


def buildTree_helper(tree, match, treeIdx, text, verbose=False):
    """Recursively builds nested sub-trees from Python AST"""

    blankLineTrimmer(tree, text)
    tree['tokens'] = [match[i] for i in range(tree['start'], tree['end']+1)
                      if match[i] and not match[i].startswith('blank')]
    tree['mismatch'] = sum([1 for i in range(tree['start'], tree['end']+1)
                            if not match[i]])
    tree['insert'] = sum([1 for i in range(tree['start'], tree['end']+1)
                          if match[i] and match[i].endswith('_insert')])

    if type(tree['ast']) in [ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.If, ast.For, ast.While, ast.With,
                             ast.ExceptHandler]:
        # body
        subtrees = [newTree(st, treeIdx, parentTree=tree, start=st.lineno)
                    for st in tree['ast'].body]
        # handlers
        if type(tree['ast']) in [ast.TryExcept]:
            subtrees += [newTree(st, treeIdx, parentTree=tree, start=st.lineno)
                         for st in tree['ast'].handlers]
        # orelse
        if type(tree['ast']) in [ast.If, ast.For, ast.While]:
            subtrees += [newTree(st, treeIdx, parentTree=tree, start=st.lineno)
                         for st in tree['ast'].orelse]
        # finalbody
        if type(tree['ast']) in [ast.TryFinally]:
            subtrees += [newTree(st, treeIdx, parentTree=tree, start=st.lineno)
                         for st in tree['ast'].finalbody]
        #
        # Common back-end processing
        #
        all_start = [x['start'] for x in subtrees] + [tree['end'] + 1]

        for i, subtree in enumerate(subtrees):
            subtree['end'] = max(all_start[i], all_start[i+1] - 1)
            buildTree_helper(subtree, match, treeIdx, text)

        tree['subtreesIdx'] = [t['idxSelf'] for t in subtrees]

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

    if verbose:
        if type(tree['ast']) in [ast.If, ast.For, ast.With, ast.While,
                                 ast.TryFinally, ast.TryExcept,
                                 ast.ExceptHandler]:
            print 'found', type(tree['ast'])


def blankLineTrimmer(tree, text):
    """Update start and end values to eliminate blank lines"""
    # trim spaces at start
    while (tree['end'] > tree['start']
           and text[tree['start']-1].strip() == ''):
        # print 'Stripping line', tree['end'], text[tree['end']-1]
        tree['start'] += 1

    # trim spaces at end
    while (tree['end'] > tree['start']
           and text[tree['end']-1].strip() == ''):
        # print 'Stripping line', tree['start'], text[tree['start']-1]
        tree['end'] -= 1


def tokenMapper(tree, tokenMap, idxTree, side='A'):
    for token in tree['tokens']:
        if not token.startswith('blank'):
            tokenMap[token][side] = tree

    if 'subtreesIdx' in tree:
        for i in tree['subtreesIdx']:
            tokenMapper(idxTree[i], tokenMap, idxTree, side=side)


def remove_invalid_tokens(tokens, tree, idxTree):
    """Removes invalid tokens from entry at it's parents"""

    before = len(tree['tokens'])
    for tok in tokens:
        tree['tokens'].remove(tok)
    tree['mismatch'] += before - len(tree['tokens'])

    if 'header_tokens' in tree:
        before = len(tree['header_tokens'])
        for tok in tokens:
            if tok in tree['header_tokens']:
                tree['header_tokens'].remove(tok)
        tree['header_mismatch'] += before - len(tree['header_tokens'])

    if tree['idxParent'] != -1:
        remove_invalid_tokens(tokens, idxTree[tree['idxParent']], idxTree)


def cleanup_matches(tree, pairs, idxTree, otherIdxTree, tokenMap):
    """Clean-up spurious matches (ie: other values in pairs)"""
    print 'selected pair:', tree['pair']
    print 'Other candidates to be ignored'
    tokens_to_ignore = set([])
    for p in pairs:
        if p['idxSelf'] == tree['pair']:
            continue

        # Compute intersection to determine tokens and
        # header-tokens spanning thisTreee and candidate
        # and remove from ignored pair

        common_tokens = set(p['tokens']).intersection(set(tree['tokens']))
        tokens_to_ignore = tokens_to_ignore.union(common_tokens)
        print 'ignoring:', list(common_tokens)

        remove_invalid_tokens(list(common_tokens), p, otherIdxTree)

    # Now remove from this Tree and it's parents
    print 'ignoring for this tree:', list(tokens_to_ignore)
    remove_invalid_tokens(list(tokens_to_ignore), tree, idxTree)

    print 'removing from token map as well'
    for tok in tokens_to_ignore:
        del tokenMap[tok]


def computePairs(tree, tokenMap, idxTree, otherIdxTree,
                 thisSide='A', verbose=False):
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
        if verbose:
            print 'Pairing:', tree['idxSelf'], 'with', tree['pair']
        return

    # Try using substrees to also resolve instances of non-unqiueness
    # at header level
    elif len(pairs) == 0 or len(pairs) > 1:
        # print 'ID:', tree['idxSelf'],
        # print 'Currently Unmatched - mismatch count:', tree['mismatch'],
        # print 'Tokens:', len(tree['tokens'])
        # if 'subtreesIdx' in tree:
        #    # print '    Subtrees:', len(tree['subtreesIdx'])
        #    pass

        # Try to match based on children:
        if 'subtreesIdx' in tree:
            candidatePairs = []
            for i in tree['subtreesIdx']:
                if 'pair' in idxTree[i]:
                    # print 'subtree match', idxTree[i]['idxSelf'], idxTree[i]['pair']
                    # print '      parents', idxTree[i]['idxParent'],
                    # print otherIdxTree[idxTree[i]['pair']]['idxParent']
                    if (otherIdxTree[idxTree[i]['pair']]['idxParent']
                            not in candidatePairs):
                        candidatePairs.append(
                            otherIdxTree[idxTree[i]['pair']]['idxParent'])

            # print 'Candidate parents:', candidatePairs
            print 'Candidate pair count:', len(candidatePairs)
            if len(candidatePairs) == 1:
                tree['pair'] = candidatePairs[0]
                if verbose:
                    print 'Pairing:', tree['idxSelf'], 'with', tree['pair'],
                    print 'via subtree matches'
                otherIdxTree[candidatePairs[0]]['pair'] = tree['idxSelf']

                if len(pairs) > 1:
                    cleanup_matches(tree, pairs, idxTree,
                                    otherIdxTree, tokenMap)
                return

    if len(pairs) > 1:
        print 'Too many pairs', len(pairs), thisSide

        # if still not resolved, try using a majority vote among candidates
        # with compatible parents
        good_pairs = [p for p in pairs
                      if 'pair' in otherIdxTree[p['idxParent']]
                      and otherIdxTree[p['idxParent']]['pair'] ==
                      idxTree[tree['idxParent']]['idxSelf']]

        best_pair = None
        best_pair_count = -1
        for p in good_pairs:
            if len(p['tokens']) > best_pair_count:
                best_pair = p
                best_pair_count = len(p['tokens'])

        if best_pair:
            tree['pair'] = best_pair['idxSelf']
            best_pair['pair'] = tree['idxSelf']
            if verbose:
                print 'Pairing:', tree['idxSelf'], 'with', tree['pair'],
                print 'via subtree matches'
            cleanup_matches(tree, pairs, idxTree,
                            otherIdxTree, tokenMap)
        else:
            print 'Unable to identify pair'
            assert False


def okToPair(tree1, tree2):
    """Determine is on can infer pair relationship"""
    if type(tree1) == type(tree2):
        # further qualify match based on type
        # << INSERT CODE HERE >>

        tree1['pair'] = tree2['idxSelf']
        tree2['pair'] = tree1['idxSelf']
        return True
    else:
        return False


def inferPairs(tree, thisIdx, otherIdx, verbose=False):
    """Infer pairs based on neighboring pairs in subtree"""

    if ('pair' not in tree or tree['mismatch'] == 0
            or 'subtreesIdx' not in tree):
        return

    otherTree = otherIdx[tree['pair']]
    if 'subtreesIdx' not in tree or 'subtreesIdx' not in otherTree:
        return
    thisSubtrees = tree['subtreesIdx']
    otherSubtrees = otherTree['subtreesIdx']

    while len(thisSubtrees) > 0 and len(otherSubtrees) > 0:

        if 'pair' in thisIdx[thisSubtrees[0]]:
            thisSubtrees = thisSubtrees[1:]
            continue

        if 'pair' in otherIdx[otherSubtrees[0]]:
            otherSubtrees = otherSubtrees[1:]
            continue

        if 'pair' in thisIdx[thisSubtrees[-1]]:
            thisSubtrees = thisSubtrees[:-1]
            continue

        if 'pair' in otherIdx[otherSubtrees[-1]]:
            otherSubtrees = otherSubtrees[:-1]
            continue

        # see if unmatched items can be linked
        if okToPair(thisIdx[thisSubtrees[0]], otherIdx[otherSubtrees[0]]):
            if verbose:
                print 'Pair Found at start', thisSubtrees[0], otherSubtrees[0]
            thisSubtrees = thisSubtrees[1:]
            otherSubtrees = otherSubtrees[1:]
            continue

        # determine which ones to ignore (look 1 further for each)
        if okToPair(thisIdx[thisSubtrees[-1]], otherIdx[otherSubtrees[-1]]):
            if verbose:
                print 'Pair found at end', thisSubtrees[-1], otherSubtrees[-1]
            thisSubtrees = thisSubtrees[:-1]
            otherSubtrees = otherSubtrees[:-1]
            continue

        break


def pruneDetail(tree, idxTree):
    """Prune sub-trees when no matching tokens for parent"""
    if 'subtreesIdx' in tree:
        if len(tree['tokens']) == 0:
            del tree['subtreesIdx']
        else:
            for i in tree['subtreesIdx']:
                pruneDetail(idxTree[i], idxTree)


def ignoreDocstrings(tree, idxTree, verbose=False):
    """Ignore any mismatch in doc_strings"""
    if ('subtreesIdx' in tree
        and (isinstance(tree['ast'], ast.Module)
             or isinstance(tree['ast'], ast.ClassDef)
             or isinstance(tree['ast'], ast.FunctionDef))):
        firstLine = idxTree[tree['subtreesIdx'][0]]
        if (firstLine['mismatch'] > 0
                and isinstance(firstLine['ast'], ast.Expr)
                and isinstance(firstLine['ast'].value, ast.Str)):
            if verbose:
                print '    ignoring docstring', tree['idxSelf'],
                print firstLine['idxSelf']
            tree['mismatch'] -= firstLine['mismatch']
            firstLine['mismatch'] = 0
    if 'subtreesIdx' in tree:
        for i in tree['subtreesIdx']:
            ignoreDocstrings(idxTree[i], idxTree)


def validateMismatches(tree, thisIdx, otherIdx, verbose=False):
    """Depth first check of mismatch pairs"""

    if 'pair' not in tree:
        return

    old_mismatch = tree['mismatch']
    new_mismatch = old_mismatch

    if 'subtreesIdx' in tree:
        if 'header_mismatch' in tree:
            new_mismatch = tree['header_mismatch']
        else:
            new_mismatch = 0
        for i in tree['subtreesIdx']:
            validateMismatches(thisIdx[i], thisIdx, otherIdx)
        # recompute mismatch count
        new_mismatch += sum([thisIdx[i]['mismatch']
                             for i in tree['subtreesIdx']])

    # Now compare this node
    otherTree = otherIdx[tree['pair']]

    if old_mismatch != new_mismatch:
        if verbose:
            print '    Updating mismatches for:', tree['idxSelf'],
            print 'was:', old_mismatch, 'now:', new_mismatch
        tree['mismatch'] = new_mismatch
    if old_mismatch > 0 and compare_ast(tree['ast'], otherTree['ast']):
        if verbose:
            print '-'*40
            print 'Match found'
            print ast.dump(tree['ast'], include_attributes=False)
            print
            print ast.dump(otherTree['ast'], include_attributes=False)

        tree['mismatch'] = 0
        if 'header_mismatch' in tree:
            tree['header_mismatch'] = 0
        if verbose:
            print '    Match:', tree['idxSelf'], otherTree['idxSelf']


def performDiff(d, verbose=False):
    """Perform diff operation on individual file"""
    if not d.b_blob or not d.b_blob.path.endswith('.py'):
        print 'Error:  Invalid blob for performDiff', d.b_blob
        assert False
    if verbose:
        print
        print '+'*60
        print
        print 'Comparing ', d.b_blob.path
        if d.a_blob.path != d.b_blob.path:
            print '    With', d.b_blob.path

    matchA, matchB = parse_diff_txt(d.diff, d.a_blob, d.b_blob)

    st_a = get_st_from_blob(d.a_blob)
    treeA, idxA = buildTree(st_a, len(matchA) - 1, matchA,
                            getBlobData(d.a_blob))

    st_b = get_st_from_blob(d.b_blob)
    treeB, idxB = buildTree(st_b, len(matchB) - 1, matchB,
                            getBlobData(d.b_blob))

    if verbose:
        print
        print 'Token Mapper'
    tokenMap = collections.defaultdict(dict)
    tokenMapper(treeA, tokenMap, idxA, side='A')
    tokenMapper(treeB, tokenMap, idxB, side='B')

    if verbose:
        print
        print '***Tree A ***'
        treeViewer(treeA, idxA, trim=True, idxOther=idxB)
        print '*'*40
        print
        print '***Tree B ***'
        treeViewer(treeB, idxB, trim=True, idxOther=idxA)

    if verbose:
        print 'Compute pairings:'
    computePairs(treeA, tokenMap, idxA, idxB, thisSide='A')
    if verbose:
        print '-'*20
    computePairs(treeB, tokenMap, idxB, idxA, thisSide='B')

    if verbose:
        print
        print 'Infer additional pairings:'
    for tree in idxA:
        inferPairs(tree, idxA, idxB)

    if verbose:
        print
        print 'Ignore Docstrings:'
    ignoreDocstrings(treeA, idxA)
    ignoreDocstrings(treeB, idxB)

    # pruneDetail(treeA, idxA)
    # pruneDetail(treeB, idxB)

    if verbose:
        print
        print '***Tree A ***'
        treeViewer(treeA, idxA, trim=False, idxOther=idxB)
        print '*'*40
        print
        print '***Tree B ***'
        treeViewer(treeB, idxB, trim=False, idxOther=idxA)

    if verbose:
        print
        print 'Comparing Pairs:'
        print '  Side A:'
    validateMismatches(treeA, idxA, idxB)
    if verbose:
        print '  Side B:'
    validateMismatches(treeB, idxB, idxA)

    if verbose:
        print
        print '***Tree A ***'
        treeViewer(treeA, idxA, trim=True, idxOther=idxB)
        print '*'*40
        print
        print '***Tree B ***'
        treeViewer(treeB, idxB, trim=True, idxOther=idxA)

    if verbose:
        print
        print '***Tree A ***'
        treeViewer(treeA, idxA, trim=False)
        print '*'*40
        print
        print '***Tree B ***'
        treeViewer(treeB, idxB, trim=False)
