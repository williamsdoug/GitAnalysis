# Generic Dumper/Loader Routines
#
# Author:  Doug Williams - Copyright 2014, 2015
#
# Last updated 2/13/2014
#
# History:
# - 1/25/15: Added history tracking, PEP-8 clean-up
# - 2/13/15: Fixed handling of dict entries with False value

import cPickle as pickle
import gzip
import json

# from jp_load_dump import convert_to_builtin_type, pload, pdump, jload, jdump


def convert_to_builtin_type(obj):
    try:
        d = repr(obj)
    except Exception, e:
        if str(type(obj)) == "<class 'git.util.Actor'>":
            d = ('<git.Actor "' + obj.name.encode('ascii', 'ignore') +
                 ' <' + obj.email + '>">')
            # d = '<git.Actor "'+ str(obj.name.encode('ascii','ignore'))
            #     +' <'+ obj.email+'>">'
        else:
            print 'unexpected type:', type(obj)
            print unicode(obj)
            raise Exception
    return d


def jdump_helper(x, f, item_per_line=True):
    skipped = 0
    # store the object
    if item_per_line and type(x) == list:
        # print type(x), 'is list'
        for v in x:
            try:
                f.write(json.dumps(v, default=convert_to_builtin_type))
                f.write("\n")
            except Exception, e:
                skipped += 1
                # print 'Error in json dumps', e
                # pp.pprint(v)
                # raise Exception
    elif item_per_line and type(x) == dict:
        # print type(x), 'is dict'
        for k, v in x.items():
            try:
                if not v:
                    v = {'json_tombstone': 1}
                v['json_key'] = k
                f.write(json.dumps(v, default=convert_to_builtin_type))
                f.write("\n")
            except Exception, e:
                skipped += 1
                # print 'Error in json dumps', e, 'key:', k
                # pp.pprint(v)
                # raise Exception
    else:
        json.dump(x, f, default=convert_to_builtin_type)

    if skipped > 0:
        print 'entried skipped:', skipped


def jload_helper(f):
    # restore the object
    result = []
    dict_result = {}
    for line in f:
            v = json.loads(line)
            if 'json_key' in v:
                # individual entries are items in a dict:
                k = v['json_key']
                del v['json_key']
                if len(v) == 1 and 'json_tombstone' in v:
                    dict_result[k] = False
                else:
                    dict_result[k] = v
                pass
            else:
                result.append(v)
    if dict_result:
        # print 'returning dict'
        return dict_result
    elif len(result) > 1:
        # print 'returning list'
        return result
    elif len(result) == 1:
        print 'returning singleton'
        return result[0]
    else:
        return False


def jdump(x, name, item_per_line=True):
    if name.endswith('z'):
        # print 'gzipped'
        with gzip.open(name, 'wb') as f:
            jdump_helper(x, f, item_per_line=item_per_line)
    else:
        with open(name, 'wb') as f:
            jdump_helper(x, f, item_per_line=item_per_line)


def jload(name):
    if name.endswith('z'):
        # print 'gzipped'
        with gzip.open(name, 'rb') as f:
            return jload_helper(f)
    else:
        with open(name, 'rb') as f:
            return jload_helper(f)


def pdump(x, name):
    if name.endswith('z'):
        with gzip.open(name, 'wb') as f:
            pickle.dump(x, f, protocol=-1)
    else:
        with open(name, 'wb') as f:
            pickle.dump(x, f, protocol=-1)


def pload(name):
    if name.endswith('z'):
        with gzip.open(name, 'rb') as f:
            return pickle.load(f)
    else:
        with open(name, 'rb') as f:
            return pickle.load(f)
