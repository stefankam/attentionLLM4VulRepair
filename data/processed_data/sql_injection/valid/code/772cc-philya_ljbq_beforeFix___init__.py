
import hashlib
import os

import pandas as pd


def query_hash(project_id, query_name, **query_params):

    id_string = "{}/{}?".format(project_id, query_name)
     
    keylist = sorted(query_params.keys())

    for key in keylist:
        id_string += "{}={}&".format(key, query_params[key])

    return hashlib.sha224(id_string.encode('utf8')).hexdigest()

    
def get_result(project_id, query_name, query_params={}, query_dir='bqsql', cache_dir='bqcache', reload=False):

    # compute file name and params hash
    qhash = query_hash(project_id, query_name, **query_params)

    # check if hash.pkl file exists or reload
    cache_file_name = os.path.join(cache_dir, "{}.pkl".format(qhash))

    if not reload and os.path.exists(cache_file_name):
        res = pd.read_pickle(cache_file_name)
    else:

        # read query from file
        query_fn = os.path.join(query_dir, "{}.sql".format(query_name))
        with open(query_fn, 'r') as query_f:
            query_templ = query_f.read()

        # substitute parameters
        <vul/>query_str = query_templ.format(**query_params)</vul>

        res = pd.io.gbq.read_gbq(query_str, project_id=project_id, dialect="standard")

        os.makedirs(cache_dir, exist_ok=True)
        res.to_pickle(cache_file_name)

    return res