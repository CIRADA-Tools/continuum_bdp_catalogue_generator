import re
import glob
import numpy as np
from astropy.table import Table

def get_subtile_name(subtile_filename):
    return re.sub(r"^(.*?/)*.*?(J.*?)\..*$",r"\2",subtile_filename)

def get_subtile_version(subtile_filename):
    return int(re.sub(r"^(.*?/)*.*?v(\d+).*$",r"\2",subtile_filename))

def fetch_duplicates(manifest_csv_file):
    df = Table.read(manifest_csv_file) # get the url subtile manifest
    u,c = np.unique(np.array([get_subtile_name(f) for f in df['file']]),return_counts=True) # get version counts
    dups = u[c>1] # duplicates
    # sort duplicate urls in ascending order of version number and extract fits file names
    ds = [re.sub(r"^(.*?/)*","",u) for u in np.sort(df[([get_subtile_name(f) in dups for f in df['file']])])['file']]
    # return {subtile_name <=> latest_version} key-value pairs
    return {get_subtile_name(f):get_subtile_version(f) for f in ds} # works beacause of pre-sorting

class DuplicatesChecker:
    def __init__(self,manifest_csv):
        # keep a {subtile_name: latest_version} dict of duplicate subtiles
        self.dups = fetch_duplicates(manifest_csv)

    def get_dups(self):
        # return dict of {duplicate_subtile_name: latest_version}
        return self.dups

    def is_dup(self,subtile_filename):
        name = get_subtile_name(subtile_filename) # get subtile name
        return name in self.dups.keys() # check if in duplicates dict
        
    def is_keep(self,subtile_filename):
        name    = get_subtile_name(subtile_filename)
        version = get_subtile_version(subtile_filename)
        if name in self.dups.keys() and version != self.dups[name]:
            # ok, this is a duplicate subtile, but not the latest version
            return False # so don't keep
        return True # keep the rest

