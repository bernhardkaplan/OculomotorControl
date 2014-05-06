import os
import sys
import json
import numpy as np
import utils
import pylab
from PlottingScripts import FigureCreator
from debug_bcpnn_testing import DebugTraces


def get_sources(params, conn_data, bg_cell_gids, action_idx):

    post_gids = bg_cell_gids['d1'][action_idx]
    all_source_gids = []
    for gid in post_gids:
        sources = utils.get_sources(conn_data, gid)
        all_source_gids += sources[:, 0].tolist()
#        print gid, sources
    print 'All sources to action %d\n' % (action_idx), np.unique(all_source_gids)

if __name__ == '__main__':
    params = utils.load_params( os.path.abspath(sys.argv[1]) )

    conn_fn = params['mpn_bgd1_merged_conn_fn']
    f_bg = file(params['bg_gids_fn'], 'r')
    bg_cell_gids = json.load(f_bg)
    d = np.loadtxt(conn_fn)
    
    for action_idx in xrange(params['n_actions']):
        get_sources(params, d, bg_cell_gids, action_idx) 

#    DB = DebugTraces()
#    DB.check_spike_files(params)


