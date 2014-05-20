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
    w_in_sum = 0.
    w_in_mean = np.zeros(params['num_msn_d1'])
    w_in_mean_sum = np.zeros(params['num_msn_d1'])
    for i_, gid in enumerate(post_gids):
        sources = utils.get_sources(conn_data, gid)
        all_source_gids += sources[:, 0].tolist()
        w_in_mean[i_] = sources[:, 2].mean()
        w_in_mean_sum[i_] = sources[:, 2].sum()
        w_in_sum += sources[:, 2].sum()
        print 'w_in_mean[%d]: %.3f +- %.3f' % (gid, w_in_mean[i_], sources[:, 2].std())
    print 'w_in_action[%d]: sum %.3f mean_sum %.3f +- %.3f' % (action_idx, w_in_sum, w_in_mean_sum.mean(), w_in_mean_sum.std())
#    print 'All sources to action %d\n' % (action_idx), np.unique(all_source_gids)


def get_target_action(d, gid):
    idx = (d[:, 0] == gid).nonzero()[0]
    w = d[idx, 2]
    w_idx_sorted = w.argsort()
    nw = 5
    diff_wmax = w[w_idx_sorted[-1]] - w[w_idx_sorted[-2]]
    print 'w:', gid, w[w_idx_sorted[-nw:]], w_idx_sorted[-nw:], diff_wmax



if __name__ == '__main__':
    params = utils.load_params( os.path.abspath(sys.argv[1]) )

    conn_fn = params['mpn_bgd1_merged_conn_fn']
    f_bg = file(params['bg_gids_fn'], 'r')
    bg_cell_gids = json.load(f_bg)
    d = np.loadtxt(conn_fn)
    src_gids = np.unique(d[:, 0])
    
    for i_src, gid in enumerate(src_gids):
        get_target_action(d, gid)
#        if i_src == 5:
#            exit(1)

#    for action_idx in xrange(params['n_actions']):
#        get_sources(params, d, bg_cell_gids, action_idx) 

#    DB = DebugTraces()
#    DB.check_spike_files(params)


