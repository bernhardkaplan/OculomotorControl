import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import numpy as np
import json
import utils
import pylab
from PlottingScripts.PlotMPNActivity import ActivityPlotter


def load_gids(params):
    f = file(params['bg_gids_fn'], 'r')
    bg_gids = json.load(f)
    f = file(params['mpn_gids_fn'], 'r')
    mpn_gids = json.load(f)
    return mpn_gids, bg_gids


def plot_conn_mat(d):
    import plot_conn_list_as_colormap as pc
    pc.plot_matrix(d)
    pylab.show()

def load_d1_spikes(params):
    fn_merged = params['bg_spikes_fn_merged']
    print 


#folder = 'Training_ITERATIVELY_1.00e-01_nRF8_nV6_clipWeights1-1_nStim6x48_it25_nactions21_blur0.10_tsim28800_taup14400'
folder = sys.argv[1]
params = utils.load_params(folder)

mpn_gids, bg_gids = load_gids(params)
src_gids = mpn_gids['exc']
tgt_gids_ = bg_gids['d1']
#print 
tgt_gids = np.reshape(tgt_gids_, (params['n_cells_d1'], 1))
#print 'src_gids', src_gids
#print 'tgt_gids', tgt_gids

conn_fn = params['mpn_bgd1_merged_conn_fn']
print 'Loading connections from:', conn_fn
d = np.loadtxt(conn_fn)
conn_mat = utils.convert_connlist_to_matrix(d, src_min=np.min(src_gids), src_max=np.max(src_gids), tgt_min=np.min(tgt_gids), tgt_max=np.max(tgt_gids))


#exit(1)

stim_range = (0, 1)
n_stim = stim_range[1] - stim_range[0]
it_max = n_stim * params['n_iterations_per_stim']
AP = ActivityPlotter(params, it_max)
nspikes = AP.bin_spiketimes()

print 'nspikes shape', nspikes.shape
print 'conn_mat shape', conn_mat.shape

exit(1)

w_out = {}
for src_gid in xrange(1, 1 + params['n_exc_mpn']):
    w_out[src_gid] = utils.get_targets(d, src_gid)[:, 1:]


# w_out[1] - w_out[2]
print w_out[1][:, 1].mean()
diff = w_out[1] - w_out[2]
print diff[:, 1].mean(), diff[:, 1].min(), diff[:, 1].max()

"""

pylab.show()
"""
