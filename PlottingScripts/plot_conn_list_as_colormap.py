import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import matplotlib
#matplotlib.use('Agg')
import pylab
import numpy as np 
import utils
import json
import simulation_parameters

def load_params_from_folder(folder_name):
    param_fn = folder_name + '/Parameters/simulation_parameters.json'
    f = file(param_fn, 'r')
    print 'Loading parameters from', param_fn
    params = json.load(f)
    return params


def plot_matrix(d, title=None):
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    print "plotting .... "
    cax = ax.pcolormesh(d, cmap='bwr')
    ax.set_xlim((0, n_tgt))
    ax.set_ylim((0, n_src))
    if title != None:
        ax.set_title(title)
    pylab.colorbar(cax)
    pylab.savefig(output_fn)
    output_fn2 = output_fn.rsplit(".dat")[0] + ".png"
    pylab.savefig(output_fn2)

    print "Output file:", output_fn


def is_it_d1(fn):
    if fn.find('d1') != -1:
        return True
    else:
        return False



if __name__ == '__main__':

    conn_list_fn = sys.argv[1]
    folder = conn_list_fn.rsplit('/')[0]
    params = load_params_from_folder(folder)

    if not os.path.exists(conn_list_fn):
        print 'Merging default connection files...'
        merge_pattern = params['mpn_bgd1_conn_fn_base']
        fn_out = params['mpn_bgd1_merged_conn_fn']
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)
        merge_pattern = params['mpn_bgd2_conn_fn_base']
        fn_out = params['mpn_bgd2_merged_conn_fn']
        utils.merge_and_sort_files(merge_pattern, fn_out, sort=False)
    print 'Loading ', conn_list_fn
    data = np.loadtxt(conn_list_fn)

    is_it_d1 = is_it_d1(conn_list_fn)
    if is_it_d1:
        tgt_cell_type = 'D1'
    else:
        tgt_cell_type = 'D2'
    src_cell_type = 'mpn'

    if src_cell_type == 'mpn':
        n_src = params['n_exc_mpn']
        src_offset = 1
    else:
        src_offset = data[:, 0].min()

    if tgt_cell_type == 'D1' or tgt_cell_type == 'D2':
        n_tgt = params['n_cells_%s' % tgt_cell_type]
        cell_gid_fn = params['parameters_folder'] + 'bg_cell_gids_pcid0.json'
        f = file(cell_gid_fn, 'r')
        p = json.load(f)
        tgt_offset = p['str%s' % tgt_cell_type][0][0]
    else:
        tgt_offset = data[:, 1].min()

    print 'n_src', n_src
    print 'n_tgt', n_tgt
    print "src offset", src_offset
    print "tgt offset", tgt_offset


    conn_mat = np.zeros((n_src, n_tgt))

    for c in xrange(data[:,0].size):
        src = data[c, 0] - src_offset
        tgt = data[c, 1] - tgt_offset
#        print 'debug', src, tgt, c
        conn_mat[src, tgt] = data[c, 2]

    output_fn = conn_list_fn.rsplit(".")[0] + "_cmap.png"
    plot_matrix(conn_mat, title=conn_list_fn)
    pylab.show()
