"""
This file contains a bunch of helper functions.
"""

import numpy as np
import os
import re
import json
import simulation_parameters
import MergeSpikefiles
import scipy.stats as stats
import random
import string

def draw_from_discrete_distribution(prob_dist, size=1):
    """
    prob_dist -- array containing probabilities

    E.g.
    xk = np.arange(7)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1)
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=100)
    """
    xk = np.arange(prob_dist.size)
    custm = stats.rv_discrete(name='custm', values=(xk, prob_dist))
    R = custm.rvs(size=size)
    return R


def get_next_stim(params, stim_params, v_eye):
    """
    Returns the stimulus parameters for a given action (v_eye) in x-direction
    """
    x_stim = stim_params[0] - (stim_params[2] + v_eye) * params['t_iteration'] / params['t_cross_visual_field']
    return (x_stim, stim_params[1], stim_params[2], stim_params[3])


def distance(x,y):   
    # alternative: np.linalg.norm(a-b)
    return np.sqrt(np.sum((x - y)**2))

def remove_empty_files(folder):
    for fn in os.listdir(folder):
        path = os.path.abspath(folder) + '/%s' % fn
        file_size = os.path.getsize(path)
        if file_size == 0:
            rm_cmd = 'rm %s' % (path)
#            print 'utils.remove_empty_files: ', rm_cmd
            os.system(rm_cmd)

def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


def save_spike_trains(params, iteration, stim_list, gid_list):
    assert (len(stim_list) == len(gid_list))
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_, nest_gid in enumerate(gid_list):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, nest_gid - 1)
            np.savetxt(fn, stim_list[i_])


def map_gid_to_action(gid, bg_cell_gids, celltype='d1'):
    """
    gid is the int corresponding to a D1/D2 neurons
    bg_cell_gids is dictionary stored in 'bg_gids_fn'
    """
    list_of_gids = bg_cell_gids[celltype]
    for i_, sublist in enumerate(list_of_gids):
        if gid in sublist:
            return i_
    return 'GID could not be mapped to action\n Check parameters!'



def get_sources(conn_list, target_gid):
    idx = (conn_list[:, 1] == target_gid).nonzero()[0]
    sources = conn_list[idx, :]
    return sources


def get_targets(conn_list, source_gid):
    idx = (conn_list[:, 0] == source_gid).nonzero()[0]
    targets = conn_list[idx, :]
    return targets


def convert_connlist_to_matrix(data, src_min=None, src_max=None, tgt_min=None, tgt_max=None):
    """
    data -- connections list (src, tgt, weight)
    """
    if src_min == None:
        src_min = np.min(data[:, 0])
    if src_max == None:
        src_max = np.max(data[:, 0])
    if tgt_min == None:
        tgt_min = np.min(data[:, 1])
    if tgt_max == None:
        tgt_max = np.max(data[:, 1])
    n_src = src_max - src_min + 1
    tgt_min, tgt_max= np.min(data[:, 1]), np.max(data[:, 1])
    n_tgt = tgt_max - tgt_min + 1
    conn_mat = np.zeros((n_src, n_tgt))
    for c in xrange(data[:,0].size):
        src = data[c, 0] - src_min
        tgt = data[c, 1] - tgt_min
        conn_mat[src, tgt] = data[c, 2]

    return conn_mat

def get_most_active_neurons(spike_data, n_cells=None):
    gids = np.unique(spike_data[:, 0])
    gids = np.array(gids, dtype=np.int)
    if n_cells == None:
        n_cells = gids.size
    n_spikes = np.zeros(gids.size)
    for i_, gid in enumerate(gids):
        n_spikes[i_] = (spike_data[:, 0] == gid).nonzero()[0].size
    idx = n_spikes.argsort()
    most_active_neuron_gids = gids[idx[-n_cells:]]
    sorted_nspikes = n_spikes[idx[-n_cells:]]
    return (most_active_neuron_gids, sorted_nspikes)


def convert_to_NEST_conform_dict(json_dict):
    testing_params = {}
    for k in json_dict.keys():
        if type(json_dict[k]) == type({}):
            d = json_dict[k]
            d_new = {}
            for key in d.keys():
                d_new[str(key)] = d[key]
            testing_params[k] = d_new
        elif type(json_dict[k]) == unicode:
            testing_params[str(k)] = str(json_dict[k])
        else:
            testing_params[str(k)] = json_dict[k]
    return testing_params


def load_params(param_fn):
    if os.path.isdir(param_fn):
        param_fn = os.path.abspath(param_fn) + '/Parameters/simulation_parameters.json'
    params = json.load(file(param_fn, 'r')) 
    return params


def compare_actions_taken(training_params, test_params):

    fn_training = training_params['actions_taken_fn']
    fn_test = test_params['actions_taken_fn']
    print 'utils.compare_actions_taken checking files:', fn_training, '\n', fn_test
    actions_training = np.loadtxt(fn_training)
    actions_test = np.loadtxt(fn_test)
    n_actions = actions_test[:, 0].size
    incorrect_iterations = []
    for i_ in xrange(n_actions):
        if actions_training[i_, 2] != actions_test[i_, 2]:
            incorrect_iterations.append(i_)

    gids_to_check = {}

    # open test params to get BG gids
    f = file(test_params['bg_gids_fn'], 'r')
    bg_gids = json.load(f)
    for i_, it in enumerate(np.unique(incorrect_iterations)):
        print 'Incorrect iteration %d: training acion index' % (it), actions_training[it, 2], ' test', actions_test[it, 2]
        gids_to_check[it] = {}
#        print 'debug utils compare_actions_taken:', actions_training
#        print 'debug utils compare_actions_taken:', actions_training[it, 2].astype(np.int)
        gids_to_check[it]['d1'] = bg_gids['d1'][actions_training[it, 2].astype(np.int)]
#        gids_to_check[it]['d2'] = bg_gids['d2'][actions_training[it, 2].astype(np.int)]
    return gids_to_check, incorrect_iterations
        


def get_min_max_gids_for_bg(params, cell_type):
    """
    cell_type -- string possibly values:
        ['d1', 'd2', 'actions', 'supervisor']
    """
    fn = params['bg_gids_fn']
    f = file(fn, 'r')
    d = json.load(f)
    gid_min, gid_max = np.infty, -np.infty
    cell_gids = d[cell_type]
    for gids in cell_gids:
        gid_min = min(gid_min, min(gids))
        gid_max = max(gid_max, max(gids))
    return gid_min, gid_max


def get_colorlist(n_colors=17):
    colorlist = ['k', 'b', 'r', 'g', 'm', 'c', 'y', \
            '#00FF99', \
            #light green
            '#FF6600', \
                    #orange
            '#CCFFFF', \
                    #light turquoise
            '#FF00FF', \
                    #light pink
            '#0099FF', \
                    #light blue
            '#CCFF00', \
                    #yellow-green
            '#D35F8D', \
                    #mauve
            '#808000', \
                    #brown-green
            '#bb99ff', \
                    # light violet
            '#7700ff', \
                    # dark violet
                    ]

    if n_colors > 17:
        r = lambda: random.randint(0,255)
        for i_ in xrange(n_colors - 17):
            colorlist.append('#%02X%02X%02X' % (r(),r(),r()))

    return colorlist


def threshold_array(d, thresh, absolute_val=True):
    """
    returns the (valid_indices, d[valid_indices) of the data array, i.e. those values that 
    if No value is above threshold, return ([], None)
    are above the threshold
    if absolute_val is True, the absolute value of d is thresholded (return d[valid_idx] can contain negative numbers)
    if absolute_val is False, only positive values are returned

    """
    if absolute_val:
        valid_idx = np.nonzero(np.abs(d) > thresh)[0]
    else:
        valid_idx = np.nonzero(d > thresh)[0]
    if valid_idx.size > 0:
        return (valid_idx, d[valid_idx])
    else:
        return ([], None)


def softmax(a, T=1):
    a_new = np.zeros(a.size)
    exp_sum = np.sum(np.exp(T * a))
    for i_ in xrange(a.size):
        a_new[i_] = np.exp(T * a[i_]) / exp_sum
    return a_new

def get_linestyles():
    linestyles = ['-', ':', '--', '-.']
    return linestyles


def get_plus_minus(rnd):
    """
    Returns either -1., or +1. as float.
    rnd -- should be your numpy.random RNG
    """
    return (rnd.randint(-1, 1) + .5) * 2


def extract_trace(d, gid):
    """
    d : voltage trace from a saved with compatible_output=False
    gid : cell_gid
    """
    indices = (d[:, 0] == gid).nonzero()[0]
    time_axis, volt = d[indices, 1], d[indices, 2]
    return time_axis, volt


def get_spikes(spiketimes_fn_merged, n_cells=0, get_spiketrains=False, gid_idx=0, NEST=True):
    """
    Returns an array with the number of spikes fired by each cell and optionally the spike train.
    nspikes[gid] 
    -- The return values (nspikes, spiketrains) are both 0 aligned
    if NEST:
        gids are +1 

    if n_cells is not given, the length of the array will be the highest gid (not advised!)
    gid_idx = 0 for NEST
    gid_idx = 1 for PyNN
    """
    time_idx = int(not(gid_idx))
    d = np.loadtxt(spiketimes_fn_merged)
    if (n_cells == 0):
        n_cells = 1 + int(np.max(d[:, gid_idx]))# highest gid
    nspikes = np.zeros(n_cells)
    spiketrains = [[] for i in xrange(n_cells)]
    if (d.size == 0):
        if get_spiketrains:
            return nspikes, spiketrains
        else:
            return spiketrains
    # seperate spike trains for all the cells
    if d.shape == (2,):
        gid = d[gid_idx]
        nspikes[int(gid)] = 1
        spiketrains[int(gid)] = [d[time_idx]]
    else:
        if NEST:
            gid_mod = 1
        else:
            gid_mod = 0
        for gid in xrange(n_cells):
            idx = d[:, gid_idx] == gid + gid_mod
            spiketrains[gid] = d[idx, time_idx]
            nspikes[gid] = d[idx, time_idx].size

    if get_spiketrains:
        return nspikes, spiketrains
    else:
        return nspikes


def get_connection_files(params, cell_type):
    fn_list = []
    pattern = params['mpn_bg%s_merged_conntracking_fn_base' % cell_type].rsplit('/')[-1]
    iterations = []
    for thing in os.listdir(params['connections_folder']):
        if string.count(thing, pattern) != 0:
            path = params['connections_folder'] + thing
            fn_list.append(path)
            m = re.match('(.*)it(\d+)\.txt$', thing)

    fn_list.sort()
    return fn_list 
        


def merge_spikes(params):

    merged_spike_fn = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], merged_spike_fn)

    cell_types = ['d1', 'd2', 'actions']
    MS = MergeSpikefiles.MergeSpikefiles(params)
    for cell_type in cell_types:
        for naction in range(params['n_actions']):
            output_fn = params['spiketimes_folder'] + params['%s_spikes_fn_merged' % cell_type] + str(naction) + '.dat'
            if not os.path.exists(output_fn):
                merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] + str(naction)
                MS.merge_spiketimes_files(merge_pattern, output_fn)
        # merge all action files

    for cell_type in cell_types:
        output_fn = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type]
        if not os.path.exists(output_fn):
            merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type]
            MS.merge_spiketimes_files(merge_pattern, output_fn)


def merge_connection_files(params):
    def merge_for_tgt_cell_type(cell_type):
        if not os.path.exists(params['mpn_bg%s_merged_conn_fn' % cell_type]):
            # merge the connection files
            merge_pattern = params['mpn_bg%s_conn_fn_base' % cell_type]
            fn_out = params['mpn_bg%s_merged_conn_fn' % cell_type]
            merge_and_sort_files(merge_pattern, fn_out, sort=True)

    def merge_for_weight_tracking(cell_type):
        for it in xrange(params['n_iterations']):
            fn_merged = params['mpn_bg%s_merged_conntracking_fn_base' % cell_type] + 'it%d.txt' % (it)
            if not os.path.exists(fn_merged):
                # merge the connection files
                merge_pattern = params['mpn_bg%s_conntracking_fn_base' % cell_type] + 'it%d_' % it
                merge_and_sort_files(merge_pattern, fn_merged, sort=True)

    merge_for_tgt_cell_type('d1')
    if params['with_d2']:
        merge_for_tgt_cell_type('d2')

    merge_pattern = params['d1_d1_conn_fn_base']
    fn_out = params['d1_d1_merged_conn_fn']
    merge_and_sort_files(merge_pattern, fn_out, sort=True)

    if params['weight_tracking']:
        # Merge the _dev files recorded for tracking the weights
        merge_for_weight_tracking('d1')
        if params['with_d2']:
            merge_for_weight_tracking('d2')



def merge_and_sort_files(merge_pattern, fn_out, sort=True, verbose=True):
    rnd_nr1 = np.random.randint(0,10**8)
    # merge files from different processors
    tmp_file = "tmp_%d" % (rnd_nr1)
    cmd = "cat %s* > %s" % (merge_pattern, tmp_file)
    if verbose:
        print 'utils.merge_and_sort_files: ', cmd
    os.system(cmd)
    # sort according to cell id
    if sort:
        sort_cmd = "sort -gk 1 %s > %s" % (tmp_file, fn_out)
#        print 'DEBUG utils.merge_and_sort_files:', sort_cmd
        os.system(sort_cmd)
        os.system("rm %s" % (tmp_file))
    else:
        mv_cmd = 'mv %s %s' % (tmp_file, fn_out)
#        print 'DEBUG utils.merge_and_sort_files:', mv_cmd
        os.system(mv_cmd)

    if verbose:
        print 'utils.merge_and_sort_files output: ', fn_out


def find_files(folder, to_match):
    """
    Use re module to find files in folder and return list of files matching the 'to_match' string
    Arguments:
    folder -- string to folder
    to_match -- a string (regular expression) to match all files in folder
    """
    assert (to_match != None), 'utils.find_files got invalid argument'
    list_of_files = []
    for fn in os.listdir(folder):
        m = re.match(to_match, fn)
        if m:
            list_of_files.append(fn)
    return list_of_files



def get_grid_index_mapping(values, bins):
    """
    Returns a 2-dim array (gid, grid_pos) mapping with values.size length, i.e. the indices of values 
    and the bin index to which each value belongs.
    values -- the values to be put in a grid
    bins -- list or array with the 1-dim grid bins 
    """

    bin_idx = np.zeros((len(values), 2), dtype=np.int)
    for i_, b in enumerate(bins):
#    for i_ in xrange(len(bins)):
#        b = bins[i_]
        idx_in_b = (values > b).nonzero()[0]
        bin_idx[idx_in_b, 0] = idx_in_b
        bin_idx[idx_in_b, 1] = i_
    return bin_idx


def sort_gids_by_distance_to_stimulus(tp, mp, t_start, t_stop, t_cross_visual_field, local_gids=None):
    """
    This function return a list of gids sorted by the distances between cells and the stimulus (in the 4-dim tuning-prop space).
    It calculates the minimal distances between the moving stimulus and the spatial receptive fields of the cells 
    and adds the distances between the motion_parameters and the preferred direction of each cell.

    Arguments:
        tp: tuning_properties array 
        tp[:, 0] : x-pos
        tp[:, 1] : y-pos
        tp[:, 2] : x-velocity
        tp[:, 3] : y-velocity
        mp: motion_parameters (x0, y0, u0, v0, orientation)

    """
    if local_gids == None: 
        n_cells = tp[:, 0].size
    else:
        n_cells = len(local_gids)
    x_dist = np.zeros(n_cells) # stores minimal distance between stimulus and cells
    # it's a linear sum of spatial distance, direction-tuning distance and orientation tuning distance
    for i in xrange(n_cells):
        x_dist[i], spatial_dist = get_min_distance_to_stim(mp, tp[i, :], t_start, t_stop, t_cross_visual_field)

    cells_closest_to_stim_pos = x_dist.argsort()
    if local_gids != None:
        gids_closest_to_stim = local_gids[cells_closest_to_stim_pos]
        return gids_closest_to_stim, x_dist[cells_closest_to_stim_pos]#, cells_closest_to_stim_velocity
    else:
        return cells_closest_to_stim_pos, x_dist[cells_closest_to_stim_pos]#, cells_closest_to_stim_velocity


def get_min_distance_to_stim(mp, tp_cell, t_start, t_stop, t_cross_visual_field): 
    """
    mp : motion_parameters (x, y, u, v, orientation), orientation is optional
    tp_cell : same format as mp
    """
    time = np.arange(t_start, t_stop, 2) # 2 [ms]
    spatial_dist = np.zeros(time.shape[0])
    x_pos_stim = mp[0] + (mp[2] * time + mp[2] * t_start) / t_cross_visual_field
    y_pos_stim = mp[1] + (mp[3] * time + mp[3] * t_start) / t_cross_visual_field
    spatial_dist = (tp_cell[0] - x_pos_stim)**2 + (tp_cell[1] - y_pos_stim)**2
    min_spatial_dist = np.sqrt(np.min(spatial_dist))

    velocity_dist = np.sqrt((tp_cell[2] - mp[2])**2 + (tp_cell[3] - mp[3])**2)

    dist =  min_spatial_dist + velocity_dist
    return dist, min_spatial_dist
    

def get_spiketimes(all_spikes, gid, gid_idx=0, time_idx=1):
    """
    Returns the spikes fired by the cell with gid
    all_spikes: 2-dim array containing all spiketimes
    gid_idx: is the column index in the all_spikes array containing GID information
    time_idx: is the column index in the all_spikes array containing time information
    """
    idx_ = (all_spikes[:, gid_idx] == gid).nonzero()[0]
    spiketimes = all_spikes[idx_, time_idx]
    return spiketimes



def extract_weight_from_connection_list(conn_list, pre_gid, post_gid):
    """
    Extract the weight that connects the pre_gid to the post_gid
    """
#    print 'debug connlist', conn_list
#    print 'debug', pre_gid, post_gid
#    print 'debug', (conn_list[:, 0] == pre_gid).nonzero()
    pre_idx = set((conn_list[:, 0] == pre_gid).nonzero()[0])
    post_idx = set((conn_list[:, 1] == post_gid).nonzero()[0])
    valid_idx = list(pre_idx.intersection(post_idx))
    if len(valid_idx) == 0:
        return 0.
    return float(conn_list[valid_idx, 2])



def get_spiketimes_within_interval(spike_data, t0, t1):
    """
    all_spikes: 2-dim array containing all spiketimes
    return those spike times which are between > t0 and <= t1
    """
    if spike_data.ndim == 2:
        t0_idx = set((spike_data[:, 1] > t0).nonzero()[0])
        t1_idx = set((spike_data[:, 1] <= t1).nonzero()[0])
        valid_idx = list(t0_idx.intersection(t1_idx))
        return spike_data[valid_idx, :]
    else:
        t0_idx = set((spike_data > t0).nonzero()[0])
        t1_idx = set((spike_data <= t1).nonzero()[0])
        valid_idx = list(t0_idx.intersection(t1_idx))
        return spike_data[valid_idx]


def communicate_local_spikes(gids, comm):

    my_nspikes = {}
    for i_, gid in enumerate(gids):
        my_nspikes[gid] = (gids == gid).nonzero()[0].size
    
    all_spikes = [{} for pid in xrange(comm.size)]
    all_spikes[comm.rank] = my_nspikes
    for pid in xrange(comm.size):
        all_spikes[pid] = comm.bcast(all_spikes[pid], root=pid)
    all_nspikes = {} # dictionary containing all cells that spiked during that iteration
    for pid in xrange(comm.size):
        for gid in all_spikes[pid].keys():
            gid_ = gid
#            gid_ = gid - 1
            all_nspikes[gid_] = all_spikes[pid][gid]
    gids_spiked = np.array(all_nspikes.keys(), dtype=np.int)
    nspikes =  np.array(all_nspikes.values(), dtype=np.int)
    comm.Barrier()
    return gids_spiked, nspikes


def filter_connection_list(d):
    """
    d -- 3 columnar connection list:
      src     tgt     weight
    returns only those rows with weight != 0
    """
    valid_idx = (d[:, 2] != 0).nonzero()[0]
    return d[valid_idx, :]



def get_xpos_log_distr(logscale, n_x, x_min=1e-6, x_max=.5):
    """
    Returns the n_hc positions
    n_x = params['n_mc_per_hc']
    x_min = params['rf_x_center_distance']
    x_max = .5 - params['xpos_hc_0']
    """
    logspace = np.logspace(np.log(x_min) / np.log(logscale), np.log(x_max) / np.log(logscale), n_x / 2 + 1, base=logscale)
    logspace = list(logspace)
    logspace.reverse()
    x_lower = .5 - np.array(logspace)
    
    logspace = np.logspace(np.log(x_min) / np.log(logscale), np.log(x_max) / np.log(logscale), n_x / 2 + 1, base=logscale)
    x_upper =  logspace + .5
    x_rho = np.zeros(n_x)
    x_rho[:n_x/2] = x_lower[:-1]

    if n_x % 2:
        x_rho[n_x/2+1:] = x_upper[1:]
    else:
        x_rho[n_x/2:] = x_upper[1:]
    return x_rho


#def get_xpos_log_distr_const_fovea(params, n_x, x_min=1e-6, x_max=.5):


def get_receptive_field_sizes_x(params, rf_x):
#    print 'rf_x', rf_x
    idx = np.argsort(rf_x)
    rf_size_x = np.zeros(rf_x.size)
    pos_idx = (rf_x[idx] > 0.5).nonzero()[0]
    neg_idx = (rf_x[idx] < 0.5).nonzero()[0]
    dx_pos_half = np.zeros(pos_idx.size)
    dx_neg_half = np.zeros(neg_idx.size)
    dx_pos_half = rf_x[idx][pos_idx][1:] - rf_x[idx][pos_idx][:-1]
    dx_neg_half = rf_x[idx][neg_idx][1:] - rf_x[idx][neg_idx][:-1]
#    print 'rf_x[idx][pos_idx]', rf_x[idx][pos_idx]
#    print 'rf_x[idx][neg_idx]', rf_x[idx][neg_idx]
#    print 'dx_pos_half', dx_pos_half
#    print 'dx_neg_half', dx_neg_half
#    print 'pos_idx', pos_idx
#    print 'idx', idx
    rf_size_x[:neg_idx.size-1] = dx_neg_half
    rf_size_x[neg_idx.size] = dx_neg_half[-1]
    if params['n_rf_x'] % 2:
        rf_size_x[pos_idx.size+2:] = dx_pos_half # for 21
    else:
        rf_size_x[pos_idx.size+1:] = dx_pos_half # for 20
    rf_size_x[pos_idx.size] = dx_pos_half[0]
    rf_size_x[idx.size / 2 - 1] = dx_pos_half[0]
    rf_size_x *= params['rf_size_x_multiplicator']
#    print 'rf_size_x', rf_size_x
    return rf_size_x


def get_receptive_field_sizes_v(params, rf_v):

#    print 'rf_v', rf_v
    idx = np.argsort(rf_v)
    rf_size_v = np.zeros(rf_v.size)
    pos_idx = (rf_v[idx] > 0.0).nonzero()[0]
    neg_idx = (rf_v[idx] < 0.0).nonzero()[0]
    dv_pos_half = np.zeros(pos_idx.size)
    dv_neg_half = np.zeros(neg_idx.size)
    dv_pos_half = rf_v[idx][pos_idx][1:] - rf_v[idx][pos_idx][:-1]
    dv_neg_half = np.abs(rf_v[idx][neg_idx][1:] - rf_v[idx][neg_idx][:-1])
    dv_neg_reverse = list(dv_neg_half)
    dv_neg_reverse.reverse()
    rf_size_v[:neg_idx.size-1] = dv_neg_reverse
    rf_size_v[pos_idx.size+1:] = dv_pos_half
    rf_size_v[pos_idx.size] = dv_pos_half[0]
    rf_size_v[idx.size / 2 - 1] = dv_neg_half[0]

    rf_size_v *= params['rf_size_v_multiplicator']
    #print 'rf_size_v', rf_size_v
    return rf_size_v



def distribute_n(n, n_proc, pid):
    """
    n: number of elements to be distributed
    pid: (int) process id of the process calling this function
    n_proc: total number of processors
    Returns the min and max index to be assigned to the processor with id pid
    """
    n_per_proc = int(n / n_proc)
    R = n % n_proc
    offset = min(pid, R)
    n_min = int(pid * n_per_proc + offset)
    if (pid < R):
        n_max = int(n_min + n_per_proc + 1)
    else:
        n_max = int(n_min + n_per_proc)
    return (n_min, n_max)


#def get_neurons_active(params, d, stim, it='all'):
#    """
#    Returns gids active during a certain stimulus (stim).

#    params  -- param dict
#    d       -- two dimensional spike data, [gid, t]
#    stim    -- integer
#    it      -- 'all' or an integer
#    """


