"""
This file contains a bunch of helper functions.
"""

import numpy as np
import os
import re
import json
import simulation_parameters
import MergeSpikefiles

def remove_empty_files(folder):
    for fn in os.listdir(folder):
        path = os.path.abspath(folder) + '/%s' % fn
        file_size = os.path.getsize(path)
        if file_size == 0:
            rm_cmd = 'rm %s' % (path)
            os.system(rm_cmd)

def get_sources(conn_list, target_gid):
    idx = conn_list[:, 1] == target_gid
    sources = conn_list[idx, :]
    return sources


def get_targets(conn_list, source_gid):
    idx = conn_list[:, 0] == source_gid
    targets = conn_list[idx, :]
    return targets


def get_most_active_neurons(spike_data, n_cells=None):
    gids = np.unique(spike_data[:, 0])
    if n_cells == None:
        n_cells = gids.size
    n_spikes = np.zeros(gids.size)
    for i_, gid in enumerate(gids):
        n_spikes[i_] = (spike_data[:, 0] == gid).nonzero()[0].size
    idx = n_spikes.argsort()
    most_active_neuron_gids = gids[idx[-n_cells:]]
    print 'most_active_neuron_gids', most_active_neuron_gids
    print 'nspikes:', n_spikes[idx[-n_cells:]]
    return most_active_neuron_gids


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
#    ps = simulation_parameters.global_parameters(param_fn)
#    folder_name = ps.params['folder_name']
#    ps.set_filenames(folder_name)


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
        gids_to_check[it]['d1'] = bg_gids['d1'][actions_training[it, 2].astype(np.int)]
        gids_to_check[it]['d2'] = bg_gids['d2'][actions_training[it, 2].astype(np.int)]
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


def get_colorlist():
    colorlist = ['k', 'b', 'r', 'g', 'm', 'c', 'y', '#FF6600', '#CCFF00', \
            '#808000', '#D35F8D']
    return colorlist

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


def merge_spikes(params):
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




def merge_and_sort_files(merge_pattern, fn_out, sort=True):
    rnd_nr1 = np.random.randint(0,10**8)
    # merge files from different processors
    tmp_file = "tmp_%d" % (rnd_nr1)
    cmd = "cat %s* > %s" % (merge_pattern, tmp_file)
    print 'utils.merge_and_sort_files: ', cmd
    os.system(cmd)
    # sort according to cell id
    if sort:
        os.system("sort -gk 1 %s > %s" % (tmp_file, fn_out))
    else:
        os.system('mv %s %s' % (tmp_file, fn_out))
    os.system("rm %s" % (tmp_file))
    print 'utils.merge_and_sort_files output: ', fn_out


def find_files(folder, to_match):
    """
    Use re module to find files in folder and return list of files matching the 'to_match' string
    Arguments:
    folder -- string to folder
    to_match -- a string (regular expression) to match all files in folder

    """
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

def get_spiketimes_within_interval(spike_data, t0, t1):
    """
    all_spikes: 2-dim array containing all spiketimes
    return those spike times which are between > t0 and <= t1
    """

    t0_idx = set((spike_data[:, 1] > t0).nonzero()[0])
    t1_idx = set((spike_data[:, 1] <= t1).nonzero()[0])
    valid_idx = list(t0_idx.intersection(t1_idx))
    return spike_data[valid_idx, :]


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
#            gid_ = gid
            gid_ = gid - 1
            all_nspikes[gid_] = all_spikes[pid][gid]
    gids_spiked = np.array(all_nspikes.keys(), dtype=np.int)
    nspikes =  np.array(all_nspikes.values(), dtype=np.int)
    comm.barrier()
    return gids_spiked, nspikes


def filter_connection_list(d):
    """
    d -- 3 columnar connection list:
      src     tgt     weight
    returns only those rows with weight != 0
    """
    valid_idx = (d[:, 2] != 0).nonzero()[0]
    return d[valid_idx, :]


#def get_neurons_active(params, d, stim, it='all'):
#    """
#    Returns gids active during a certain stimulus (stim).

#    params  -- param dict
#    d       -- two dimensional spike data, [gid, t]
#    stim    -- integer
#    it      -- 'all' or an integer
#    """


