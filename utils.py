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


def vector_average_spike_data(d, tuning_prop, n_tp_bins, t_range=None, n_time=1, tp_idx=0, tp_range=None):
    """
    d = spike_data -- 2-dim array: format [[GID_0, time_0], [GID_1, time_1], ....]
    tuning_prop -- n_cells x 4 array
    n_tp_bins -- number of bins the readout is projected to (number of positions in space / velocity domain)
    t_range -- time range
    n_time -- time axis binning
    tp_idx  -- determines the dimension for readout, given by the column used in tuning_prop
    tp_range -- tuple giving the min and max value for the readout / projection range
    """
    output_array = np.zeros((n_time, n_tp_bins))
    if t_range == None:
        t_min = np.min(d[:, 1])
        t_max = np.max(d[:, 1])
    else: 
        (t_min, t_max) = t_range
       
    t_axis = np.linspace(t_min, t_max, n_time + 1)
#    print 't_axis', t_axis, t_min, t_max
    if tp_range == None:
        tp_range = (np.min(tuning_prop[:, tp_idx]), np.max(tuning_prop[:, tp_idx]))
    tp_bins = np.linspace(tp_range[0], tp_range[1], n_tp_bins)
    for it_ in xrange(n_time):
        t0 = t_axis[it_]
        t1 = t_axis[it_ + 1]
        sd = get_spiketimes_within_interval(d, t0, t1)
        if sd.size > 0:
            idx = np.array(sd[:, 0], dtype=np.int) - 1
            hist, bin_edges = np.histogram(tuning_prop[idx, tp_idx], bins=n_tp_bins, range=(tp_range[0], tp_range[1]))
            N = sd[:, 0].size
#            print 'debug t0 t1', t0, t1
            output_array[it_, :] = hist / float(N)

#    output_array[-1, :] = 
    return output_array


def get_optimal_action(params, stim_params):
    try:
        action_bins = np.loadtxt(params['bg_action_bins_fn'])[:, 0]
    except:
        action_bins = []
        n_bins_x = np.int(np.round((params['n_actions'] - 1) / 2.))
        n_bins_y = np.int(np.round((params['n_actions'] - 1) / 2.))
        v_scale_half = ((-1.) * np.logspace(np.log(params['v_min_out']) / np.log(params['log_scale']),
                            np.log(params['v_max_out']) / np.log(params['log_scale']), num=n_bins_x,
                            endpoint=True, base=params['log_scale'])).tolist()
        v_scale_half.reverse()
        action_bins += v_scale_half
        action_bins += [0.]
        v_scale_half = (np.logspace(np.log(params['v_min_out']) / np.log(params['log_scale']),
                            np.log(params['v_max_out']) / np.log(params['log_scale']), num=n_bins_x,
                            endpoint=True, base=params['log_scale'])).tolist()
        action_bins += v_scale_half

    all_outcomes = np.zeros(len(action_bins))
    for i_, action in enumerate(action_bins):    
        all_outcomes[i_] = get_next_stim(params, stim_params, action)[0]
    best_action_idx = np.argmin(np.abs(all_outcomes - .5))
    best_speed = action_bins[best_action_idx ]
    return (best_speed, 0, best_action_idx)


def get_start_and_stop_iteration_for_stimulus_from_motion_params(motion_params_fn):
    """
    Returns a dictionary with (x, v) as key and {'start': <int>, 'stop' <int>} as value, indicating 
    the start and stop iteration (line in the motion_params_fn) during which the stimulus has been trained.
    """

    d = np.loadtxt(motion_params_fn)
    if len(d.shape) == 1: # only one stimulus
        d = d.reshape((1, 4))
    trained_stim = {}
    cnt = 0
    for i_ in xrange(d.shape[0]):
        stim_params = (d[i_, 0], d[i_, 2])
        if not trained_stim.has_key(stim_params):
            trained_stim[stim_params] = {}
            trained_stim[stim_params]['start'] = i_
            cnt += 1
        else:
            trained_stim[stim_params]['stop'] = i_ + 1
            trained_stim[stim_params]['cnt'] = cnt
    return trained_stim


def get_stim_offset(params):
    """
    As long as the parameter dictionary does not contain training_stim_offset
    this helper function returns the first stim for which entry in d1/d2_actions_trained is not len = 0
    """
    stim_offset_d1 = np.inf
    stim_offset_d2 = np.inf
    for stim_cnt in params['d1_actions_trained'].keys():
#        print 'stim_cnt', stim_cnt, params['d1_actions_trained'][stim_cnt], len(params['d1_actions_trained'][stim_cnt]), min(stim_offset_d1, stim_cnt) #np.min(stim_offset_d1, stim_cnt)
        if len(params['d1_actions_trained'][stim_cnt]) != 0:
#            print 'd1', stim_cnt, stim_offset_d1, np.min(stim_offset_d1, stim_cnt)
            stim_offset_d1 = min(stim_offset_d1, int(stim_cnt))
        if len(params['d2_actions_trained'][stim_cnt]) != 0:
#            print 'd2', stim_cnt, stim_offset_d2, np.min(stim_offset_d2, stim_cnt)
            stim_offset_d2 = min(stim_offset_d2, int(stim_cnt))
    return min(stim_offset_d1, stim_offset_d2)


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
    x_stim = stim_params[0] + (stim_params[2] - v_eye) * params['t_iteration'] / params['t_cross_visual_field']
    return (x_stim, stim_params[1], stim_params[2], stim_params[3])



def get_sigmoid_params(params, x_pre, v_stim):
    """
    Based on the stimulus parameters, return the coefficients / parameters for a sigmoidal
    reward function
    """
    x_pre_range = (0., 0.5) # absolute displacement
    tau_range = (40., 100.) 
    # tau_range[0] --> affects the stimuli that start at x_pre_range[0], i.e. in the periphery
    # tau_range[1] --> affects the stimuli that start at x_pre_range[1], near the center
#    tau = transform_quadratic(x_pre, 'neg', tau_range, x_pre_range)
    tau = transform_linear(x_pre, tau_range, x_pre_range)

    v_stim_max = 2.
    abs_speed_factor = transform_linear(np.abs(v_stim), [0.5, 1.], [0., v_stim_max])
#    tau *= abs_speed_factor
    # take into account how far the stimulus moves
    dx = v_stim * params['t_iteration'] / params['t_cross_visual_field']
    c_range = (0.35 - np.sign(v_stim) * dx, 0.1 - np.sign(v_stim) * dx) 
    # c_range --> determines the transition point from neg->pos reward (exactly if |K_min| == K_max)
    # c_raneg[1] --> determines tolerance for giving reward near center
    c = transform_quadratic(x_pre, 'pos', c_range, x_pre_range)
    c *= abs_speed_factor
#    c = transform_linear(x_pre, c_range, x_pre_range)
    return c, tau

def sigmoid(x, a, b, c, d, tau):
    # d = limit value for x -> - infinity
    # a, b = limit value for x -> + infinity
    # tau, c = position for transition
    f_x = a / (b + d * np.exp(-tau * (x - c)))
    return f_x


def get_reward_sigmoid(x_new, stim_params, params):
    """
    Computes the reward based on the resulting position
    """
    if params == None:
        K_min = -1.
        K_max = 1.
    else:
        K_min = params['neg_kappa']
        K_max = params['pos_kappa']
    a = 1.
    b = 1.
    d = 1.
    x_center = 0.5 
    c, tau = get_sigmoid_params(params, stim_params[0], stim_params[2])
    R = K_max - (K_max - K_min) * sigmoid(np.abs(x_new - x_center), a, b, c, d, tau)
    return R

    """
def get_reward_gauss()
    x_old = stim_params[0]
    v_stim = stim_params[2]
    dx_i = x_old - .5 
    dx_j = x_new - .5
    dx_i_abs = np.abs(dx_i)
    dx_j_abs = np.abs(dx_j)

    x_fac = (np.abs(dx_i_abs  - dx_j_abs) / .5) ** 2
#    x_fac = (dx_i_abs  - dx_j_abs) / .5
    v_fac = np.abs(v_stim)


#    A = 0.5

    w_x = .1
    w_v = .01

    reward_width = w_x * x_fac + w_v * v_fac + reward_width_min
    a = 1.
    b = 1.
    c = 5.    
    d = 10.
    tau = 1.
    # tau * c determine the width 
    # for x_fac * v_fac = 0 --> sigma_r = a / (b + d * exp(tau * c))

#    r_amp = np.abs(v_stim) / 50.
#    r_amp = 1.

#    reward_width = reward_width_min + r_amp * np.abs(x_old - .5)**3
#    reward_width = reward_width_min + r_amp * np.abs(x_old - .5)**3

#    reward_width = a / (b + d * np.exp(- tau * (w_v * v_fac * w_x * x_fac - c)))

    x_displ_new = np.abs(x_new - .5)

#    R = np.exp(-(x_displ_new)**2 / (2 * reward_width)) + K_min
    R = (K_max - K_min) * np.exp(-(x_displ_new)**2 / (2 * reward_width)) + K_min

    return R, reward_width
    """


def get_reward_from_perceived_states(old_pos, new_pos, punish_overshoot=1., params=None):
    """
    Computes the reward based on the two consecutive positions
    """
    if params == None:
        K_min = -1.
        K_max = 1.
    else:
        K_min = params['neg_kappa']
        K_max = params['pos_kappa']

    dx_i = old_pos - .5 # -2 and -1 because self.iteration is + 1 (because compute_input has been called before)
    dx_j = new_pos - .5
    dx_i_abs = np.abs(dx_i)
    dx_j_abs = np.abs(dx_j)

    relative_improvement = (dx_i_abs - dx_j_abs) / .5
    R = (dx_i_abs - dx_j_abs ) / dx_i_abs
    if R < -2.:
        R = -2.
    return R

#    return R
#    if relative_improvement < 0.:
#        R = K_min
#    else:
#        R = -K_min + (K_max - K_min) * relative_improvement
#    return R
    
#    R = 1. - (dx_j_abs / dx_i_abs)**2
#    return np.sign(R)
#    return R

#    delta_x_abs = dx_j_abs - dx_i_abs # if diff_dx_abs < 0: # improvement
#    R = - delta_x_abs / .5
#    if np.sign(dx_i) != np.sign(dx_j): # 'overshoot'
#        R *= punish_overshoot
#    return R



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


def get_bg_gid_ranges(params):
    f = file(params['bg_gids_fn'], 'r')
    gids = json.load(f)
    cell_types = ['d1', 'd2', 'action']
    # python 2.6
    gid_ranges = {}
    for ct in cell_types:
        gid_ranges[ct] = []
    # python 2.7
    #gid_ranges = {'%s' % ct : [] for ct in cell_types}

    for ct in cell_types:
        gid_ranges[ct] = (np.min(gids[ct]), np.max(gids[ct]))
    return gid_ranges


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
        ['d1', 'd2', 'action', 'supervisor']
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


def get_plus_minus(rnd, n=1):
    """
    Returns either -1., or +1. as float.
    rnd -- should be your numpy.random RNG
    """
    return (rnd.randint(-1, 1, n) + .5) * 2


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

#    if not os.path.exists(params['folder_name']):
#        print 'ERRER Folder
#    print 'debug',os.path.exists(params['folder_name'])
#    exit(1)
    assert os.path.exists(params['folder_name']), 'ERROR in utils.merge_spikes: Folder does not exist: %s' % (params['folder_name'])
    merged_spike_fn = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    if not os.path.exists(merged_spike_fn):
        merge_and_sort_files(params['spiketimes_folder'] + params['mpn_exc_spikes_fn'], merged_spike_fn)

    cell_types = ['d1', 'd2', 'action']
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



def extract_weight_from_connection_list(conn_list, pre_gid, post_gid, idx=None):
    """
    Extract the weight that connects the pre_gid to the post_gid
    """
#    print 'debug connlist', conn_list
#    print 'debug', pre_gid, post_gid
#    print 'debug', (conn_list[:, 0] == pre_gid).nonzero()
    if idx == None:
        idx = 2
    pre_idx = set((conn_list[:, 0] == pre_gid).nonzero()[0])
    post_idx = set((conn_list[:, 1] == post_gid).nonzero()[0])
    valid_idx = list(pre_idx.intersection(post_idx))
    if len(valid_idx) == 0:
        return 0.
#    print 'debug', valid_idx, idx, conn_list[valid_idx, idx], pre_gid, post_gid
    return float(conn_list[valid_idx, idx])



def get_spiketimes_within_interval(spike_data, t0, t1):
    """
    all_spikes: 2-dim array containing all spiketimes
    return those spike times which are between > t0 and <= t1
    """
    assert (t0 < t1), 'get_spiketimes_within_interval got wrong order of t0 and t1: first time must be smaller than t1'
    if spike_data.ndim == 2:
        t0_idx = (spike_data[:, 1] > t0).nonzero()[0]
        t1_idx = (spike_data[:, 1] <= t1).nonzero()[0]
        valid_idx = set(t0_idx).intersection(t1_idx)
        return spike_data[list(valid_idx), :]
    else:
        t0_idx = set((spike_data > t0).nonzero()[0])
        t1_idx = set((spike_data <= t1).nonzero()[0])
        valid_idx = list(t0_idx.intersection(t1_idx))
        return spike_data[valid_idx]

#def get_spikedata_within_interval(spike_data, t0, t1):
#    if spike_data.ndim == 2:
#        t0_idx = set((spike_data[:, 1] > t0).nonzero()[0])
#        t1_idx = set((spike_data[:, 1] <= t1).nonzero()[0])
#        valid_idx = list(t0_idx.intersection(t1_idx))
#        return spike_data[valid_idx, :]
#    else:
#        t0_idx = set((spike_data > t0).nonzero()[0])
#        t1_idx = set((spike_data <= t1).nonzero()[0])
#        valid_idx = list(t0_idx.intersection(t1_idx))
#        return spike_data[valid_idx]


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



def get_receptive_field_sizes_x(params, rf_x):
    idx = np.argsort(rf_x)
    rf_size_x = np.zeros(rf_x.size)
    pos_idx = (rf_x[idx] > 0.5).nonzero()[0]
    neg_idx = (rf_x[idx] < 0.5).nonzero()[0]
    dx_pos_half = np.zeros(pos_idx.size)
    dx_neg_half = np.zeros(neg_idx.size)
    dx_pos_half = rf_x[idx][pos_idx][1:] - rf_x[idx][pos_idx][:-1]
    dx_neg_half = rf_x[idx][neg_idx][1:] - rf_x[idx][neg_idx][:-1]
    rf_size_x[:neg_idx.size-1] = dx_neg_half
    rf_size_x[neg_idx.size] = dx_neg_half[-1]
    if params['n_rf_x'] % 2:
        rf_size_x[pos_idx.size+2:] = dx_pos_half # for 21
    else:
        rf_size_x[pos_idx.size+1:] = dx_pos_half # for 20
    rf_size_x[pos_idx.size] = dx_pos_half[0]
    rf_size_x[idx.size / 2 - 1] = dx_pos_half[0]
    rf_size_x *= params['rf_size_x_multiplicator']
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



def transform_linear(x, y_range, x_range=None):
    """
    x: single value or x-range to be linearly mapped into y_range
    y_min, y_max : lower and upper boundaries for the range into which x
                   is transformed to
    if x_range == None:
        x must be a list or array with more than one element, other wise no mapping is possible into y-range

    Returns y = f(x), f(x) = m * x + b
    """
    error_txt = 'Error: can not map a single value without x_range into the given y_range. \n \
            Please give x_range or use an array (or list) for the x parameter when calling utils.linear_transformation!'
    if x_range == None:
        x_min = np.min(x)
        x_max = np.max(x)
        assert (x_min != x_max), error_txt
    else:
        x_min = np.min(x_range)
        x_max = np.max(x_range)
    y_min, y_max = y_range
    assert (x_min != x_max), error_txt
    return (y_min + (y_max - y_min) / (x_max - x_min) * (x - x_min))


def transform_quadratic(x, a, y_range, x_range=None):
    """
    Returns the function   f(x) = a * x**2 + b * x + c    for a value or interval.
    (however, this function internally works with the vertex form f(x) = a * (x - h)**2 + k (where (h, k) are the vertex' (x, y) coordinates
    The vertex coordinates are derived depending on x (or x_range), y_range and 'a'

    x -- either a list or array of x-values to be mapped, or a single value
        if x is a single value x_range can not be None
    a -- 'pos' or 'neg' (if 
        if a == 'pos': parabola is open upwards --> a > 0
           a == 'neg': parabola is open downwards --> a < 0
        if a > 0 and y_range[0] < y_range[1] --> quadratic increase from left (x_range[0]) to right x_range[1] (parabola open 'upwards')
        if a < 0 and y_range[0] < y_range[1] --> quadratic approach from x_range[0] to x_range[1] (parabola open 'downwards')
        if a > 0 and y_range[0] > y_range[1] --> quadratic decrease from left to right (parabola open upwards)
        if a < 0 and y_range[0] > y_range[1] --> quadratic decrease from left to right (parabola open downwards)
    """
    
    if a != 'pos' and a != 'neg':
        raise ValueError('The parameter \'a\' must be either a \'neg\' or \'pos\' and determines whether the parabola implementing your quadratic fit is open upwards or downwards')
    error_txt = 'Error: can not map a single value without x_range into the given y_range. \n \
            Please give x_range or use an array (or list) for the x parameter when calling utils.linear_transformation!'
    if x_range != None:
        assert x_range[0] < x_range[1], 'Error: please give x_range as tuple with the smaller element first, e.g.  x_range = (0, 1) and NOT (1, 0)'
    else:
        x_range = (np.min(x), np.max(x))
    assert (x_range[0] != x_range[1]), error_txt

    assert a != 0, 'if you want a == 0, you should use utils.transform_linear'
    # determine the vertex and the other point of the parabola
    if a == 'neg' and y_range[0] < y_range[1]:
        vertex = (x_range[1], y_range[1])
        x0 = x_range[0]
        y0 = y_range[0]
    elif a == 'neg' and y_range[0] > y_range[1]:
        vertex = (x_range[0], y_range[0])
        x0 = x_range[1]
        y0 = y_range[1]
    elif a == 'pos' and y_range[0] < y_range[1]:
        vertex = (x_range[0], y_range[0])
        x0 = x_range[1]
        y0 = y_range[1]
    elif a == 'pos' and y_range[0] > y_range[1]:
        vertex = (x_range[1], y_range[1])
        x0 = x_range[0]
        y0 = y_range[0]
            
    alpha = (y0 - vertex[1]) / (x0 - vertex[0])**2
    f_x = alpha * (x - vertex[0])**2 + vertex[1]
    return f_x
