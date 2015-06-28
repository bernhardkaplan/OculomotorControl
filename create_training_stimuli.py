import sys
import os
import VisualInput
import MotionPrediction
import BasalGanglia
import json
import simulation_parameters
import CreateConnections
import nest
import numpy as np
import time
import os
import utils
from copy import deepcopy
import pylab
from PlottingScripts.plot_training_samples import Plotter
import random
import matplotlib 
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection


def create_stimuli_from_grid_center_and_tuning_prop(params):
    VI = VisualInput.VisualInput(params)

    tp = VI.set_tuning_prop_1D_with_const_fovea(cell_type='exc')
    np.savetxt(params['tuning_prop_exc_fn'], tp)

    training_stimuli_sample = VI.create_training_sequence_iteratively()     # motion params drawn from the cells' tuning properties
    training_stimuli_grid = VI.create_training_sequence_from_a_grid()       # sampled from a grid layed over the tuning property space
    training_stimuli_center = VI.create_training_sequence_around_center()   # sample more from the center in order to reduce risk of overtraining action 0 and v_x_max
    training_stimuli = np.zeros((params['n_stim_training'], 4))
    n_grid = int(np.round(params['n_stim_training'] * params['frac_training_samples_from_grid']))
    n_center = int(np.round(params['n_stim_training'] * params['frac_training_samples_center']))
    random.seed(params['visual_stim_seed'])
    np.random.seed(params['visual_stim_seed'])
    training_stimuli[:n_center, :] = training_stimuli_center
    training_stimuli[n_center:n_center+n_grid, :] = training_stimuli_grid[random.sample(range(params['n_stim_training']), n_grid), :]

#    training_stimuli[:n_grid, :] = training_stimuli_grid[random.sample(range(params['n_stim_training']), n_grid), :]
#    training_stimuli[n_grid:n_grid+n_center, :] = training_stimuli_center 
    training_stimuli[n_grid+n_center:, :] = training_stimuli_sample[random.sample(range(params['n_stim_training']), params['n_stim_training'] - n_grid - n_center), :]
#    print 'n_grid', n_grid, 'n_center', n_center, 'n_tuning_prop', params['n_stim_training'] - n_grid - n_center
    return training_stimuli


def create_stimuli_along_a_trajectory(params):
    """
    For all params in stim_params_start, create a trajectory with params['n_steps_training_trajectory']
    """
    assert params['n_steps_training_trajectory'] > 1, 'If n_steps_training_trajectory == 1: please use another function'

    VI = VisualInput.VisualInput(params)
    stim_params_grid = VI.create_training_sequence_from_a_grid()       # sampled from a grid layed over the tuning property space

    n_stim = stim_params_grid[:, 0].size 
    shuffled_idx = range(n_stim)
    #np.random.shuffle(shuffled_idx)

    stim_params_start = stim_params_grid[shuffled_idx, :]

#    print 'debug stim_params_start', stim_params_start
#    print 'debug stim_params_start', stim_params_start.shape
    BG = BasalGanglia.BasalGanglia(params, dummy=True)
    output_array = np.zeros((n_stim * params['n_steps_training_trajectory'], 4))
    for i_stim in xrange(n_stim):
        [x, y, u, v] = stim_params_start[i_stim, :]
        for it_ in xrange(params['n_steps_training_trajectory'] - 1):
            output_array[i_stim * params['n_steps_training_trajectory'] + it_, :] = [x, y, u, v]
            (best_speed, vy, best_action_idx) = BG.get_optimal_action_for_stimulus(output_array[i_stim * params['n_steps_training_trajectory'] + it_, :])
            print 'i_stim it_', i_stim, it_, 'x y u v', x, u, best_speed, best_action_idx
            (x, y, u, v) = utils.get_next_stim(params, output_array[i_stim * params['n_steps_training_trajectory'] + it_, :], best_speed)
            output_array[i_stim * params['n_steps_training_trajectory']+ it_ + 1, :] = [x, y, u, v]

    return output_array


def create_non_overlapping_training_stimuli(params):

    training_stim = np.zeros((params['n_stim_training'], 4))
    i_stim = 0

    VI = VisualInput.VisualInput(params)
    tp = VI.set_tuning_prop_1D_with_const_fovea(cell_type='exc')

    x_lim_frac = .95
    v_lim_frac = .5 #
    xlim = ((1. - x_lim_frac) * (np.max(tp[:, 0]) - np.min(tp[:, 0])), x_lim_frac * np.max(tp[:, 0]))
    vlim = (v_lim_frac * np.min(tp[:, 2]), v_lim_frac * np.max(tp[:, 2]))

#    n_x = np.int(np.round( 1. / params['blur_X']))
    n_x = np.int(np.round((xlim[1] - xlim[0]) / params['blur_X']))
    n_v = np.int(np.round((vlim[1] - vlim[0]) / params['blur_V']))
    print 'create_non_overlapping_training_stimuli: n_x, n_v:', n_x, n_v
    x_grid = np.linspace(xlim[0], xlim[1], n_x)
    v_grid = np.linspace(vlim[0], vlim[1], n_v)
    for i_cycle in xrange(params['n_training_cycles']):
        for i_v in xrange(params['n_training_v']):
            for i_x in xrange(params['n_training_x']):
                training_stim[i_stim, 0] = x_grid[i_x % n_x]
                training_stim[i_stim, 1] = .5
                training_stim[i_stim, 2] = v_grid[i_v % n_v]
                training_stim[i_stim, 3] = .0
                i_stim += 1

    return training_stim


def plot_stim_after_action(params, mp, ax=None):
    """
    mp  --  training_stimuli (4 columns, n_training_stim rows)
    ax  -- axis to plot on
    """
    if ax == None:
        fig = pylab.figure()
        ax = fig.add_subplot(111)



    bounds = range(params['n_actions'])
    cmap = matplotlib.cm.jet
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(np.arange(bounds[0], bounds[-1], 1.))
    colors = m.to_rgba(range(params['n_actions']))

    patches = []
    for i_ in xrange(params['n_stim_training']):

        (best_speed, vy, best_action_idx) = utils.get_optimal_action(params, mp[i_, :])
        mp_next = utils.get_next_stim(params, mp[i_, :], best_speed)
        print 'Stim %d mp: (%.2f, %.2f) requires action %d' % (i_, mp[i_, 0], mp[i_, 2], best_action_idx)
        ax.plot(mp_next[0], mp_next[2], '*', markersize=10, color=colors[best_action_idx], markeredgewidth=1)
        ellipse = mpatches.Ellipse((mp_next[0], mp_next[2]), params['blur_X'], params['blur_V'], linewidth=0, alpha=0.1)
        ellipse.set_facecolor('r')
        patches.append(ellipse)
        ax.add_artist(ellipse)
    collection = PatchCollection(patches)#, alpha=0.1)
    ax1.add_collection(collection)


if __name__ == '__main__':


    if len(sys.argv) < 2:
        GP = simulation_parameters.global_parameters()
        params = GP.params
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
    else:
        params = utils.load_params(sys.argv[1])
    assert params['training'], 'Set training = True, otherwise you will get confused because of inconsistent n_stim values'

    print 'n_cycles', params['n_training_cycles']
    np.random.seed(params['visual_stim_seed'])
    BG = BasalGanglia.BasalGanglia(params, dummy=True)

#    training_stimuli = create_non_overlapping_training_stimuli(params)
    if params['n_steps_training_trajectory'] > 1:
        training_stimuli = create_stimuli_along_a_trajectory(params)
    else:
        VI = VisualInput.VisualInput(params)
        tp = VI.set_tuning_prop_1D_with_const_fovea_and_const_velocity(cell_type='exc')
        training_stimuli = create_stimuli_from_grid_center_and_tuning_prop(params)
#        training_stimuli = VI.create_training_sequence_from_a_grid()       # sampled from a grid layed over the tuning property space


    print 'debug', training_stimuli
    print 'Debug saving training_stimuli to:', params['training_stimuli_fn']
    np.savetxt(params['training_stimuli_fn'], training_stimuli)


#    exit(1)



    VI = VisualInput.VisualInput(params)
    supervisor_states, action_indices, motion_params_precomputed = VI.get_supervisor_actions(training_stimuli, BG)
    output_array = np.zeros((len(action_indices), 4))
    print 'action_indices', action_indices
    output_array[:, 0] = np.array(BG.action_bins_x)[action_indices]
    output_array[:, 2] = action_indices
#    output_array[:, 2] = BG.action_bins_x[action_indices]
    print 'supervisor_states:', supervisor_states
    print 'action_indices:', action_indices
    np.savetxt(params['supervisor_states_fn'], supervisor_states)
    np.savetxt(params['action_indices_fn'], action_indices, fmt='%d')
    np.savetxt(params['actions_taken_fn'], output_array)
#    np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)


    n_bins = params['n_actions']
    cnt, bins = np.histogram(action_indices, bins=n_bins, range=(np.min(action_indices), np.max(action_indices)))
    idx_never_done = np.nonzero(cnt == 0)[0]
    print 'Actions never done:', idx_never_done

    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax1.bar(bins[:-1], cnt, width=bins[1]-bins[0])
    ax1.set_xlabel('Actions taken')
    ax1.set_ylabel('Count')
    ax1.set_xlim((0, params['n_actions']))

    Plotter = Plotter(params)#, it_max=1)
#    ax = Plotter.plot_precomputed_actions(plot_cells=True, n_samples_to_plot=params['n_stim'])
#    plot_stim_after_action(params, training_stimuli, ax=ax)

    Plotter.plot_training_sample_space(plot_process=False, motion_params_fn=params['training_stimuli_fn'])
#    output_fn = params['figures_folder'] + 'training_stimuli.png'
#    pylab.savefig(output_fn, dpi=200)

#    ax = Plotter.plot_precomputed_actions(plot_cells=True, n_samples_to_plot=params['n_stim'])
#    ax = None
#    plot_stim_after_action(params, training_stimuli, ax=ax)


#    Plotter.plot_training_sample_space(plot_process=True)


    print 'Saving training stimuli parameters to:', params['training_stimuli_fn']
    pylab.show()
