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
    VI = VisualInput.VisualInput(params)
    stim_params_grid = VI.create_training_sequence_from_a_grid()       # sampled from a grid layed over the tuning property space

    n_stim = stim_params_grid[:, 0].size 
    shuffled_idx = range(n_stim)
    #np.random.shuffle(shuffled_idx)

    stim_params_start = stim_params_grid[shuffled_idx, :]

    print 'debug stim_params_start', stim_params_start
    print 'debug stim_params_start', stim_params_start.shape
    BG = BasalGanglia.BasalGanglia(params, dummy=True)
    output_array = np.zeros((n_stim * params['n_steps_training_trajectory'], 4))
    for i_stim in xrange(n_stim):

        print 'debug', i_stim
        [x, y, u, v] = stim_params_start[i_stim, :]
        for it_ in xrange(params['n_steps_training_trajectory'] - 1):
            output_array[i_stim * params['n_steps_training_trajectory'] + it_, :] = [x, y, u, v]
            (best_speed, vy, best_action_idx) = BG.get_optimal_action_for_stimulus(output_array[i_stim * params['n_steps_training_trajectory'] + it_, :])
            print 'i_stim it_', i_stim, it_, 'x y u v', x, u, best_speed, best_action_idx
            (x, y, u, v) = utils.get_next_stim(params, output_array[i_stim * params['n_steps_training_trajectory'] + it_, :], best_speed)
            output_array[i_stim * params['n_steps_training_trajectory']+ it_ + 1, :] = [x, y, u, v]

    return output_array


if __name__ == '__main__':


    if len(sys.argv) < 2:
        GP = simulation_parameters.global_parameters()
        params = GP.params
        GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
    else:
        params = utils.load_params(sys.argv[1])

    print 'n_cycles', params['n_training_cycles']
    np.random.seed(params['visual_stim_seed'])
    BG = BasalGanglia.BasalGanglia(params, dummy=True)
    print 'Saving training stimuli parameters to:', params['training_stimuli_fn']


    training_stimuli = create_stimuli_along_a_trajectory(params)

#    training_stimuli = create_stimuli_from_grid_center_and_tuning_prop(params)

    print 'debug', training_stimuli
    print 'Debug saving training_stimuli to:', params['training_stimuli_fn']
    np.savetxt(params['training_stimuli_fn'], training_stimuli)


#    exit(1)



    VI = VisualInput.VisualInput(params)
    supervisor_states, action_indices, motion_params_precomputed = VI.get_supervisor_actions(training_stimuli, BG)
    output_array = np.zeros((len(action_indices), 2))
    print 'action_indices', action_indices
    output_array[:, 0] = action_indices
    output_array[:, 1] = np.array(BG.action_bins_x)[action_indices]
#    output_array[:, 2] = BG.action_bins_x[action_indices]
    print 'supervisor_states:', supervisor_states
    print 'action_indices:', action_indices
    np.savetxt(params['supervisor_states_fn'], supervisor_states)
    np.savetxt(params['action_indices_fn'], action_indices, fmt='%d')
    np.savetxt(params['actions_taken_fn'], output_array)
#    np.savetxt(params['motion_params_precomputed_fn'], motion_params_precomputed)



    Plotter = Plotter(params)#, it_max=1)
    Plotter.plot_training_sample_space(plot_process=False, motion_params_fn=params['training_stimuli_fn'])
    output_fn = params['figures_folder'] + 'training_stimuli.png'
    pylab.savefig(output_fn, dpi=200)
#    Plotter.plot_training_sample_space(plot_process=True)
    Plotter.plot_precomputed_actions(plot_cells=True)

    pylab.show()
