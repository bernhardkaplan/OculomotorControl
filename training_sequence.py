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


GP = simulation_parameters.global_parameters()
params = GP.params
GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
print 'n_cycles', params['n_training_cycles']

np.random.seed(params['visual_stim_seed'])
BG = BasalGanglia.BasalGanglia(params, dummy=True)
VI = VisualInput.VisualInput(params)

tp = VI.set_tuning_prop_1D_with_const_fovea(cell_type='exc')

if params['mixed_training_cycles']:
    training_stimuli = VI.create_training_sequence_RBL_mixed_within_a_cycle()
else:
    training_stimuli = VI.create_training_sequence_RBL_cycle_blocks()
print 'training_stimuli block\n', training_stimuli

reward_lookup = {}
####################################
#   T R A I N   A   S T I M U L U S 
####################################

count_actions_trained_d1 = np.zeros(params['n_actions'])
count_actions_trained_d2 = np.zeros(params['n_actions'])
i_stim  = 0
repeat_action_threshold = 0.05
trained_actions = []
pos_reward = False
for i_cycle in xrange(params['n_training_cycles']):
    for i_v in xrange(params['n_training_v']):
        for i_trials_per_speed in xrange(params['n_training_x']):

            stim_params = training_stimuli[i_stim, :]

            # FAKE SIMULATION
            if reward_lookup.has_key((stim_params[0], stim_params[2])):
                prev_reward, prev_action = reward_lookup[(stim_params[0], stim_params[2])]
                if prev_reward > repeat_action_threshold:
                    r, selected_action = reward_lookup[(stim_params[0], stim_params[2])]
                    print 'taking the same action as before', 
                else:
                    selected_action = np.random.randint(0, params['n_actions'])
                    print 'R < thresh; taking a new random action', selected_action
            else:
                selected_action = np.random.randint(0, params['n_actions'])
                print 'New stimulus; taking a new random action', selected_action

            trained_action = [BG.action_bins_x[selected_action], 0., selected_action]
            trained_actions.append(trained_action[2])
            next_stim = utils.get_next_stim(params, stim_params, BG.action_bins_x[selected_action])
            R = utils.get_reward_from_perceived_states(stim_params[0], next_stim[0])

            if R > 0:
                count_actions_trained_d1[trained_action[2]] += 1
            else:
                count_actions_trained_d2[trained_action[2]] += 1
            reward_lookup[(stim_params[0], stim_params[2])] = (R, trained_action[2])
            print 'stim_params for i_stim %d' % i_stim, stim_params, 'R:', R, 'selected_action:', selected_action, 'v_eye:', BG.action_bins_x[selected_action], 'next stim', next_stim

#            if count_actions_trained_d1[trained_action[2]] > 3:
#                break

#            if len(trained_actions) > 2:
#                if (trained_actions[-1] == trained_actions[-2] == selected_action):
#                    print 'interrupt cycle'
#                    break
            trained_actions.append(trained_action[2])
            i_stim += 1
            print 'reward_lookup', i_cycle, i_v, i_trials_per_speed
            for k in reward_lookup.keys():
                print k, reward_lookup[k]
                
print 'count_actions_trained_d1', count_actions_trained_d1
print 'count_actions_trained_d2', count_actions_trained_d2

