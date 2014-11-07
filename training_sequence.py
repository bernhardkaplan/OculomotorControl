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

# old
#if params['mixed_training_cycles']:
#    training_stimuli = VI.create_training_sequence_RBL_mixed_within_a_cycle()
#else:
#    training_stimuli = VI.create_training_sequence_RBL_cycle_blocks()


training_stimuli = VI.get_training_stimuli()
print 'different training_stimuli \n', training_stimuli

idx = range(training_stimuli[:, 0].size)
np.random.shuffle(idx)
output_fn = 'training_stimuli_nV%d_nX%d.dat' % (params['n_training_v'], params['n_training_x'])
print 'Saving the training stimuli to:', output_fn
np.savetxt(output_fn, training_stimuli[idx, :])
exit(1)

repeat_action_threshold = 0.05

for i_cycle in xrange(params['n_training_cycles']):
    print '\n================ NEW CYCLE ======================'
    order_of_stim = range(params['n_training_stim_per_cycle'])
    np.random.shuffle(order_of_stim)
    stimulus_action_lookup = {}
    actions_per_stim = [{a: 0 for a in xrange(params['n_actions'])} for i in xrange(params['n_training_stim_per_cycle'])] 
    for i_ in xrange(params['n_training_stim_per_cycle']):
        i_stim = order_of_stim[i_]
        # stores the stim_id the number each action has been trained within that cycle
        cnt_trial = 0  # counts the total number of trials for any action (including pos and neg reward trials)

        count_actions_trained_d1 = np.zeros(params['n_actions'])
        count_actions_trained_d2 = np.zeros(params['n_actions'])
        actions_to_select = range(params['n_actions'])
        stim_params = training_stimuli[i_stim, :]
        while (cnt_trial < params['n_max_trials_same_stim']): # independent of rewards
            print '\nDEBUG stim_params', stim_params, 'cnt_trial: %d\ti_stim %d\ti_cycle %d' % (cnt_trial, i_stim, i_cycle)

            # FAKE SIMULATION
            # check if this stimulus has been presented
            if stimulus_action_lookup.has_key((stim_params[0], stim_params[2])):
                print 'OLD stim', 
                # get the history with previously selected actions, the given rewards and the number of trials 
                prev_action, prev_reward = stimulus_action_lookup[(stim_params[0], stim_params[2])][-1] # only check the last action taken for that stimulus
                if prev_reward < repeat_action_threshold:
                    selected_action = actions_to_select[np.random.randint(0, len(actions_to_select))]
                    print ', but taking a NEW random action:', selected_action
                    R = BG.get_binary_reward(stim_params, selected_action)
                    stimulus_action_lookup[(stim_params[0], stim_params[2])].append((selected_action, R))
                    actions_to_select.remove(selected_action)
                else:
                    selected_action = prev_action
                    R = BG.get_binary_reward(stim_params, selected_action)
                    print ', taking the SAME action as before, action=', selected_action
            else:
                selected_action = np.random.randint(0, params['n_actions'])
                print 'NEW stim, NEW rnd action:', selected_action
                R = BG.get_binary_reward(stim_params, selected_action)
                stimulus_action_lookup[(stim_params[0], stim_params[2])] = [(selected_action, R)]
                actions_to_select.remove(selected_action)

            actions_per_stim[i_stim][selected_action] += 1
            print 'Reward for action %d: %.2f' % (selected_action, R)

            if R > 0:
                count_actions_trained_d1[selected_action] += 1
            else:
                count_actions_trained_d2[selected_action] += 1

            cnt_trial += 1
            if (actions_per_stim[i_stim][selected_action] >= params['n_max_trials_pos_rew']):# and R > 0): 
                # new stimulus!
                i_stim += 1
                cnt_trial = 0
                print 'Ending training for this stimulus for this stimulus'
                print 'count_actions_trained_d1', count_actions_trained_d1
                print 'count_actions_trained_d2', count_actions_trained_d2
                break


