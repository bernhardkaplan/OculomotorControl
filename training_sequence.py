
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


GP = simulation_parameters.global_parameters()
params = GP.params
GP.write_parameters_to_file(params['params_fn_json'], params) # write_parameters_to_file MUST be called before every simulation
print 'n_cycles', params['n_training_cycles']

np.random.seed(params['visual_stim_seed'])
BG = BasalGanglia.BasalGanglia(params, dummy=True)
VI = VisualInput.VisualInput(params)

tp = VI.set_tuning_prop_1D_with_const_fovea(cell_type='exc')

#stim_params = list(params['initial_state'])
stim_params = [0., 0.5, 0., 0.]

#(required_v_eye, v_y, action_idx) = BG.get_optimal_action_for_stimulus(stim_params)
#action_v = [required_v_eye, 0.]
#print 'action_v', action_v
#stim_params = utils.get_next_stim(params, stim_params, required_v_eye)
#print 'next mp:', stim_params


n_stim = params['n_training_stim_per_cycle'] * params['n_training_cycles']
all_mp = np.zeros((n_stim, 4))
print 'initial motion_params', stim_params
i_stim = 0

stim_type = []
d1_actions_trained = []
d2_actions_trained = []
all_actions_trained = []
speeds_trained = []
positions_trained = []


training_stimuli = np.zeros((params['n_stim_training'], 4))
#v_lim_frac = .7
#v_lim = (v_lim_frac * np.min(tp[:, 2]), v_lim_frac * np.max(tp[:, 2]))
#v_grid = np.linspace(v_lim[0], v_lim[1], params['n_training_v'])
#VI.RNG.shuffle(v_training)

training_stimuli = np.zeros((params['n_stim_training'], 4))
training_stimuli_sample = VI.create_training_sequence_iteratively()     # motion params drawn from the cells' tuning properties
training_stimuli_grid = VI.create_training_sequence_from_a_grid()       # sampled from a grid layed over the tuning property space
training_stimuli_center = VI.create_training_sequence_around_center()   # sample more from the center in order to reduce risk of overtraining action 0 and v_x_max
n_grid = int(np.round(params['n_stim_training'] * params['frac_training_samples_from_grid']))
n_center = int(np.round(params['n_stim_training'] * params['frac_training_samples_center']))
n_training_stim_from_tp_sampling = params['n_stim_training'] - n_grid - n_center
training_stimuli[:n_grid, :] = training_stimuli_grid[np.random.choice(range(params['n_stim_training']), n_grid), :]
training_stimuli[n_grid:n_grid+n_center, :] = training_stimuli_center 
training_stimuli[n_grid+n_center:, :] = training_stimuli_sample[np.random.choice(range(params['n_stim_training']), n_training_stim_from_tp_sampling), :]
#VI.RNG.shuffle(v_training)

v_stim_cnt = 0
for i_cycle in xrange(params['n_training_cycles']):
    for i_v in xrange(params['n_training_v']):

#        stim_params = training_stimuli[i_stim, :]
#        stim_params = training_stimuli[i_stim, :]

        # sample stimulus speed from tuning properties
#        stim_params[2] = tp[np.random.choice(tp[:, 0].size), 2]
        stim_params = [0., 0.5, 0., 0.]
        # get start position some where in the periphery
        pm = utils.get_plus_minus(np.random)
        if pm > 0:
            stim_params[0] = np.random.uniform(.5 + params['center_stim_width'], 1.)
        else:
            stim_params[0] = np.random.uniform(0, .5 - params['center_stim_width'])
#        stim_params[2] = np.random.choice(v_grid, 1) + utils.get_plus_minus(VI.RNG) * VI.RNG.uniform(0, params['training_stim_noise_v'])
        stim_params[2] = training_stimuli[v_stim_cnt, 2]

        for i_x in xrange(params['n_training_x']):
            for i_neg in xrange(params['suboptimal_training']):
                (required_v_eye, v_y, action_idx) = BG.get_non_optimal_action_for_stimulus(stim_params)
                action_v = [required_v_eye, 0.]
                stim_type.append(2)
                d2_actions_trained.append(action_idx)
                all_actions_trained.append(action_idx)
                speeds_trained.append(stim_params[2])
                positions_trained.append(stim_params[0])
                # sim
#                training_stimuli[i_stim, :] = stim_params
                all_mp[i_stim, :] = stim_params
                i_stim += 1

            # one training with the correct / optimal action
            (required_v_eye, v_y, action_idx) = BG.get_optimal_action_for_stimulus(stim_params)
#            training_stimuli[i_stim, :] = stim_params
            all_mp[i_stim, :] = deepcopy(stim_params)
            stim_type.append(1)
            action_v = [required_v_eye, 0.]
            stim_params = utils.get_next_stim(params, stim_params, required_v_eye)
            stim_params = list(stim_params)
            d1_actions_trained.append(action_idx)
            all_actions_trained.append(action_idx)
            speeds_trained.append(stim_params[2])
            positions_trained.append(stim_params[0])
            i_stim += 1
        v_stim_cnt += 1

print 'Saving training sequence parameters to:', params['training_sequence_fn']
np.savetxt(params['training_sequence_fn'], all_mp)

#    for i_ in xrange(params['n_training_stim_per_cycle']):
#        all_mp[i_stim, :] = deepcopy(stim_params)
#        if i_ < params['suboptimal_training']:
#        else:
#            (required_v_eye, v_y, action_idx) = BG.get_optimal_action_for_stimulus(stim_params)
#            stim_type.append(1)
#            action_v = [required_v_eye, 0.]
#            stim_params = utils.get_next_stim(params, stim_params, required_v_eye)
#            stim_params = list(stim_params)
#            d1_actions_trained.append(action_idx)
#        all_actions_trained.append(action_idx)
#        speeds_trained.append(stim_params[2])
#        i_stim += 1
#        print 'action_v', action_v, action_idx, stim_params

#    stim_params[2] = np.random.uniform(-params['v_max_tp'], params['v_min_tp'])


#print 'all_mp', all_mp
#print 'stimtype', stim_type
#print 'all_actions_trained', all_actions_trained
#print 'positions_trained:', positions_trained

import pylab

fig1 = pylab.figure()
ax = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)
n_stim = i_stim 
ax.plot(range(n_stim), all_mp[:, 0])
ax2.plot(range(n_stim), all_mp[:, 2])
ax.set_ylabel('Retinal displacement')
ax2.set_ylabel('Stimulus speed')
for i_ in xrange(i_stim):
    if stim_type[i_] == 2:
        ax.scatter(i_, all_mp[i_, 0], marker='v', c='r', s=100)
    else:
        ax.scatter(i_, all_mp[i_, 0], marker='o', c='b', s=100)
    ax.text(i_, all_mp[i_, 0] + 0.1, '%d' % all_actions_trained[i_])
ax.set_xlim((0, n_stim))
ax2.set_xlim((0, n_stim))

fig2 = pylab.figure()
ax1 = fig2.add_subplot(111)
n_bins = (np.max(d1_actions_trained) - np.min(d1_actions_trained))
cnt, bins = np.histogram(d1_actions_trained, bins=n_bins, range=(np.min(d1_actions_trained), np.max(d1_actions_trained)))
binwidth = .5 * (bins[1] - bins[0])
ax1.bar(bins[:-1], cnt, width=binwidth, label='d1 trained', facecolor='b')

n_bins = (np.max(d2_actions_trained) - np.min(d2_actions_trained))
cnt, bins = np.histogram(d2_actions_trained, bins=n_bins, range=(np.min(d2_actions_trained), np.max(d2_actions_trained)))
binwidth = .5 * (bins[1] - bins[0])
ax1.bar(bins[:-1] + binwidth, cnt, width=binwidth, label='d2 trained', facecolor='r')
pylab.legend()


ax1.set_xlabel('Actions taken')
ax1.set_ylabel('Count')
#    ax1.set_xlim((0, params['n_actions']))

fig3 = pylab.figure()
axv = fig3.add_subplot(111)
n_bins = 20
cnt, bins = np.histogram(speeds_trained, bins=n_bins, range=(np.min(speeds_trained), np.max(speeds_trained)))
binwidth = bins[1] - bins[0]
axv.bar(bins[:-1], cnt, width=binwidth, facecolor='b')
axv.set_xlabel('Speeds trained')


fig4 = pylab.figure()
ax_x = fig4.add_subplot(111)
n_bins = 20
cnt, bins = np.histogram(positions_trained, bins=n_bins, range=(np.min(positions_trained), np.max(positions_trained)))
binwidth = bins[1] - bins[0]
ax_x.bar(bins[:-1], cnt, width=binwidth, facecolor='b')
ax_x.set_xlabel('Positions trained')


pylab.show()


