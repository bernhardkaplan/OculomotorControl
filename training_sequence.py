
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

BG = BasalGanglia.BasalGanglia(params, dummy=True)
VI = VisualInput.VisualInput(params)

tp = VI.set_tuning_prop_1D_with_const_fovea(cell_type='exc')

stim_params = list(params['initial_state'])
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

for i_cycle in xrange(params['n_training_cycles']):
    for i_ in xrange(params['n_training_stim_per_cycle']):
        all_mp[i_stim, :] = deepcopy(stim_params)
        if i_ % params['suboptimal_training'] == 1:
            (required_v_eye, v_y, action_idx) = BG.get_non_optimal_action_for_stimulus(stim_params)
            stim_type.append(2)
            d2_actions_trained.append(action_idx)
        else:
            (required_v_eye, v_y, action_idx) = BG.get_optimal_action_for_stimulus(stim_params)
            stim_type.append(1)
            action_v = [required_v_eye, 0.]
            stim_params = utils.get_next_stim(params, stim_params, required_v_eye)
            stim_params = list(stim_params)
            d1_actions_trained.append(action_idx)
        all_actions_trained.append(action_idx)
        speeds_trained.append(stim_params[2])
        i_stim += 1
#        print 'action_v', action_v, action_idx, stim_params

    # get start position some where in the periphery
    pm = utils.get_plus_minus(np.random)
    if pm > 0:
        stim_params[0] = np.random.uniform(.6, 1.)
    else:
        stim_params[0] = np.random.uniform(0, .4)

#    stim_params[2] = np.random.uniform(-params['v_max_tp'], params['v_min_tp'])
    # sample stimulus speed from tuning properties
    stim_params[2] = tp[np.random.choice(tp[:, 0].size), 2]


print 'all_mp', all_mp
print 'stimtype', stim_type
print 'all_actions_trained', all_actions_trained

import pylab
fig = pylab.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
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


fig = pylab.figure()
ax1 = fig.add_subplot(111)
n_bins = (np.max(d1_actions_trained) - np.min(d1_actions_trained))
cnt, bins = np.histogram(d1_actions_trained, bins=n_bins, range=(np.min(d1_actions_trained), np.max(d1_actions_trained)))
binwidth = .5 * (bins[1] - bins[0])
ax1.bar(bins[:-1], cnt, width=binwidth, label='d1 trained', facecolor='b')

n_bins = (np.max(d2_actions_trained) - np.min(d2_actions_trained))
cnt, bins = np.histogram(d2_actions_trained, bins=n_bins, range=(np.min(d2_actions_trained), np.max(d2_actions_trained)))
binwidth = .5 * (bins[1] - bins[0])
ax1.bar(bins[:-1] + binwidth, cnt, width=binwidth, label='d2 trained', facecolor='r')


ax1.set_xlabel('Actions taken')
ax1.set_ylabel('Count')
#    ax1.set_xlim((0, params['n_actions']))

fig = pylab.figure()
axv = fig.add_subplot(111)
n_bins = 20
cnt, bins = np.histogram(speeds_trained, bins=n_bins, range=(np.min(speeds_trained), np.max(speeds_trained)))
binwidth = bins[1] - bins[0]
axv.bar(bins[:-1], cnt, width=binwidth, label='speeds trained', facecolor='b')


pylab.legend()
pylab.show()


