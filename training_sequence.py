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

stim_type = []
d1_actions_trained = []
d2_actions_trained = []
all_actions_trained = []
speeds_trained = []
positions_trained = []
x_stim_only = []

training_stimuli = VI.create_training_sequence_RBL()

#d1_training_stim_and_actions = np.zeros((params['n_stim_training'], 3)) # x, v, action_idx

v_lim_frac = .7
v_lim = (v_lim_frac * np.min(tp[:, 2]), v_lim_frac * np.max(tp[:, 2]))
#v_grid = np.linspace(v_lim[0], v_lim[1], params['n_training_v'])
#VI.RNG.shuffle(v_training)


v_stim_cnt = 0

stim_params = [0., 0.5, 0., 0.]

rewards_bg = []
rewards = []
stim_pos = []
stim_v = []

n_stim = 0
all_mp = []

for i_cycle in xrange(params['n_training_cycles']):
    for i_v in xrange(params['n_training_v']):
        stim_params[2] = training_speeds[i_v]

        BG.reset_pool_of_possible_actions(v_stim=stim_params[2])
        for i_x in xrange(params['n_training_x']):

            # get a new starting position somewhere in the periphery
            pm = utils.get_plus_minus(np.random)
            if pm > 0:
                stim_params[0] = np.random.uniform(.5 + params['center_stim_width'], 1.)
            else:
                stim_params[0] = np.random.uniform(0, .5 - params['center_stim_width'])

            while True:
                # select a random action taking only the direction of the speed into account
                print 'BG possible actions:', BG.all_action_idx
                rnd_action = BG.get_random_action(stim_params[2])
                if rnd_action == False:
                    # reinitialize the stimulus
                    break
                else:
                    rnd_action_idx, v_rnd = rnd_action
            
                # here would be a stimulus representation
                n_stim += 1
                all_actions_trained.append(rnd_action_idx)

                all_mp.append(stim_params)
                # evaluation of previous action
                x_old = stim_params[0]
                stim_pos.append(stim_params[0])
                stim_v.append(stim_params[2])
                R_BG = BG.get_reward_from_action(rnd_action_idx, stim_params)
                rewards_bg.append(R_BG)
                stim_params = utils.get_next_stim(params, stim_params, v_rnd)
                x_stim_only.append(utils.get_next_stim(params, stim_params, 0.)
                stim_params = list(stim_params)
                R = utils.get_reward_from_perceived_states(x_old, stim_params[0])
                rewards.append(R)
                print 'cycle %d i_v %d v_stim=%.2f action (%d, v=%.2f) i_x %d x = (before action) %.2f (after action %.2f)\tR_BG=%.2f R = %.2f' % (i_cycle, i_v, stim_params[2], rnd_action_idx, v_rnd, i_x, x_old, stim_params[0], R_BG, R)


from test_reward_schedule import plot_reward_schedule
plot_reward_schedule(stim_pos, rewards)

fig1 = pylab.figure()
ax = fig1.add_subplot(211)
ax2 = fig1.add_subplot(212)

#pylab.show()

#exit(1)

all_mp = np.array(all_mp)
#print 'Saving training sequence parameters to:', params['training_sequence_fn']
#np.savetxt(params['training_sequence_fn'], all_mp)
#np.savetxt(params['action_indices_fn'], d1_actions_trained)


#print 'all_mp', all_mp
#print 'stimtype', stim_type
#print 'all_actions_trained', all_actions_trained
#print 'positions_trained:', positions_trained


ax.plot(range(n_stim), all_mp[:, 0])
ax2.plot(range(n_stim), all_mp[:, 2])
ax.set_ylabel('Retinal displacement')
ax2.set_ylabel('Stimulus speed')
for i_ in xrange(n_stim):
#    if stim_type[i_] == 2:
#        ax.scatter(i_, all_mp[i_, 0], marker='v', c='r', s=100)
#    else:
#        ax.scatter(i_, all_mp[i_, 0], marker='o', c='b', s=100)
    ax.text(i_, all_mp[i_, 0] + 0.1, '%d' % all_actions_trained[i_])
ax.set_xlim((0, n_stim))
ax2.set_xlim((0, n_stim))


pylab.show()
exit(1)

#fig2 = pylab.figure()
#ax1 = fig2.add_subplot(111)
#n_bins = (np.max(d1_actions_trained) - np.min(d1_actions_trained))
#cnt, bins = np.histogram(d1_actions_trained, bins=n_bins, range=(np.min(d1_actions_trained), np.max(d1_actions_trained)))
#binwidth = .5 * (bins[1] - bins[0])
#ax1.bar(bins[:-1], cnt, width=binwidth, label='d1 trained', facecolor='b')

#n_bins = (np.max(d2_actions_trained) - np.min(d2_actions_trained))
#cnt, bins = np.histogram(d2_actions_trained, bins=n_bins, range=(np.min(d2_actions_trained), np.max(d2_actions_trained)))
#binwidth = .5 * (bins[1] - bins[0])
#ax1.bar(bins[:-1] + binwidth, cnt, width=binwidth, label='d2 trained', facecolor='r')
#pylab.legend()

#ax1.set_xlabel('Actions taken')
#ax1.set_ylabel('Count')

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


