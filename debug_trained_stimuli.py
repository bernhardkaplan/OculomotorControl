import os 
import sys
import json
import utils
import pylab
import numpy as np
import re

fn_base = 'Training_NEW5_nactions17_30_temp0.5_nC1__nStim30_'
trained_stimuli_d1 = []
trained_stimuli_d2 = []

for thing in os.listdir('.'):
    m = re.match(fn_base + '(\d+)-(\d+)_gainD1_0\.2_.*', thing)
    if m:
#        print thing, m.groups()
    
#    if thing.find(fn_base) != -1:
        params = utils.load_params(thing)
        d1_actions_trained = params['d1_actions_trained']
        for i_stim in d1_actions_trained.keys():
            if len(d1_actions_trained[i_stim]) != 0:
                trained_stimuli_d1 += d1_actions_trained[i_stim]

        d2_actions_trained = params['d2_actions_trained']
        for i_stim in d2_actions_trained.keys():
            if len(d2_actions_trained[i_stim]) != 0:
                trained_stimuli_d2 += d2_actions_trained[i_stim]

        print 'folder', params['folder_name']
        print 'trained_stimuli_d1:', trained_stimuli_d1


count, bins = np.histogram(trained_stimuli_d1)
print 'count, bins', count, '\n', bins
fig = pylab.figure()
ax1 = fig.add_subplot(211)
ax1.bar(bins[:-1], count, width=1)
ax1.set_ylabel('Count')
ax1.set_title('D1 training')
print 'D1 trainings:', count.sum()

count, bins = np.histogram(trained_stimuli_d2)
ax1 = fig.add_subplot(212)
ax1.bar(bins[:-1], count, width=1)
ax1.set_xlabel('Action index')
ax1.set_ylabel('Count')
ax1.set_title('D2 training')
print 'D2 trainings:', count.sum()


pylab.show()


