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


training_stimuli = VI.create_training_sequence_RBL_cycle_blocks()
print 'training_stimuli block', training_stimuli


training_stimuli = VI.create_training_sequence_RBL_mixed_within_a_cycle()
print 'training_stimuli mixed ', training_stimuli
