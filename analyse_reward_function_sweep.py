import json
import numpy as np


fn = 'reward_functino_parameter_sweep_set_2.json'
f = file(fn, 'r')
d = json.load(f)

for v_ in d.values():
    rp = [v_['n_actions'], v_['reward_function_speed_multiplicator_range'], v_['reward_tolerance'], v_['reward_transition']]
    if v_['n_no_pos_reward'] == 0:
        print '\nOK', v_['n_no_pos_reward'], rp, '\n'
    else:
        print 'not OK', v_['n_no_pos_reward']


