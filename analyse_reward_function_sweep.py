import json
import numpy as np


"""
Analyses the json output dictionary written by: PlottingScripts/show_reward_stimulus_maps.py
"""

fn = 'reward_function_quadrMap_parameter_sweep_delayIn0_delayOut0.json'
fn = 'reward_function_quadrMap_parameter_sweep_delayIn75_delayOut75.json'
f = file(fn, 'r')
d = json.load(f)

good_params = []

cnt_good = 0
#for v_ in d.values():
for key in d.keys():
    v_ = d[key]
    rp = [v_['n_actions'], v_['reward_function_speed_multiplicator_range'], v_['reward_tolerance'], v_['reward_transition_range']]
    if v_['n_no_pos_reward'] == 0:
        print 'OK', key, v_['n_no_pos_reward'], rp
        good_params.append((key, v_))
        cnt_good += 1

# sort according to the number of n_too_much_reward
n_too_much_reward_array = np.zeros((cnt_good, 2))
for i_, (key, gp) in enumerate(good_params):
    print gp['n_too_much_reward'], i_, key
    n_too_much_reward_array[i_, 0] = gp['n_too_much_reward']
    n_too_much_reward_array[i_, 1] = key
print 'that was n_too_much_reward, i_, key'

#idx_sorted = np.argsort(n_too_much_reward_array)
#print 'n_too_much_reward_array (sorted):', n_too_much_reward_array[idx_sorted]
#print 'good param identifier sorted according to n_too_much_reward:', idx_sorted

# extract those parameter sets that have the lowest n_too_much_reward
n_too_much_reward_min = np.min(n_too_much_reward_array[:, 0]) 
print 'n_too_much_reward_min:', n_too_much_reward_min
filtered_idx = np.where(n_too_much_reward_array[:, 0] == n_too_much_reward_min)[0]
keys_with_min_n_too_much_reward = n_too_much_reward_array[filtered_idx, 1]
print 'Good params: idx', filtered_idx, '\n', n_too_much_reward_array[filtered_idx, :]
print 'keys_with_min_n_too_much_reward ', keys_with_min_n_too_much_reward 

good_params_low_n_too_much_reward = {}
for fidx in keys_with_min_n_too_much_reward:
    p = d[str(int(fidx))]
#    print 'reward_tolerance', p['reward_tolerance'], 'n_actions:', p['n_actions'], 'reward_function_speed_multiplicator_range', p['reward_function_speed_multiplicator_range'], 'reward_transition_range', p['reward_transition_range']
    if p['n_actions'] == 15 and p['reward_tolerance'] == 0.03:
        print p
    good_params_low_n_too_much_reward[int(fidx)] = d

