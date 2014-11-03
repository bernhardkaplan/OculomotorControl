import os
import sys
import numpy as np
import json


folder_name_base = 'Training_DEBUG_RBL_titer25_2500_nStim6x1_taup50000_gain1.00/tmp/'

fn_base = 'w_mean_'
fns = []

for thing in os.listdir(folder_name_base):
    if thing.find(fn_base) != -1:
        fns.append(thing)

print 'Found files:', fns
results = {} 

print '\n\n\n'

# Create the dictionary with the results that you expect manually
parameter_set  = {}
for fn in fns:
    path = folder_name_base + fn
#    print 'Loading: ', path
    f = file(path, 'r')
    d = json.load(f)

    pid = d['param_set_id']
    w = d['w_mean']
#    print 'pid, w', pid, w, d['action_idx']
    if not results.has_key(d['action_idx']):
        results[d['action_idx']] = {}
    results[d['action_idx']][pid] = w
    parameter_set[pid] = d['bcpnn_params']


# compare the results that you're interested in
n_ = len(results[9].keys())
d_final = np.zeros((n_, 4)) # pid, w[10], w[9], w_diff, w_diff / w[9]

for i_, pid in enumerate(results[9].keys()):
    if results[10].has_key(pid):
        d_final[i_, 0] = pid
        d_final[i_, 1] = results[9][pid]
        d_final[i_, 2] = results[10][pid]
        d_final[i_, 3] = d_final[i_, 2] - d_final[i_, 1]

print 'Final results:', d_final
output_fn = 'bcpnn_sweep_wmean_results.dat'
print 'Saving to:', output_fn
np.savetxt(output_fn, d_final)

fp_out = 'bcpnn_parameter_sets.json'
f_p = file(fp_out, 'w')
print 'Writing to:', fp_out
json.dump(parameter_set, f_p, indent=2)
f_p.close()

idx = d_final[:, 3].argsort()
#print 'd_final sorted:', d_final[idx, 3]
for i_ in xrange(1, 5):
    print 'Maximal relative difference found at:', idx[-i_], 'w_diff_max:', d_final[idx[-i_], 3], '\nparameter set:', d_final[idx[-i_], 0], 'w_final_10:', d_final[idx[-i_], 2], results[10][idx[-i_]]
    print 'with bcpnn parameters:', parameter_set[d_final[idx[-i_], 0]]
