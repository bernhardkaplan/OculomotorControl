import nest
import numpy as np

np.random.seed(0)

try:
    nest.Install('pt_module')
except:
    nest.Install('pt_module')

n_src = 10
n_tgt = n_src
src = nest.Create('iaf_cond_exp_bias', n_src)
tgt = nest.Create('iaf_cond_exp_bias', n_tgt)

src_idx = [1, 3, 5, 7]
tgt_idx = [i_ + n_src for i_ in src_idx]
weights = list(np.random.rand(len(src_idx)))
pi = list(np.random.rand(len(src_idx)))
pj = list(np.random.rand(len(src_idx)))
pij = list(np.random.rand(len(src_idx)))
delays = list(np.ones(len(src_idx)))

print 'DEBUG src', src, type(src)
print 'DEBUG tgt', tgt, type(tgt)
print 'weights', weights, type(weights)
print 'delays', delays, type(delays)


#param_dict = [ {'pi' : pi[i_]} for i_ in xrange(len(src_idx))]
param_dict = [ {'p_i' : pi[i_], 'p_j': pj[i_], 'p_ij': pij[i_], 'weight': weights[i_], 'delay': delays[i_]} for i_ in xrange(len(src_idx))]

#nest.Connect(src_idx, tgt_idx, weights, delays, 'bcpnn_synapse', params=param_dict)
nest.Connect(src_idx, tgt_idx, params=param_dict, model='bcpnn_synapse')


