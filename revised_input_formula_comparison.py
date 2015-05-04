# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import pylab
import simulation_parameters
import utils
import numpy as np
import VisualInput
from copy import deepcopy


GP = simulation_parameters.global_parameters()
folder_name = 'DEBUG/'
GP.set_filenames(folder_name=folder_name)
params = GP.params
GP.create_folders()
GP.write_parameters_to_file()
#GP.write_parameters_to_file(params['params_fn_json']


params['f_max_stim'] = 2000.
params['blur_X'] = 0.1
params['blur_V'] = 0.3
VI_old = VisualInput.VisualInput(params)
stim_params = [0.25, 0.5, 1.5, 0.]
VI_old.current_motion_params = deepcopy(stim_params)
stim_old, ss_old = VI_old.compute_input(range(params['n_exc_mpn']), [0., 0.], use_additive_beta=False)
rfs_old = VI_old.rf_sizes


params_new = deepcopy(params)
params_new['blur_X'] = 0.00
params_new['blur_V'] = 0.00
params_new['f_max_stim'] = 1000. #* (1. - params['blur_X'] - params['blur_V'])
params_new['w_input_exc_mpn'] = 15.
#params_new['rf_size_x_multiplicator'] = 0.1
#params_new['rf_size_v_multiplicator'] = 0.3
VI_new = VisualInput.VisualInput(params_new)
VI_new.current_motion_params = deepcopy(stim_params)
stim_new, ss_new = VI_new.compute_input(range(params_new['n_exc_mpn']), [0., 0.], use_additive_beta=True)
rfs_new = VI_new.rf_sizes


n_spikes_in_new = np.zeros(params_new['n_exc_mpn'])
n_spikes_in_old = np.zeros(params['n_exc_mpn'])
for i_ in xrange(params['n_exc_mpn']):
    n_spikes_in_old[i_] = len(stim_old[i_])
    n_spikes_in_new[i_] = len(stim_new[i_])
print 'NEW vs OLD'
print 'Total number input spikes:'
print 'sum: %.2f\t%2f' % (n_spikes_in_new.sum(), n_spikes_in_old.sum())
print 'std: %.2f\t%2f' % (n_spikes_in_new.std(), n_spikes_in_old.std())
print 'max: %.2f\t%2f' % (n_spikes_in_new.max(), n_spikes_in_old.max())
print 'n_in: %d\t%d' % (n_spikes_in_new.nonzero()[0].size, n_spikes_in_old.nonzero()[0].size)
print 'frac: %.2f\t%.2f' % (100. * n_spikes_in_new.nonzero()[0].size / float(params['n_exc_mpn']), \
                            100. * n_spikes_in_old.nonzero()[0].size / params['n_exc_mpn'])

print '\nTotal number input conductance:'
print 'sum: %.2f\t%2f' % (n_spikes_in_new.sum() * params_new['w_input_exc_mpn'], n_spikes_in_old.sum() * params['w_input_exc_mpn'])
print 'std: %.2f\t%2f' % (n_spikes_in_new.std() * params_new['w_input_exc_mpn'], n_spikes_in_old.std() * params['w_input_exc_mpn'])
print 'max: %.2f\t%2f' % (n_spikes_in_new.max() * params_new['w_input_exc_mpn'], n_spikes_in_old.max() * params['w_input_exc_mpn'])
print 'n_in: %d\t%d' % (n_spikes_in_new.nonzero()[0].size, n_spikes_in_old.nonzero()[0].size)
print 'frac: %.2f\t%.2f' % (100. * n_spikes_in_new.nonzero()[0].size / float(params_new['n_exc_mpn']), \
                            100. * n_spikes_in_old.nonzero()[0].size / params['n_exc_mpn'])


print '\nRFS x'
print 'NEW vs OLD'
print 'min: %.3e\t%.3e' % (rfs_old[:, 0].min(), rfs_new[:, 0].min())
print 'max: %.4f\t%.4f' % (rfs_old[:, 0].max(), rfs_new[:, 0].max())
print 'mean: %.4f\t%.4f' % (rfs_old[:, 0].mean(), rfs_new[:, 0].mean())
print '\nRFS v'
print 'NEW vs OLD'
print 'min: %.3e\t%.3e' % (rfs_old[:, 2].min(), rfs_new[:, 2].min())
print 'max: %.4f\t%.4f' % (rfs_old[:, 2].max(), rfs_new[:, 2].max())
print 'mean: %.4f\t%.4f' % (rfs_old[:, 2].mean(), rfs_new[:, 2].mean())



fig = pylab.figure()
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.bar(range(params['n_exc_mpn']), n_spikes_in_old, width=1)
ax2.bar(range(params['n_exc_mpn']), n_spikes_in_new, width=1)
ax3.bar(range(params['n_exc_mpn']), n_spikes_in_old * params['w_input_exc_mpn'], width=1)
ax4.bar(range(params['n_exc_mpn']), n_spikes_in_new * params_new['w_input_exc_mpn'], width=1)

ax1.set_title('OLD')
ax2.set_title('NEW')
ax3.set_title('$G_in$ old')
ax4.set_title('$G_in$ new')
ax1.set_xlim((0, params['n_exc_mpn']))
ax2.set_xlim((0, params['n_exc_mpn']))


pylab.show()

