# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

import simulation_parameters
import utils
import numpy as np
folder = 'Training_nRF500_expSyn_d1recFalse_nStim10x20_it25_tsim40000_taup20000/'
params = utils.load_params(folder)
d = np.loadtxt(params['actions_taken_fn'])

# <codecell>

fig = pylab.figure()
ax = fig.add_subplot(111)
print 'n_actions', params['n_actions']
print 'd:', d[:, 2]
cnt, bins = np.histogram(d[:, 2], bins=params['n_actions'], range=(0, params['n_actions']))
print 'bins', bins
ax.bar(bins[:-1], cnt, width=bins[1]-bins[0])
ax.set_xlabel('Actions taken')
ax.set_ylabel('Count')
ax.set_xlim((0, params['n_actions']))

# <codecell>


# <codecell>


