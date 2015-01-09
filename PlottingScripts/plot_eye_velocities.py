import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)
import simulation_parameters
import utils
import numpy as np
import json
import pylab
import FigureCreator
import BasalGanglia
import matplotlib


plot_params = {'backend': 'png',
              'axes.labelsize': 20,
              'axes.titlesize': 20,
              'text.fontsize': 20,
              'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'legend.pad': 0.2,     # empty space around the legend box
              'legend.fontsize': 16,
               'lines.markersize': 1,
               'lines.linewidth': 3,
              'path.simplify': False,
              'figure.subplot.left' : 0.17, 
              'figure.subplot.bottom' : 0.15, 
              'figure.subplot.right' : 0.97, 
              'figure.subplot.top' : 0.92, 
              'figure.subplot.hspace' : 0.45,
              } 

pylab.rcParams.update(plot_params)


try: 
    params = utils.load_params_from_file(sys.argv[1])
except:
    import simulation_parameters
    param_tool = simulation_parameters.global_parameters()
    params = param_tool.params

# ---- Plot 2 D action_idx (int) vs Output Speed (vx)
fig = pylab.figure(figsize=(8, 8))
#fig = pylab.figure(figsize=FigureCreator.get_fig_size(800, portrait=False))
#fig = pylab.figure(figsize=FigureCreator.get_fig_size(800, portrait=True))
ax = fig.add_subplot(111)

bounds = range(params['n_actions'])
cmap = matplotlib.cm.jet
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
m.set_array(np.arange(bounds[0], bounds[-1], 1.))
rgba_colors = m.to_rgba(bounds)
cb = fig.colorbar(m)
cb.set_label('Action indices')#, fontsize=24)
#cb.set_label('Action indices')#, fontsize=24)

BG = BasalGanglia.BasalGanglia(params, dummy=True)
action_mapping = BG.action_bins_x
for i_ in xrange(params['n_actions']):
    ax.plot(i_, action_mapping[i_], 'o', color=rgba_colors[i_], markersize=10)
#ax.plot(range(len(action_mapping)), action_mapping, 'o', markersize=10)
ax.set_xlabel('Action index')
ax.set_ylabel('Action $v_x$ [a.u.]')
ax.set_title('Eye velocity of different actions')
ax.plot((-1, params['n_actions']), (0., 0.), ls='--', c='k', lw=3)
ax.set_xlim((0 - 0.50, params['n_actions'] - 1 + 0.50))
ax.set_ylim((min(action_mapping) - 0.50, max(action_mapping) + 0.50))
output_fn = params['figures_folder'] + 'vx_output_action.png'
print 'Saving output action mapping to:', output_fn
fig.savefig(output_fn, dpi=200)

pylab.show()
