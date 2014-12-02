import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import numpy as np
import pylab
import matplotlib
import utils
from FigureCreator import get_fig_size
import simulation_parameters
import BasalGanglia

def f(x, a, b, c, d, tau):
    # d = limit value for x -> - infinity
    # a, b = limit value for x -> + infinity
    # tau, c = position for transition
    f_x = a / (b + d * np.exp(-tau * (x - c)))
    return f_x

def get_sigmoid_params(x_pre):
    """
    Linearly map x_pre (= stimulus position) to two parameters for the reward function
    """
    x_pre_range = (0., 0.5) # absolute displacement
    tau_range = (20., 500.) 
    # tau_range[0] --> affects the stimuli that start at x_pre_range[0], i.e. in the periphery
    # tau_range[1] --> affects the stimuli that start at x_pre_range[1], near the center
    tau = utils.transform_linear(x_pre, tau_range, x_pre_range)
    c_range = (0.35, 0.03) 
    # c_range --> determines the transition point from neg->pos reward (exactly if |K_min| == K_max)
    # c_raneg[1] --> determines tolerance for giving reward near center
    c = utils.transform_linear(x_pre, c_range, x_pre_range)
    return c, tau


def update_rcParams():

    plot_params = {'backend': 'png',
                  'axes.labelsize': 24,
                  'axes.titlesize': 20,
                  'text.fontsize': 20,
                  'xtick.labelsize': 16,
                  'ytick.labelsize': 16,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 14,
                   'lines.markersize': 1,
                   'lines.markeredgewidth': 0.,
                   'lines.linewidth': 1,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.subplot.left':.17,
                  'figure.subplot.bottom':.12,
                  'figure.subplot.right':.94,
                  'figure.subplot.top':.92,
                  'figure.subplot.hspace':.08, 
                  'figure.subplot.wspace':.30, 
                  'figure.figsize': get_fig_size(1200, portrait=False)}
    pylab.rcParams.update(plot_params)
    return plot_params

if __name__ == '__main__':

    GP = simulation_parameters.global_parameters()
    params = GP.params
    BG = BasalGanglia.BasalGanglia(params, dummy=True)
    actions_v = BG.action_bins_x
    plot_params = update_rcParams()
    x_start = .0
    x_stop = 1.0
    n_x = 1000
    x = np.linspace(x_start, x_stop, n_x)
    x_center = 0.5

    K_max = 2.
    K_min = -2.
    v_stim = -1.0

    n_curves = 4
    x_pre_range = np.linspace(0.05, 0.5, n_curves, endpoint=True)

    # map number of parameters to colors
    if n_curves > 1:
        bounds = np.array(x_pre_range)
        cmap = matplotlib.cm.jet
        norm = matplotlib.colors.Normalize(vmin=np.min(bounds), vmax=np.max(bounds))#, clip=True)
        m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        m.set_array(np.arange(np.min(bounds), np.max(bounds), .1))
        rgba_colors = m.to_rgba(bounds)

    action_idx = range(params['n_actions'])
    cmap_actions = matplotlib.cm.copper
    norm_actions = matplotlib.colors.Normalize(vmin=np.min(action_idx), vmax=np.max(action_idx))#, clip=True)
    m_actions = matplotlib.cm.ScalarMappable(norm=norm_actions, cmap=cmap_actions)
    m_actions.set_array(np.arange(np.min(action_idx), np.max(action_idx), .1))
    rgba_colors_actions = m_actions.to_rgba(action_idx)

    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    title = 'Reward function, $v_{stim}=%.1f$' % v_stim
    ax1.set_title(title)
    if n_curves > 1:
        cb = fig.colorbar(m)
        cb.set_label('$|x_{stim}^{pre action}|$')


    a = 1.
    b = 1.
    d = 1.
    ls = '-'
    lw = 3
    for i_, x_pre_action in enumerate(x_pre_range): 
        c, tau = get_sigmoid_params(x_pre_action)
        y = K_max - (K_max - K_min) * f(np.abs(x - x_center), a, b, c, d, tau)
        label_txt = '$\\tau=%.1f, c=%.1f, x_{stim}^{pre action}=%.1f$' % (tau, c, x_pre_action)
        if n_curves > 1:
            color = rgba_colors[i_]
        else:
            color = 'b'
        ax1.plot(x, y, ls='-', color=color, label=label_txt, lw=3)

        stim_params = (x_pre_action, .5, v_stim, .0)

        x_post_action = np.zeros(params['n_actions'])
        for i_a in xrange(params['n_actions']):
            x_post_action[i_a] = utils.get_next_stim(params, stim_params, actions_v[i_a])[0]

            R = K_max - (K_max - K_min) * f(np.abs(x_post_action[i_a] - x_center), a, b, c, d, tau)
            print 'debug', rgba_colors_actions[i_a]
            p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), ls='-', alpha=0.3, marker='*', markersize=10, mfc=color, color=color, lw=2)
#            p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), ls=ls, marker='o', markersize=5, mfc=color, color=rgba_colors_actions[i_a], lw=2)

#            if n_curves > 1:
#            else:
#                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), ls=ls, color='b', lw=lw)

#        R[i_v, i_], sigma_r = utils.get_reward_gauss(x_post_action[i_v, i_], stim_params, params)

#        y0 = f(0, a, b, c, d, tau) # y0 = a / (b + d * np.exp( tau * c))
#        y_max = np.max(y)
#        print 'f(0) = %.3e\t check: a / (b + d * exp(tau * c)) = %.3e ' % (y0, a / (b + d * np.exp( tau[0] * c[0])))
#        print 'f_max = %.3f' % (y_max)

    ax1.set_xlim((-0.1, 1.1))
    ylim = ax1.get_ylim()
    xlim = ax1.get_xlim()
    
#    ax1.plot((x_center - c[0], x_center - c[0]), (ylim[0], ylim[1]), c='k', ls='--', label = '$c = %.1f$' % c[0])
#    ax1.plot((x_center - c[1], x_center - c[1]), (ylim[0], ylim[1]), c='k', ls='--', label = '$c = %.1f$' % c[1])
    ax1.plot((xlim[0], xlim[1]), (0., 0.), ls='-', c='k', lw=3)
    ax1.plot((0.5, 0.5), (ylim[0], ylim[1]), ls='-', c='k', lw=3)

    ax1.legend(loc='upper right')
    ax1.set_xlabel('$x_{post}$')
    ax1.set_ylabel('Reward')

    pylab.show()
