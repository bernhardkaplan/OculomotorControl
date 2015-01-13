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
import matplotlib.gridspec as gridspec

def sigmoid(x, a, b, c, d, tau):
    # d = limit value for x -> - infinity
    # a, b = limit value for x -> + infinity
    # tau, c = position for transition
    f_x = a / (b + d * np.exp(-tau * (x - c)))
    return f_x

def update_rcParams():

    plot_params = {'backend': 'png',
                  'axes.labelsize': 28,
                  'axes.titlesize': 32,
                  'text.fontsize': 20,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'legend.pad': 0.2,     # empty space around the legend box
                  'legend.fontsize': 14,
                   'lines.markersize': 1,
                   'lines.markeredgewidth': 0.,
                   'lines.linewidth': 4,
                  'font.size': 12,
                  'path.simplify': False,
                  'figure.subplot.left':.17,
                  'figure.subplot.bottom':.12,
                  'figure.subplot.right':.94,
                  'figure.subplot.top':.92,
                  'figure.subplot.hspace':.25, 
                  'figure.subplot.wspace':.25, 
                  'figure.figsize': get_fig_size(1200, portrait=False)}
    pylab.rcParams.update(plot_params)
    return plot_params

if __name__ == '__main__':

    GP = simulation_parameters.global_parameters()
    params = GP.params
    BG = BasalGanglia.BasalGanglia(params, dummy=True)
    all_actions_v = BG.action_bins_x
#    actions_v = [all_actions_v[0], all_actions_v[1], all_actions_v[2],  \
#                all_actions_v[9], \
#                all_actions_v[-1], all_actions_v[-2], all_actions_v[-3]]
    actions_v = all_actions_v
    n_actions_to_plot = len(actions_v)

    plot_params = update_rcParams()
    x_start = .0
    x_stop = 1.0
    n_x = 1000
    x = np.linspace(x_start, x_stop, n_x)
    x_center = 0.5

    K_max = params['pos_kappa']
    K_min = params['neg_kappa']
    stim_speeds = [-1.5, -.1, 1.]

#    x_pre_range = [0.05, 0.2, 0.5]
    x_pre_range = [0.00, 0.2, 0.5]
    linecolors = ['b', 'g', 'r', 'k']
    n_curves = len(x_pre_range)

    action_idx = range(params['n_actions'])
    cmap_actions = matplotlib.cm.copper
    norm_actions = matplotlib.colors.Normalize(vmin=np.min(action_idx), vmax=np.max(action_idx))#, clip=True)
    m_actions = matplotlib.cm.ScalarMappable(norm=norm_actions, cmap=cmap_actions)
    m_actions.set_array(np.arange(np.min(action_idx), np.max(action_idx), .1))
    rgba_colors_actions = m_actions.to_rgba(action_idx)

    linestyles = ['-', ':', '--', '-.']
    markers = ['o', '*', 'D', '^']

    fig = pylab.figure()
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1],
                          height_ratios=[1,1])
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    n_pos_reward = np.zeros((len(stim_speeds), n_curves))
    n_neg_reward = np.zeros((len(stim_speeds), n_curves))
    problematic_stimuli_hard = []
    problematic_stimuli_soft = []

    plots = []
    labels = []
    for i_stim, v_stim in enumerate(stim_speeds):
        a = 1.
        b = 1.
        d = 1.
        ls = '-'
        lw = 3
        for i_, x_pre_action in enumerate(x_pre_range): 
            print '\n\tNew x_pre_action: %.1f' % (x_pre_action)
            c, tau = utils.get_sigmoid_params(params, x_pre_action, v_stim)
            y = K_max - (K_max - K_min) * utils.sigmoid(np.abs(x - x_center), a, b, c, d, tau)
#            label_txt = '$\\tau=%.1f, c=%.2f, x_{stim}^{pre action}=%.1f v_{stim}=%.1f$ ' % (tau, c, x_pre_action, v_stim)
            label_txt = '$v_{stim}=%.1f\ \\tau=%.1f\ c=%.2f$' % (v_stim, tau, c)
            label_txt = '$v_{stim}=%.1f$' % (v_stim)
            if n_curves > 1:
#                color = rgba_colors[i_]
                color = linecolors[i_]
            else:
                color = 'b'
#            ax1.plot(x, y, color=color, ls=linestyles[i_stim % len(linestyles)])
#            ax1.plot(x, y, color=color, ls=linestyles[i_stim % len(linestyles)], label=label_txt)
            p, = ax1.plot(x, y, color=color, ls=linestyles[i_stim % len(linestyles)])

            stim_params = (x_pre_action, .5, v_stim, .0)

#            x_post_action = np.zeros(params['n_actions'])
#            for i_a in xrange(params['n_actions']):
#                x_post_action[i_a] = utils.get_next_stim(params, stim_params, actions_v[i_a])[0]

            x_post_action = np.zeros(n_actions_to_plot)
            for i_a in xrange(n_actions_to_plot):
                x_post_action[i_a] = utils.get_next_stim(params, stim_params, actions_v[i_a])[0]

                R = K_max - (K_max - K_min) * utils.sigmoid(np.abs(x_post_action[i_a] - x_center), a, b, c, d, tau)
                if R > 0:
                    print '\tPos reward R=%.1e\tfor v_stim %.1f\tx_pre: %1f\taction %d (%.1f) --> x_post: %.1f' % (R, v_stim, x_pre_action, i_a, actions_v[i_a], x_post_action[i_a])
                    n_pos_reward[i_stim, i_] += 1
                elif R <= 0:
                    n_neg_reward[i_stim, i_] += 1
#                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), alpha=0.4, marker='o', ls=linestyles[i_stim % len(linestyles)], markersize=12, mfc=color, color=color)#, lw=2)
                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), alpha=0.4, ls=linestyles[i_stim % len(linestyles)], color=color)#, lw=2)
#                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), alpha=0.4, marker=markers[i_stim % len(markers)], ls=linestyles[i_stim % len(linestyles)], markersize=10, markeredgewidth=1, mfc=color, color=color, lw=2)
                ax1.plot(x_post_action[i_a], R, marker='o', markersize=10, mfc=color)
                ax1.plot(x_pre_action, 0., marker='*', markersize=25, mfc=color, markeredgewidth=1)

        plots.append(p)
        labels.append(label_txt)
#                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), alpha=1.0, marker='*', ls=linestyles[i_stim % len(linestyles)], markersize=20, markeredgewidth=1, mfc=color, color=color, lw=2)

    #            p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), ls=ls, marker='o', markersize=5, mfc=color, color=rgba_colors_actions[i_a], lw=2)

    #            if n_curves > 1:
    #            else:
    #                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), ls=ls, color='b', lw=lw)

    #        R[i_v, i_], sigma_r = utils.get_reward_gauss(x_post_action[i_v, i_], stim_params, params)

    #        y0 = f(0, a, b, c, d, tau) # y0 = a / (b + d * np.exp( tau * c))
    #        y_max = np.max(y)
    #        print 'f(0) = %.3e\t check: a / (b + d * exp(tau * c)) = %.3e ' % (y0, a / (b + d * np.exp( tau[0] * c[0])))
    #        print 'f_max = %.3f' % (y_max)

    # evaluate the reward function based on the number of actions that have been
    # rewarded positively or negatively 
    for i_stim, v_stim in enumerate(stim_speeds):
        for i_, x_pre_action in enumerate(x_pre_range): 
            if np.abs(x_pre_action - x_center) > 0.3:
                n_soft_thresh = 1
            else:
                n_soft_thresh = 6
            if n_pos_reward[i_stim, i_] < 1: 
                problematic_stimuli_hard.append((x_pre_action, v_stim))
            elif n_pos_reward[i_stim, i_] > n_soft_thresh: 
                problematic_stimuli_soft.append((x_pre_action, v_stim))
            print 'Number of positive (negative) rewards = %d (%d)\tfor v_stim %.1f\tx_pre: %1f' % (n_pos_reward[i_stim, i_], n_neg_reward[i_stim, i_], v_stim, x_pre_action)
            print 'Number of positive (negative) rewards = %d (%d)\tfor v_stim %.1f\tx_pre: %1f' % (n_pos_reward[i_stim, i_], n_neg_reward[i_stim, i_], v_stim, x_pre_action)
    print 'Stimuli that got rewarded too little:\n', np.array(problematic_stimuli_hard)
    print 'Stimuli that got rewarded too often (more than 1, or 6 than times):\n', np.array(problematic_stimuli_soft)
     
    
#    ax1.plot((x_center - c[0], x_center - c[0]), (ylim[0], ylim[1]), c='k', ls='--', label = '$c = %.1f$' % c[0])
#    ax1.plot((x_center - c[1], x_center - c[1]), (ylim[0], ylim[1]), c='k', ls='--', label = '$c = %.1f$' % c[1])

#    ax1.legend(loc='upper left')
#    ax1.legend((labels), loc='upper left')
    ax1.set_xlabel('Stimulus position before $x_{pre}$ and after $x\'$ action')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward function based on sigmoidals')

    ax1.set_xlim((-0.2, 0.9))
    ax1.set_ylim((K_min - 0.1, K_max + 0.1))
    ylim = ax1.get_ylim()
    xlim = ax1.get_xlim()
#    function_txt = '$R(x_{post}) = \\frac{1}{1 + exp(-\\ta(|x_{post} - .5| - c(x_{pre}, v_{stim})))}$'
#    function_txt = '$R(x\') = \\frac{1}{1 + exp(-\\ta(|x\' - .5| - c(x_{pre}, v_{stim})))}$'
#    function_txt = '$R(x\') = \\frac{1}{1 + exp(-\\tau\cdot(|x\' - .5| - c))}$'
    function_txt = '$R = \\frac{1}{1 + exp(-\\tau\cdot(|x\' - .5| - c))}$'
    ax1.annotate(function_txt, xy=(xlim[0]+0.04, 1.6),  xycoords='data',
                            xytext=(-0, 30), textcoords='offset points',
                                    bbox=dict(boxstyle="round", fc="1.0", alpha=1.0), fontsize=28)


    ax1.plot((xlim[0], xlim[1]), (0., 0.), ls='-', c='k', lw=3)
    ax1.plot((0.5, 0.5), (ylim[0], ylim[1]), ls='-', c='k', lw=3)


    # TAU vs x
    x_pre_range = np.linspace(0., 0.5, 100)
#    x_pre_range = (0., 0.5) # absolute displacement
    k_range = params['k_range']
    # k_range[0] --> affects the stimuli that start at x_pre_range[0], i.e. in the periphery
    # k_range[1] --> affects the stimuli that start at x_pre_range[1], near the center
#    tau = transform_quadratic(x_pre, 'neg', k_range, x_pre_range)
#    tau = utils.transform_linear(x_pre_range, k_range)
    k = k_range[0]

    ax2.set_title('Sigmoid parameters')
    ax2.set_ylabel('$\\tau\ (x_{pre})$')
    ax2.set_xlabel('$x_{pre}$')
#    ax2.plot(x_pre_range, tau)
    ax2.plot((x_pre_range[0], x_pre_range[-1]), k_range)

    yticks = ax2.get_yticks()
#    print 'delme', yticks[0].get_text()
#    ax2.set_yticklabels([i.get_text() for i in yticks[:-1]])
    ax2.set_yticklabels(['%d' % i for i in yticks[:-1]])

    # take into account how far the stimulus moves
    v_stim_max = 2.
    stim_speeds = [-1.5, -.1, 1.0]
    linestyles = ['-', ':', '--', '-.']

    for i_v, v_stim in enumerate(stim_speeds):
#        abs_speed_factor = utils.transform_linear(np.abs(v_stim), [0.5, 1.], [0., v_stim_max])


        dx = 20 * v_stim * params['t_iteration'] / params['t_cross_visual_field']
#        c_range = (0.5 - np.sign(v_stim) * dx, 0.02 - np.sign(v_stim) * dx) 
#        best_case = 0.5 - (v_stim + params['v_max_out']) * params['t_iteration'] / params['t_cross_visual_field']
#        best_case = 0.5 - (v_stim + params['v_max_out']) * params['t_iteration'] / params['t_cross_visual_field'] + 0.01
#        tolerance = params['reward_tolerance']
#        c_range = (best_case, tolerance)

        # c_range --> determines the transition point from neg->pos reward (exactly if |K_min| == K_max)
        # c_raneg[1] --> determines tolerance for giving reward near center
#        c = utils.transform_quadratic(x_pre_range, 'pos', c_range)
#        abs_speed_factor = np.abs(v_stim)
#        c *= abs_speed_factor


        best_case = 0.5 - (v_stim + params['v_max_out']) * params['t_iteration'] / params['t_cross_visual_field'] + 0.01
        tolerance = params['reward_tolerance']
        c_range = (best_case, tolerance)
    #    c = transform_quadratic(x_pre, 'pos', c_range, x_pre_range)
        c = utils.transform_linear(x_pre_range, c_range)

        ax3.plot(x_pre_range, c, ls=linestyles[i_v], label='$v_{stim}=%.1f$' % v_stim, color='k')

    ax3.set_ylabel('$c\ (x_{pre}, v_{stim})$')
    ax3.set_xlabel('$x_{pre}$')
    pylab.legend()

    output_fn = 'reward_function.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    pylab.show()
