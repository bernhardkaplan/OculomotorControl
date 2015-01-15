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
    n_x = 100
    x = np.linspace(x_start, x_stop, n_x)
    x_center = 0.5

    K_max = params['pos_kappa']
    K_min = params['neg_kappa']

#    stim_speeds = [-1.5, 0., 1.5]
    stim_speeds = np.arange(-1.5, 1.6, 0.1)
#    stim_speeds = [0., -.5, -1.49]

#    x_pre_range = [0.5]

#    x_pre_range = [0., 1.0]
    x_pre_range = np.arange(0., 1.05, 0.05)
#    x_pre_range = np.arange(0.5, 1.05, 0.10)
    linecolors = ['b', 'g', 'r', 'k', 'c', 'y', 'orange', 'magenta', 'darkblue', 'lightgray', 'olive', 'sandybrown', 'pink', 'darkcyan']
    n_curves = len(x_pre_range)

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

            # the stimulus is actually at a different position when the perception stimulus reaches the cortex
            x_pre_action_with_delay = x_pre_action - v_stim * params['delay_input'] / params['t_cross_visual_field']
            c, tau = utils.get_sigmoid_params(params, x_pre_action_with_delay, v_stim)

            y = K_max - (K_max - K_min) * utils.sigmoid(np.abs(x - x_center), a, b, c, d, tau)
#            label_txt = '$v_{stim}=%.1f\ \\tau=%.1f\ c=%.2f$' % (v_stim, tau, c)
#            label_txt = '$v_{stim}=%.1f$' % (v_stim)
            color = linecolors[i_ % len(linecolors)]

            ax1.plot(x, y, color=color, ls=linestyles[i_stim % len(linestyles)]) # straight line
            stim_params = (x_pre_action, .5, v_stim, .0)
            stim_params_evaluation = (x_pre_action_with_delay, stim_params[1], stim_params[2], stim_params[3]) # the reward function 'knows' that a delay_input exists

            x_post_action = np.zeros(n_actions_to_plot)
            for i_a in xrange(n_actions_to_plot):
                x_post_action[i_a] = utils.get_next_stim(params, stim_params, actions_v[i_a])[0] # the next stimulus position takes into account both delay_input and delay_output

#                R = K_max - (K_max - K_min) * utils.sigmoid(np.abs(x_post_action[i_a] - x_center), a, b, c, d, tau)
                R = utils.get_reward_sigmoid(x_post_action[i_a], stim_params_evaluation, params)  # the reward function needs to operate on the updated positions, taking into account both delay_input, delay_output

                if R > 0:
#                    print '\tPos reward R=%.1e\tfor v_stim %.1f\tx_pre: %1f\taction %d (%.1f) --> x_post: %.1f' % (R, v_stim, x_pre_action, i_a, actions_v[i_a], x_post_action[i_a])
                    n_pos_reward[i_stim, i_] += 1
                elif R <= 0:
                    n_neg_reward[i_stim, i_] += 1
#                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), alpha=0.4, marker='o', ls=linestyles[i_stim % len(linestyles)], markersize=12, mfc=color, color=color)#, lw=2)

                ax1.plot((x_pre_action_with_delay, x_post_action[i_a]), (0., R), alpha=0.4, ls=linestyles[i_stim % len(linestyles)], color=color)#, lw=2)

#                p, = ax1.plot((x_pre_action, x_post_action[i_a]), (0., R), alpha=0.4, marker=markers[i_stim % len(markers)], ls=linestyles[i_stim % len(linestyles)], markersize=10, markeredgewidth=1, mfc=color, color=color, lw=2)
                ax1.plot(x_post_action[i_a], R, marker='o', markersize=10, mfc=color)
                p, = ax1.plot(x_pre_action_with_delay, 0., marker='D', markersize=15, mfc=color, markeredgewidth=1)
                ax1.plot(x_pre_action, 0., marker='*', markersize=20, mfc=color, markeredgewidth=1)

                # relative error
#                R_rel = (np.abs(x_pre_action - 0.5) - np.abs(x_post_action[i_a] - 0.5))/ np.abs(x_pre_action - 0.5)
#                ax1.plot((x_pre_action, x_post_action[i_a]), (0, R_rel), marker='*', markersize=25, color=color, mfc=color, markeredgewidth=1)

        label_txt = '$x_{pre}^{with input delay}$'
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
    print 'Stimuli that got rewarded too often (more than 1, or 6 than times):\n', np.array(problematic_stimuli_soft)
    print 'Stimuli that got rewarded too little:\n', np.array(problematic_stimuli_hard)
     
#    ax1.legend(loc='upper left')
    ax1.legend((labels), loc='upper left')
    ax1.set_xlabel('Stimulus position before $x_{pre}$ and after $x\'$ action')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward function based on sigmoidals')

    ax1.set_xlim((-0.2, 1.2))
    ax1.set_ylim((K_min - 0.1, K_max + 0.1))
    ylim = ax1.get_ylim()
    xlim = ax1.get_xlim()
#    function_txt = '$R = \\frac{1}{1 + exp(-\\tau\cdot(|x\' - .5| - c))}$'
#    ax1.annotate(function_txt, xy=(xlim[0]+0.04, 1.6),  xycoords='data',
#                            xytext=(-0, 30), textcoords='offset points',
#                                    bbox=dict(boxstyle="round", fc="1.0", alpha=1.0), fontsize=28)


    ax1.plot((xlim[0], xlim[1]), (0., 0.), ls='-', c='k', lw=3)
    ax1.plot((0.5, 0.5), (ylim[0], ylim[1]), ls='-', c='k', lw=3)


    # TAU vs x
#    x_pre_range = np.linspace(0., 0.5, 100)
    x_pre_range = np.linspace(0., 1.0, 500)
#    x_pre_range = (0., 0.5) # absolute displacement
    k_range = params['k_range']
    # k_range[0] --> affects the stimuli that start at x_pre_range[0], i.e. in the periphery
    # k_range[1] --> affects the stimuli that start at x_pre_range[1], near the center
#    tau = transform_quadratic(x_pre, 'neg', k_range, x_pre_range)
#    tau = utils.transform_linear(x_pre_range, k_range)
    k = k_range[0]

    ax2.set_title('Sigmoid parameters')
#    ax2.set_ylabel('Speed-tolerance-increase')
    ax2.set_xlabel('$x_{pre}$')
#    ax2.plot(x_pre_range, tau)
    ax2.plot((x_pre_range[0], x_pre_range[-1]), k_range)

    yticks = ax2.get_yticks()
#    print 'delme', yticks[0].get_text()
#    ax2.set_yticklabels([i.get_text() for i in yticks[:-1]])
    ax2.set_yticklabels(['%d' % i for i in yticks[:-1]])

    # take into account how far the stimulus moves
    v_stim_max = 2.
    linestyles = ['-', ':', '--', '-.']

    for i_v, v_stim in enumerate(stim_speeds):

        c_curve = np.zeros(x_pre_range.size)

        for i_x in xrange(x_pre_range.size):

            c, k_ = utils.get_sigmoid_params(params, x_pre_range[i_x], v_stim)
            c_curve[i_x] = c

        ax3.plot(x_pre_range, c_curve, ls=linestyles[i_v % len(linestyles)], label='$v_{stim}=%.1f$' % v_stim, color='k')

    ax3.set_ylabel('$c\ (x_{pre}, v_{stim})$')
    ax3.set_xlabel('$x_{pre}$')
    pylab.legend()

    output_fn = 'reward_function.png'
    print 'Saving figure to:', output_fn
    pylab.savefig(output_fn, dpi=200)
    print 'stim_speeds:', stim_speeds
#    pylab.show()
