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
import json



def update_rcParams():

    plot_params = {'backend': 'png',
                  'axes.labelsize': 23,
                  'axes.titlesize': 24,
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
                  'figure.subplot.hspace':.28, 
                  'figure.subplot.wspace':.28, 
                  'figure.figsize': get_fig_size(1200, portrait=False)}
    pylab.rcParams.update(plot_params)
    return plot_params

if __name__ == '__main__':

    GP = simulation_parameters.global_parameters()
    params = GP.params

    plot_params = update_rcParams()
    n_x, n_v = 40, 40
    stim_speeds = np.linspace(-1.5, 1.5, n_v, endpoint=True)
    x_pre_range = np.linspace(0., 1.00, n_x, endpoint=True)

    a = 1.
    b = 1.
    d = 1.

    too_much_reward_thresh = 10
    linecolors = ['b', 'g', 'r', 'k', 'c', 'y', 'orange', 'magenta', 'darkblue', 'lightgray', 'olive', 'sandybrown', 'pink', 'darkcyan']
    linestyles = ['-', ':', '--', '-.']


    output_data = {}
    
    cnt_ = 0 
    plot = False
    for speed_mult in [[0.5, 1.0], [0.5, 1.5], [0.5, 2.0], [1.0, 1.5], [1.0, 2.0]]:
        params['reward_function_speed_multiplicator_range'] = speed_mult
        for n_actions in np.arange(15, 29, 2):
            params['n_actions'] = n_actions
            BG = BasalGanglia.BasalGanglia(params, dummy=True)
            all_actions_v = BG.action_bins_x
            actions_v = all_actions_v
            n_actions_to_plot = len(actions_v)
            for k_range in [[100, 100]]:#, [100, 10], [100, 5]]:
                params['reward_transition_range'] = k_range
                for rew_tol in np.arange(0.03, 0.08, 0.01):
                    params['reward_tolerance'] = rew_tol
                    n_pos_reward = np.zeros((len(stim_speeds), len(x_pre_range)))
                    n_neg_reward = np.zeros((len(stim_speeds), len(x_pre_range)))
                    for i_stim, v_stim in enumerate(stim_speeds):
                        for i_x, x_pre_action in enumerate(x_pre_range): 
                            x_pre_action_with_delay = x_pre_action - v_stim * params['delay_input'] / params['t_cross_visual_field']
                            x_post_action = np.zeros(n_actions_to_plot)
                            c, tau = utils.get_sigmoid_params(params, x_pre_action_with_delay, v_stim)
                            stim_params = (x_pre_action, .5, v_stim, .0)
                            stim_params_evaluation = (x_pre_action_with_delay, stim_params[1], stim_params[2], stim_params[3]) # the reward function 'knows' that a delay_input exists
                            for i_a in xrange(n_actions_to_plot):
                                x_post_action[i_a] = utils.get_next_stim(params, stim_params, actions_v[i_a], params['with_input_delay'], params['with_output_delay'])[0] # the next stimulus position takes into account both delay_input and delay_output
                                R = utils.get_reward_sigmoid(x_post_action[i_a], stim_params_evaluation, params)  # the reward function needs to operate on the updated positions, taking into account both delay_input, delay_output
                                if R > 0:
                                    n_pos_reward[i_stim, i_x] += 1
                                elif R <= 0:
                                    n_neg_reward[i_stim, i_x] += 1

                    xidx, yidx = np.where(n_pos_reward == 0)
                    n_no_pos_reward = xidx.size
                    xidx_, yid_ = np.where(n_pos_reward > too_much_reward_thresh)
                    n_too_much_reward = xidx_.size

                    output_data[cnt_] = {'reward_transition_range': k_range, 'reward_tolerance': rew_tol, 'n_actions': params['n_actions'], 'too_much_reward_thresh': too_much_reward_thresh, \
                            'reward_function_speed_multiplicator_range': params['reward_function_speed_multiplicator_range'], 'n_no_pos_reward': n_no_pos_reward, 'n_too_much_reward' : n_too_much_reward, \
                            'n_stim_tested': n_pos_reward.size, 'reward_transition': params['reward_transition'], 'map_reward_transition_speed': params['map_reward_transition_speed'], \
                            'map_reward_transition_point': params['map_reward_transition_point'], 'delay_input': params['delay_input'], 'delay_output': params['delay_output'], 't_iteration': params['t_iteration']}
                    print 'cnt %d' % cnt_, output_data[cnt_]

                    if plot:
                        fig = pylab.figure()
                        ax1 = fig.add_subplot(221)
                        print "plotting .... "
                        cax1 = ax1.pcolormesh(n_pos_reward, cmap='hot', vmin=0, vmax=too_much_reward_thresh)
                        ax1.set_ylim((0, n_pos_reward.shape[0]))
                        ax1.set_xlim((0, n_pos_reward.shape[1]))
                        cbar1 = pylab.colorbar(cax1)
                        cbar1.set_label('Num positive rewards')
                        ax1.set_ylabel('$v_{stim}$')
                        for x, y in zip(xidx, yidx):
                            ax1.plot(y+.5, x+.5, 'v', markersize=10, color='b')

                        ax2 = fig.add_subplot(223)
                        cax2 = ax2.pcolormesh(n_neg_reward, cmap='hot', vmin=0, vmax=params['n_actions'])
                        ax2.set_ylim((0, n_pos_reward.shape[0]))
                        ax2.set_xlim((0, n_pos_reward.shape[1]))
                        cbar2 = pylab.colorbar(cax2)
                        cbar2.set_label('Num negative rewards')
                        ax2.set_ylabel('$v_{stim}$')
                        ax2.set_xlabel('$x_{stim}$')

                        title = 'Quadratic C-mapping $n_{actions}=$%d Reward tolerance=%.2f\n speed mult: %.1f-%.1f k=%d' % (params['n_actions'], params['reward_tolerance'], \
                                params['reward_function_speed_multiplicator_range'][0], params['reward_function_speed_multiplicator_range'][1], \
                                params['reward_transition'])
                        ax1.set_title(title)
                        ax2.set_title('Too little reward: %d (%.1f),\ntoo much reward: %d (%.1f)' % (n_no_pos_reward, n_no_pos_reward/float(n_pos_reward.size) * 100., \
                                n_too_much_reward, n_too_much_reward / float(n_neg_reward.size) * 100.))

                        new_xticklabels = []
                        n_xticks = 5
                        xticklabels = np.linspace(x_pre_range[0], x_pre_range[-1], n_xticks, endpoint=True)
                        for x_ in xticklabels:
                            new_xticklabels.append('%.2f' % x_)
                        ax1.set_xticks(np.linspace(0, x_pre_range.size, n_xticks))
                        ax1.set_xticklabels(new_xticklabels)
                        ax2.set_xticks(np.linspace(0, x_pre_range.size, n_xticks))
                        ax2.set_xticklabels(new_xticklabels)

                        new_yticklabels = []
                        n_yticks = 5
                        yticklabels = np.linspace(stim_speeds[0], stim_speeds[-1], n_yticks, endpoint=True)
                        for x_ in yticklabels:
                            new_yticklabels.append('%.2f' % x_)
                        ax1.set_yticks(np.linspace(0, stim_speeds.size, n_yticks))
                        ax1.set_yticklabels(new_yticklabels)
                        ax2.set_yticks(np.linspace(0, stim_speeds.size, n_yticks))
                        ax2.set_yticklabels(new_yticklabels)

                        ax3 = fig.add_subplot(122)
                        x = np.linspace(0.0, 1.0, 1000)
                        x_center = 0.5
                        K_max = params['pos_kappa']
                        K_min = params['neg_kappa']
                        i_c = 0
                        for i_x, x_pre_action in enumerate([0.5, 0.25, 0.]):
                            for i_v, v_stim in enumerate([1.5, 0.5, 0.]):
                                x_pre_action_with_delay = x_pre_action - v_stim * params['delay_input'] / params['t_cross_visual_field']
                                c, tau = utils.get_sigmoid_params(params, x_pre_action_with_delay, v_stim)
                                y = K_max - (K_max - K_min) * utils.sigmoid(np.abs(x - x_center), a, b, c, d, tau)
                                ax3.plot(x, y, label='$x_{stim}^{delayed}=%.2f,\ v_{stim}=%.1f$' % (x_pre_action_with_delay, v_stim), c=linecolors[i_x % len(linecolors)], ls=linestyles[i_v % len(linestyles)])
                                pylab.legend()
                                i_c += 1
                        xlim = ax3.get_xlim()
                        ylim = ax3.get_ylim()
                        ax3.plot((xlim[0], xlim[1]), (0., 0.), c='k', ls='--', lw=3)
                        ax3.plot((.5, .5), (ylim[0], ylim[1]), c='k', ls='--', lw=3)
                        ax3.set_xlabel('$x_{pre\ action}$')
                        ax3.set_ylabel('Reward')
                        output_fn = 'reward_distribution_quadraticCmapping_nactions%d_rewTolerance_%.2f_speedMult_%.1f-%.1f_k%d_cnt%d.png' % (params['n_actions'], params['reward_tolerance'], \
                                params['reward_function_speed_multiplicator_range'][0], params['reward_function_speed_multiplicator_range'][1], \
                                params['reward_transition'], cnt_)
                        print 'Saving to:', output_fn
                        pylab.savefig(output_fn)

#                        output_fn = 'delme_debug_pd_cnt%d_maps.json' % cnt_
#                        f = file(output_fn, 'w')
#                        json.dump(params, f, indent=2)
#                        f.close()
#                        del fig

                    cnt_ += 1

    output_fn = 'reward_function_quadrMap_parameter_sweep_delayIn%d_delayOut%d.json' % (params['delay_input'], params['delay_output'])
    print 'Saving output data to:', output_fn
    f = file(output_fn, 'w')
    json.dump(output_data, f, indent=2)
    f.close()

#    pylab.show()
