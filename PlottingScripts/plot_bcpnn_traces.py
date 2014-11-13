import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

import sys
import os
import utils
import re
import numpy as np
import pylab
import MergeSpikefiles
import PlotMPNActivity
import BCPNN
import simulation_parameters
import FigureCreator
import json

def create_K_vectors(params, stim_range, dt=0.1, tgt_cell_type='d1'):
    
    rewards = np.loadtxt(params['K_values_fn']) 
    n_stim = stim_range[1] - stim_range[0]
    t_max = n_stim * params['n_iterations_per_stim'] * params['t_iteration']
    n = np.int(t_max/ dt) + 1 # +1 because the length needs to be the same computed in the BCPNN module after convert_spiketrain_to_trace
    K_vec_compute = np.zeros(n)
    K_vec_plot = np.zeros(n)
    for it_ in xrange(stim_range[0] * params['n_iterations_per_stim'], stim_range[1] * params['n_iterations_per_stim']):
        idx_1 = np.int((it_ - stim_range[0] * params['n_iterations_per_stim']) * params['t_iteration'] / dt) 
        idx_2 = np.int((it_ + 1 - stim_range[0] * params['n_iterations_per_stim']) * params['t_iteration'] / dt) 
        if (rewards[it_] < 0) and tgt_cell_type == 'd1':
            R = 0
        elif (rewards[it_] < 0) and tgt_cell_type == 'd2':
            R = -rewards[it_]
        elif (rewards[it_] > 0) and tgt_cell_type == 'd2':
            R = 0
        else:
            R = rewards[it_]
        K_vec_compute[idx_1:idx_2] = R * np.ones(idx_2 - idx_1)
        K_vec_plot[idx_1:idx_2] = rewards[it_] * np.ones(idx_2 - idx_1)
    return K_vec_compute, K_vec_plot



class TracePlotter(object):

    def __init__(self, params, cell_type_post=None):
        self.params = params
        if cell_type_post == None:
            self.cell_type_post = 'd1'
        else:
            self.cell_type_post = cell_type_post 

        self.d_pre = None   # spike times within a certain interval
        self.d_post = None

        f = file(params['bg_gids_fn'], 'r')
        self.bg_gids = json.load(f) 
        
    def load_spikes(self, fn_pre, fn_post):
        print 'TracePlotter loads:', fn_pre
        self.pre_spikes = np.loadtxt(fn_pre)
        print 'TracePlotter loads:', fn_post
        self.post_spikes = np.loadtxt(fn_post)


    def select_cells_most_active_neurons(self, spike_data, n_cells, it_range_for_selection):
        """
        """
        t_range = (it_range_for_selection[0] * self.params['t_iteration'], it_range_for_selection[1] * self.params['t_iteration'])
        d = utils.get_spiketimes_within_interval(spike_data, t_range[0], t_range[1])
        (gids, nspikes) = utils.get_most_active_neurons(d, n_cells)
        gids = gids.astype(np.int)
        return gids


    def select_cells(self, pre_gids=None, post_gids=None, n_pre=1, n_post=1, it_range=None):
        if it_range == None:
            self.it_range = (0, 1)
            print 'No it_range given to select_cells, using:', self.it_range
        else:
            self.it_range = it_range
        self.t_range = (self.it_range[0] * self.params['t_iteration'], self.it_range[1] * self.params['t_iteration'])
        self.d_pre = utils.get_spiketimes_within_interval(self.pre_spikes, self.t_range[0], self.t_range[1])
        self.d_post = utils.get_spiketimes_within_interval(self.post_spikes, self.t_range[0], self.t_range[1])
#        print 'Most active neurons during t_range', self.t_range
        if pre_gids == None:
            mpn_gids = np.unique(self.d_pre[:, 0])
            (pre_gids, nspikes) = utils.get_most_active_neurons(self.d_pre, n_pre)
            pre_gids = pre_gids.astype(np.int)
#        print 'Pre_gids (%d most active pre-synaptic neurons)' % (n_pre), list(pre_gids), 'nspikes:', nspikes

        if post_gids == None:
            mpn_gids = np.unique(self.d_post[:, 0])
            (post_gids, nspikes) = utils.get_most_active_neurons(self.d_post, n_post)
            post_gids = post_gids.astype(np.int)
#        print 'Post (%d most active post-synaptic neurons)' % (n_post), list(post_gids), 'nspikes:', nspikes
        return pre_gids, post_gids


    def compute_traces(self, pre_gids, post_gids, it_range, K_vec=None, gain=1.):
        """
        K_vec -- vector holding Kappa values, must of same length as the pre-synaptic spike trace after calling convert_spiketrain_to_trace
        """
        self.t_range = (it_range[0] * self.params['t_iteration'], it_range[1] * self.params['t_iteration'])
        self.d_pre = utils.get_spiketimes_within_interval(self.pre_spikes, self.t_range[0], self.t_range[1])
        self.d_post = utils.get_spiketimes_within_interval(self.post_spikes, self.t_range[0], self.t_range[1])
        bcpnn_traces = []
        gid_pairs = []
        self.params['params_synapse_%s_MT_BG' % self.cell_type_post]['gain'] = gain
        for pre_gid in pre_gids:
            idx_pre = (np.array(self.d_pre[:, 0] == pre_gid)).nonzero()[0]
            st_pre = self.d_pre[idx_pre, 1]
            for post_gid in post_gids:
                idx_post = (np.array(self.d_post[:, 0] == post_gid)).nonzero()[0]
                st_post = self.d_post[idx_post, 1]
                s_pre = BCPNN.convert_spiketrain_to_trace(st_pre, self.t_range[1], t_min=self.t_range[0])
                s_post = BCPNN.convert_spiketrain_to_trace(st_post, self.t_range[1], t_min=self.t_range[0])

                wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, self.params['params_synapse_%s_MT_BG' % self.cell_type_post], K_vec=K_vec)
                bcpnn_traces.append([wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post])
                gid_pairs.append((pre_gid, post_gid))

        return bcpnn_traces, gid_pairs



    def plot_trace_with_spikes(self, bcpnn_traces, bcpnn_params, dt, t_offset=0., output_fn=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None, \
            extra_txt=None, w_title=None):
        # unpack the bcpnn_traces
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, pre_trace, post_trace = bcpnn_traces
        t_axis = dt * np.arange(zi.size) + t_offset
        plots = []
        pylab.rcParams.update({'figure.subplot.hspace': 0.50})
        if fig == None:
            fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
        else:
            ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()
        linewidth = 1
        self.title_fontsize = 24
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (bcpnn_params['tau_i'], bcpnn_params['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c=color_pre, lw=linewidth, ls=':')
        ax1.plot(t_axis, post_trace, c=color_post, lw=linewidth, ls=':')
        p1, = ax1.plot(t_axis, zi, c=color_pre, label='$z_i$', lw=linewidth)
        p2, = ax1.plot(t_axis, zj, c=color_post, label='$z_j$', lw=linewidth)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z, loc='upper left')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax5.plot(t_axis, pi, c=color_pre, lw=linewidth)
        p2, = ax5.plot(t_axis, pj, c=color_post, lw=linewidth)
        p3, = ax5.plot(t_axis, pij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax5.set_title('$\\tau_{p} = %d$ ms' % \
                (bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax5.legend(plots, labels_p, loc='upper left')
        ax5.set_xlabel('Time [ms]')
        ax5.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c=color_pre, lw=linewidth)
        p2, = ax3.plot(t_axis, ej, c=color_post, lw=linewidth)
        p3, = ax3.plot(t_axis, eij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (bcpnn_params['tau_e']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p, loc='upper left')
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w, loc='upper left')
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')
        if w_title != None:
            ax4.set_title(w_title)

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_, loc='upper left')
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

        if K_vec != None:
#            print 'debug t_axis K_vec sizes', len(t_axis), len(K_vec)
            p1, = ax2.plot(t_axis, K_vec, c='k', lw=linewidth)
            ax2.set_ylabel('Reward')
            ylim = (np.min(K_vec), np.max(K_vec))
#            ylim = ax2.get_ylim()
            y0 = ylim[0] - (ylim[1] - ylim[0]) * 0.05
            y1 = ylim[1] + (ylim[1] - ylim[0]) * 0.05
            ax2.set_ylim((y0, y1))
        if extra_txt != None:
            ax2.set_title(extra_txt)

        if output_fn != None:
            print 'Saving traces to:', output_fn
            pylab.savefig(output_fn)

        return fig




    def plot_trace(self, bcpnn_traces, bcpnn_params, dt, output_fn=None, info_txt=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-'):
        # unpack the bcpnn_traces
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, pre_trace, post_trace = bcpnn_traces
        t_axis = dt * np.arange(zi.size)
        plots = []
        pylab.rcParams.update({'figure.subplot.hspace': 0.50})
        if fig == None:
            fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
            ax1 = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)
            ax6 = fig.add_subplot(326)
        else:
            ax1, ax2, ax3, ax4, ax5, ax6 = fig.get_axes()
        linewidth = 1
        self.title_fontsize = 24
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (bcpnn_params['tau_i'], bcpnn_params['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c=color_pre, lw=linewidth, ls=':')
        ax1.plot(t_axis, post_trace, c=color_post, lw=linewidth, ls=':')
        p1, = ax1.plot(t_axis, zi, c=color_pre, label='$z_i$', lw=linewidth)
        p2, = ax1.plot(t_axis, zj, c=color_post, label='$z_j$', lw=linewidth)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z, loc='upper left')
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax2.plot(t_axis, pi, c=color_pre, lw=linewidth)
        p2, = ax2.plot(t_axis, pj, c=color_post, lw=linewidth)
        p3, = ax2.plot(t_axis, pij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax2.set_title('$\\tau_{p} = %d$ ms' % \
                (bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax2.legend(plots, labels_p, loc='upper left')
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c=color_pre, lw=linewidth)
        p2, = ax3.plot(t_axis, ej, c=color_post, lw=linewidth)
        p3, = ax3.plot(t_axis, eij, c=color_joint, lw=linewidth, ls=style_joint)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (bcpnn_params['tau_e']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p, loc='upper left')
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w, loc='upper left')
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_, loc='upper left')
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

        ax5.set_yticks([])
        ax5.set_xticks([])

        w_max, w_end, w_avg = np.max(wij), wij[-1], np.mean(wij)

#        info_txt_ = 'Weight max: %.2e\nWeight end: %.2e\nWeight avg: %.2e\n' % \
#                (w_max, w_end, w_avg)
#        info_txt_ += info_txt
        if info_txt != '':
            ax5.annotate(info_txt, (.02, .05), fontsize=18)


#        ax5.set_xticks([])
#        output_fn = self.params['figures_folder'] + 'traces_tauzi_%04d_tauzj%04d_taue%d_taup%d_dx%.2e_dv%.2e_vstim%.1e.png' % \
#                (bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'], self.dx, self.dv, self.v_stim)
#        output_fn = self.params['figures_folder'] + 'traces_dx%.2e_dv%.2e_vstim%.1e_tauzi_%04d_tauzj%04d_taue%d_taup%d.png' % \
#                (self.dx, self.dv, self.v_stim, bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'])


        if output_fn != None:
            print 'Saving traces to:', output_fn
            pylab.savefig(output_fn)

        return fig

    def get_weights(self, pre_gids, post_gids):

        if self.params['training']:
            fn = self.params['mpn_bgd1_merged_conn_fn']
        else:
            # load the realized connections -- with weights in [nS] !!!
            fn = self.params['connections_folder'] + 'merged_mpn_bg_%s_connections_debug.txt' % self.cell_type_post
        

def get_mean_weights(params, list_of_all_bcpnn_traces):

    n_ = len(list_of_all_bcpnn_traces)
    w_mean_values = np.zeros(n_)
    for i_ in xrange(n_):
        wij_trace = list_of_all_bcpnn_traces[i_][0]
        # average over certain time
        idx_0 = (params['n_stim'] - 1) * params['n_iterations_per_stim'] * params['t_iteration'] / params['dt']
        w_mean_values[i_] = np.mean(wij_trace[idx_0:])
    return w_mean_values


if __name__ == '__main__':

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    cell_type_post = 'd1'
    bcpnn_params = params['params_synapse_%s_MT_BG' % cell_type_post]
    bcpnn_params['K'] = params['pos_kappa']
    bcpnn_init = 0.001
    bcpnn_params['p_i'] = bcpnn_init
#    bcpnn_params['tau_i'] = 20.
#    bcpnn_params['tau_j'] = 10.
#    bcpnn_params['tau_e'] = .1

#    bcpnn_params['tau_i'] = float(sys.argv[2])
#    bcpnn_params['tau_e'] = float(sys.argv[3])
#    bcpnn_params['tau_p'] = float(sys.argv[4])
#    action_idx = int(sys.argv[5])
#    script_id = int(sys.argv[6]) # for identification of parameter set
#    param_set_id = int(sys.argv[7])
    action_idx = 13
    script_id = 0 
    param_set_id = 0

    dt = params['dt']
#    stim_range = (0, params['n_training_trials'])
    stim_range = (0, 68)
#    stim_range = (0, params['n_stim'])
    n_stim = stim_range[1] - stim_range[0]
#    plot_range = (0, n_stim * params['n_iterations_per_stim'])
    plot_range = (stim_range[0] * params['n_iterations_per_stim'], stim_range[1] * params['n_iterations_per_stim'])
    K_values = np.loadtxt(params['K_values_fn'])
    K_vec_compute, K_vec_plot = create_K_vectors(params, stim_range, dt, cell_type_post)

    gain = params['gain_MT_%s' % cell_type_post]
    fn_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    fn_post = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type_post]
    if (not os.path.exists(fn_pre)) or (not os.path.exists(fn_post)):
        utils.merge_spikes(params)
    
    TP = TracePlotter(params, cell_type_post)
    TP.load_spikes(fn_pre, fn_post)
    n_pre = 1
    n_post = 1
    it_range_pre_cell_selection = (38 * params['n_iterations_per_stim'], 3 + 38 * params['n_iterations_per_stim'])

#    it_range_pre_cell_selection = (0 + stim_range[0] * params['n_iterations_per_stim'], 3 + stim_range[0] * params['n_iterations_per_stim'])
#    it_range_pre_cell_selection = (0, 3)

#    pre_gids = TP.select_cells_most_active_neurons(TP.pre_spikes, n_pre, it_range_pre_cell_selection)
#    pre_gids = [1044]
    pre_gids = [int(sys.argv[2])] 
    post_gids = TP.bg_gids[cell_type_post][action_idx]

    print 'pre_gids:', pre_gids
    print 'post_gids:', post_gids

    all_traces, gid_pairs = TP.compute_traces(pre_gids, post_gids, plot_range, gain=gain, K_vec=K_vec_compute)
    output_fn_base = params['figures_folder'] + 'bcpnn_trace_'
    fig = None
    output_fn = None
    w_means = get_mean_weights(params, all_traces)
    w_mean, w_std = np.mean(w_means), np.std(w_means)
    print 'w_mean:', w_mean, '+-', w_std
    t_offset = stim_range[0] * params['n_iterations_per_stim'] * params['t_iteration']
    for i_, traces in enumerate(all_traces):
        info_txt = 'Action idx: %d' % (action_idx)
        w_title = '$w_{mean}=%.2f \pm %.2f$' % (w_mean, w_std)
        fig = TP.plot_trace_with_spikes(traces, bcpnn_params, dt, t_offset=t_offset, output_fn=output_fn, fig=fig, \
            K_vec=K_vec_plot, extra_txt=info_txt, w_title=w_title)
    
    # cell_type_post 
#    cell_type_post = 'd2'
#    post_gids = TP.bg_gids[cell_type_post][action_idx]
#    all_traces, gid_pairs = TP.compute_traces(pre_gids, post_gids, plot_range, gain=gain, K_vec=K_vec_compute)
#    output_fn_base = params['figures_folder'] + 'bcpnn_trace_'
#    w_means = get_mean_weights(params, all_traces)
#    w_mean, w_std = np.mean(w_means), np.std(w_means)
#    print 'w_mean:', w_mean, '+-', w_std
#    for i_, traces in enumerate(all_traces):
#        info_txt = 'Action idx: %d' % (action_idx)
#        w_title = '$w_{mean}=%.2f \pm %.2f$' % (w_mean, w_std)
#        fig = TP.plot_trace_with_spikes(traces, bcpnn_params, dt, output_fn=output_fn, fig=fig, \
#            K_vec=K_vec_plot, extra_txt=info_txt, w_title=w_title)
#    


            

    to_write = {'script_id' : script_id, \
            'bcpnn_params' : bcpnn_params, \
            'param_set_id' : param_set_id, \
            'w_mean': w_mean, \
            'w_std': w_std, \
            'action_idx' : action_idx}
    f_out = file(params['tmp_folder'] + 'w_mean_%d.json' % (script_id), 'w')
    json.dump(to_write, f_out, indent=2)

    output_fn = params['figures_folder'] + 'bcpnn_trace_action_mpn_%s_it%d-%d_piinit%.1e_tau_i%d_j%d_e%d_p%d_a%d_pregid%d.png' % \
            (cell_type_post, stim_range[0], stim_range[1], bcpnn_params['p_i'], bcpnn_params['tau_i'], bcpnn_params['tau_j'], bcpnn_params['tau_e'], bcpnn_params['tau_p'], action_idx, pre_gids[0])
    print 'Saving to:', output_fn
    fig.savefig(output_fn)
    if len(sys.argv) < 2:
        pylab.show()
