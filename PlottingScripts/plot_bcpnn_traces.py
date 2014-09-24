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


class TracePlotter(object):

    def __init__(self, params, cell_type_post=None):
        self.params = params
        if cell_type_post == None:
            self.cell_type_post = 'd1'
        else:
            self.cell_type_post = cell_type_post 

        self.d_pre = None   # spike times within a certain interval
        self.d_post = None

        
    def load_spikes(self, fn_pre, fn_post):
        print 'TracePlotter loads:', fn_pre
        self.pre_spikes = np.loadtxt(fn_pre)
        print 'TracePlotter loads:', fn_post
        self.post_spikes = np.loadtxt(fn_post)


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


    def compute_traces(self, pre_gids, post_gids, it_range, K_vec=None):
        """
        K_vec -- vector holding Kappa values, must of same length as the pre-synaptic spike trace after calling convert_spiketrain_to_trace
        """
        self.t_range = (it_range[0] * self.params['t_iteration'], it_range[1] * self.params['t_iteration'])
        print 'DEBUG', self.t_range, self.params['t_iteration'], it_range
        self.d_pre = utils.get_spiketimes_within_interval(self.pre_spikes, self.t_range[0], self.t_range[1])
        self.d_post = utils.get_spiketimes_within_interval(self.post_spikes, self.t_range[0], self.t_range[1])
        bcpnn_traces = []
        gid_pairs = []
        for pre_gid in pre_gids:
            idx_pre = (np.array(self.d_pre[:, 0] == pre_gid)).nonzero()[0]
            st_pre = self.d_pre[idx_pre, 1]
            for post_gid in post_gids:
                idx_post = (np.array(self.d_post[:, 0] == post_gid)).nonzero()[0]
                st_post = self.d_post[idx_post, 1]
                s_pre = BCPNN.convert_spiketrain_to_trace(st_pre, self.t_range[1])
                s_post = BCPNN.convert_spiketrain_to_trace(st_post, self.t_range[1])

                wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, self.params['params_synapse_%s_MT_BG' % self.cell_type_post], K_vec=K_vec)
                bcpnn_traces.append([wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post])
                gid_pairs.append((pre_gid, post_gid))

        return bcpnn_traces, gid_pairs



    def plot_trace_with_spikes(self, bcpnn_traces, bcpnn_params, dt, output_fn=None, fig=None, \
            color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=None):
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
        ax1.legend(plots, labels_z)
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
        ax5.legend(plots, labels_p)
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
        ax3.legend(plots, labels_p)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w)
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_)
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

        if K_vec != None:
            p1, = ax2.plot(t_axis, K_vec, c='k', lw=linewidth)

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
        ax1.legend(plots, labels_z)
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
        ax2.legend(plots, labels_p)
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
        ax3.legend(plots, labels_p)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w)
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c=color_pre, lw=linewidth)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_)
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
            # 
            fn = self.params['mpn_bgd1_merged_conn_fn']
        else:
            # load the realized connections -- with weights in [nS] !!!
            fn = self.params['connections_folder'] + 'merged_mpn_bg_%s_connections_debug.txt' % self.cell_type_post
        

if __name__ == '__main__':

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    cell_type_post = 'd2'
    bcpnn_params = params['params_synapse_%s_MT_BG' % cell_type_post]
    bcpnn_params['K'] = 1.
    dt = params['dt']
    fn_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    fn_post = params['spiketimes_folder'] + params['%s_spikes_fn_merged_all' % cell_type_post]
    if (not os.path.exists(fn_pre)) or (not os.path.exists(fn_post)):
        utils.merge_spikes(params)
    
    it_range = (0, 4)
#    it_range = (0, 3)
    TP = TracePlotter(params, cell_type_post)
    TP.load_spikes(fn_pre, fn_post)
    n_pre = 2
    n_post = 2
    pre_gids, post_gids = TP.select_cells(n_pre=n_pre, n_post=n_post, it_range=it_range)
#    pre_gids = [2238]
#    pre_gids = [1158]
#    post_gids = [5023]

    plot_range = (0, 10)
    w = TP.get_weights(pre_gids, post_gids)
    all_traces, gid_pairs = TP.compute_traces(pre_gids, post_gids, plot_range)
    output_fn_base = params['figures_folder'] + 'bcpnn_trace_'
    for i_, traces in enumerate(all_traces):
        output_fn = output_fn_base + '%d_%d.png' % (gid_pairs[i_][0], gid_pairs[i_][1])
        info_txt = 'Pre: %d  Post: %d' % (gid_pairs[i_][0], gid_pairs[i_][1])
        TP.plot_trace(traces, bcpnn_params, dt, output_fn=output_fn, info_txt=info_txt)
    pylab.show()

