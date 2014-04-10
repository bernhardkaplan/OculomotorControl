import os, sys, inspect
# use this if you want to include modules from a subforder
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
print 'cmd_subfolder', cmd_subfolder
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

    def __init__(self, params, it_range=None):
        self.params = params
        if it_range == None:
            self.it_range = (0, 1)
        else:
            self.it_range = it_range
        self.t_range = (self.it_range[0] * self.params['t_iteration'], self.it_range[1] * self.params['t_iteration'])
        self.dt = params['dt']
        self.bcpnn_params = self.params['params_synapse_d1_MT_BG']


        
    def load_spikes(self, fn_pre, fn_post):
        print 'TracePlotter loads:', fn_pre
        self.pre_spikes = np.loadtxt(fn_pre)
        print 'TracePlotter loads:', fn_post
        self.post_spikes = np.loadtxt(fn_post)

    def select_cells(self, pre_gids=None, post_gids=None):
        n_pre = 1
        n_post = 1

        if pre_gids == None:
            self.d_pre = utils.get_spiketimes_within_interval(self.pre_spikes, self.t_range[0], self.t_range[1])
            mpn_gids = np.unique(self.d_pre[:, 0])
            pre_gids = utils.get_most_active_neurons(self.d_pre, n_pre)
#        print 'Pre_gids', pre_gids

        if post_gids == None:
            self.d_post = utils.get_spiketimes_within_interval(self.post_spikes, self.t_range[0], self.t_range[1])
#            print 'DEBUG', self.d_post
            mpn_gids = np.unique(self.d_post[:, 0])
            post_gids = utils.get_most_active_neurons(self.d_post, n_post)
#        print 'post_gids', post_gids
        return pre_gids, post_gids

    def compute_traces(self, pre_gids, post_gids):
        bcpnn_traces = []
        gid_pairs = []
        for pre_gid in pre_gids:
            idx_pre = (self.d_pre[:, 0] == pre_gid).nonzero()[0]
            st_pre = self.d_pre[idx_pre, 1]
            print 'pre_gid spiketrain', pre_gid, st_pre
            for post_gid in post_gids:
                idx_post = (self.d_post[:, 0] == post_gid).nonzero()[0]
                st_post = self.d_post[idx_post, 1]
                print 'post_gid spiketrain', post_gid, st_post
                s_pre = BCPNN.convert_spiketrain_to_trace(st_pre, self.t_range[1])
                s_post = BCPNN.convert_spiketrain_to_trace(st_post, self.t_range[1])

                wij, bias, pi, pj, pij, ei, ej, eij, zi, zj = BCPNN.get_spiking_weight_and_bias(s_pre, s_post, self.params['params_synapse_d1_MT_BG'])
                bcpnn_traces.append([wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post])
                gid_pairs.append((pre_gid, post_gid))

        return bcpnn_traces, gid_pairs



    def plot_trace(self, bcpnn_traces, output_fn):
        wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, pre_trace, post_trace = bcpnn_traces
        t_axis = self.dt * np.arange(zi.size)
        plots = []
        pylab.rcParams.update({'figure.subplot.hspace': 0.30})
        fig = pylab.figure(figsize=FigureCreator.get_fig_size(1200, portrait=False))
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)
        self.title_fontsize = 18
        ax1.set_title('$\\tau_{z_i} = %d$ ms, $\\tau_{z_j} = %d$ ms' % \
                (self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j']), fontsize=self.title_fontsize)
        ax1.plot(t_axis, pre_trace, c='b', lw=2, ls=':')
        ax1.plot(t_axis, post_trace, c='g', lw=2, ls=':')
        p1, = ax1.plot(t_axis, zi, c='b', label='$z_i$', lw=2)
        p2, = ax1.plot(t_axis, zj, c='g', label='$z_j$', lw=2)
        plots += [p1, p2]
        labels_z = ['$z_i$', '$z_j$']
        ax1.legend(plots, labels_z)
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('z-traces')

        plots = []
        p1, = ax2.plot(t_axis, pi, c='b', lw=2)
        p2, = ax2.plot(t_axis, pj, c='g', lw=2)
        p3, = ax2.plot(t_axis, pij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_p = ['$p_i$', '$p_j$', '$p_{ij}$']
        ax2.set_title('$\\tau_{p} = %d$ ms' % \
                (self.bcpnn_params['tau_p']), fontsize=self.title_fontsize)
        ax2.legend(plots, labels_p)
        ax2.set_xlabel('Time [ms]')
        ax2.set_ylabel('p-traces')

        plots = []
        p1, = ax3.plot(t_axis, ei, c='b', lw=2)
        p2, = ax3.plot(t_axis, ej, c='g', lw=2)
        p3, = ax3.plot(t_axis, eij, c='r', lw=2)
        plots += [p1, p2, p3]
        labels_p = ['$e_i$', '$e_j$', '$e_{ij}$']
        ax3.set_title('$\\tau_{e} = %d$ ms' % \
                (self.bcpnn_params['tau_e']), fontsize=self.title_fontsize)
        ax3.legend(plots, labels_p)
        ax3.set_xlabel('Time [ms]')
        ax3.set_ylabel('e-traces')

        plots = []
        p1, = ax4.plot(t_axis, wij, c='b', lw=2)
        plots += [p1]
        labels_w = ['$w_{ij}$']
        ax4.legend(plots, labels_w)
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('Weight')

        plots = []
        p1, = ax6.plot(t_axis, bias, c='b', lw=2)
        plots += [p1]
        labels_ = ['bias']
        ax6.legend(plots, labels_)
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Bias')

#        ax5.set_yticks([])
#        ax5.set_xticks([])
#        ax5.annotate('$v_{stim} = %.2f, v_{0}=%.2f, v_{1}=%.2f$\ndx: %.2f\
#                \nWeight max: %.3e\nWeight end: %.3e\nWeight avg: %.3e\nt(w_max): %.1f [ms]' % \
#                (self.v_stim, self.tp_s[0][2], self.tp_s[1][2], self.dx, self.w_max, self.w_end, self.w_avg, \
#                self.t_max * dt), (.1, .1), fontsize=20)

#        ax5.set_xticks([])
#        output_fn = self.params['figures_folder'] + 'traces_tauzi_%04d_tauzj%04d_taue%d_taup%d_dx%.2e_dv%.2e_vstim%.1e.png' % \
#                (self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j'], self.bcpnn_params['tau_e'], self.bcpnn_params['tau_p'], self.dx, self.dv, self.v_stim)
#        output_fn = self.params['figures_folder'] + 'traces_dx%.2e_dv%.2e_vstim%.1e_tauzi_%04d_tauzj%04d_taue%d_taup%d.png' % \
#                (self.dx, self.dv, self.v_stim, self.bcpnn_params['tau_i'], self.bcpnn_params['tau_j'], self.bcpnn_params['tau_e'], self.bcpnn_params['tau_p'])
        print 'Saving traces to:', output_fn
        pylab.savefig(output_fn)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    fn_pre = params['spiketimes_folder'] + params['mpn_exc_spikes_fn_merged']
    fn_post = params['spiketimes_folder'] + params['d1_spikes_fn_merged_all']
    if (not os.path.exists(fn_pre)) or (not os.path.exists(fn_post)):
        utils.merge_spikes(params)
    
    it_range = (0, 10)

    TP = TracePlotter(params, it_range)
    TP.load_spikes(fn_pre, fn_post)
#    pre_gids, post_gids = TP.select_cells()
    pre_gids, post_gids = [813], [5555]

    all_traces, gid_pairs = TP.compute_traces(pre_gids, post_gids)
    output_fn_base = params['figures_folder'] + 'bcpnn_trace_'
    for i_, traces in enumerate(all_traces):
        output_fn = output_fn_base + '%d_%d.png' % (gid_pairs[i_][0], gid_pairs[i_][1])
        TP.plot_trace(traces, output_fn)

    pylab.show()
