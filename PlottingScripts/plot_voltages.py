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
import json
import pylab
import MergeSpikefiles
import simulation_parameters
import FigureCreator
pylab.rcParams.update(FigureCreator.plot_params_portrait)

class VoltPlotter(object):

    def __init__(self, params, it_range=None):
        self.params = params
        if it_range == None:
            self.it_range = (0, self.params['n_iterations_per_stim'])
        else:
            self.it_range = it_range
        self.t_range = (self.it_range[0] * self.params['t_iteration'], self.it_range[1] * self.params['t_iteration'])
        self.dt = params['dt']
        self.loaded_files = {}
        self.n_fig_x = 1
        self.n_fig_y = 1
        self.color_cnt = 0
        self.colorlist = utils.get_colorlist()
#        self.create_fig()

    def get_filename_base(self, gid):
        # get the population from the gid
        f1 = file(self.params['mpn_gids_fn'], 'r')
        mpn_gids = json.load(f1)
        if gid in mpn_gids['exc']:
            pop =  'mpn_exc'
            fn_base = self.params['mpn_exc_volt_fn']
            return fn_base
#        elif gid in mpn_gids['inh']:
#            pop =  'mpn_inh'
#            fn_base = self.params['mpn_inh_volt_fn']
#            return fn_base
        else:
            f2 = file(self.params['bg_gids_fn'], 'r')
            bg_gids = json.load(f2)
            for i_, subpop in enumerate(bg_gids['d1']):
                if gid in subpop:
                    fn_base = self.params['d1_volt_fn'] + '%d-' % (i_)
                    return fn_base
            for i_, subpop in enumerate(bg_gids['d2']):
                if gid in subpop:
                    fn_base = self.params['d2_volt_fn'] + '%d-' % (i_)
                    return fn_base
            for i_, subpop in enumerate(bg_gids['action']):
                if gid in subpop:
                    fn_base = self.params['actions_volt_fn'] + '%d-' % (i_)
                    return fn_base


    def load_volt_file(self, fn):
        if fn in self.loaded_files.keys():
            loaded = True
        else:
            loaded = False
        if not loaded:
            path = self.params['spiketimes_folder'] + fn
            print 'plot_voltages loads', path
            self.loaded_files[fn] = np.loadtxt(path)
        return self.loaded_files[fn]


    def get_trace(self, gid, fn_base=None):
        if fn_base == None:
            fn_base = self.get_filename_base(gid)
        volt_fns = utils.find_files(self.params['spiketimes_folder'], fn_base)
        for fn in volt_fns:
            d = self.load_volt_file(fn)
            time_axis, volt = utils.extract_trace(d, gid)
            if volt.size > 0:
                return (time_axis, volt)
        return [], []


    def get_recorded_bg_gids_for_action(self, action, cell_type='d1'):
        f = file(self.params['bg_gids_fn'], 'r')
        bg_gids = json.load(f)
        fn_base = self.params['%s_volt_fn' % cell_type] + '%d-' % (action)
        volt_fns = utils.find_files(self.params['spiketimes_folder'], fn_base)
        for fn in volt_fns:
            d = self.load_volt_file(fn)
            gids = np.unique(d[:, 0])
            return gids


    def create_fig(self):
        self.fig = pylab.figure()


    def plot_trace(self, trace, ax=None, fig_cnt=1, legend_txt=None, color=None):
        if ax == None:
            ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)

        t_axis = np.arange(trace.size) * self.params['dt_volt']

        if color == None:
            color = self.colorlist[self.color_cnt]
        p, = ax.plot(t_axis, trace, label=legend_txt, c=color)
        self.color_cnt += 1

        if fig_cnt == self.n_fig_x * self.n_fig_y:
            ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Voltage [mV]')
        ax.set_xlim((self.t_range[0], self.t_range[1]))

        ax.legend()

    def add_subplot(self, fig_cnt):
        ax = self.fig.add_subplot(self.n_fig_y, self.n_fig_x, fig_cnt)
        self.color_cnt = 0
        return ax
    

def plot_voltage_gids_all_actions(params, gid_list_of_lists, fn_filter='bg_volt'):
    """
    Load all possible voltage files but plot only those traces given in gids
    """

    P = VoltPlotter(params)
    P.create_fig()
    all_traces = []
    all_gids = []

    ax = P.add_subplot(1)

    for i_action, gids_for_action in enumerate(gid_list_of_lists):
        print 'debug gids_for action', i_action, gids_for_action 
        for gid in gids_for_action:
            t_axis, trace = P.get_trace(gid, fn_filter)
            P.plot_trace(trace, color=P.colorlist[i_action])
#            print 'MPN gid', gid, len(trace), len(t_axis)
            all_traces.append(trace)
            all_gids.append(gid)

#    for i_, trace in enumerate(all_traces):
#        P.plot_trace(trace, ax, legend_txt='%d' % all_gids[i_])

def plot_voltage_gids_for_action(params, gid_list, fn_filter='bg_volt'):
    """
    Load all possible voltage files but plot only those traces given in gids
    """

    P = VoltPlotter(params)
    P.create_fig()
    all_traces = []
    all_gids = []

    ax = P.add_subplot(1)

    for gid in gid_list:
        print 'debug', gid, gid_list
        t_axis, trace = P.get_trace(gid, fn_filter)
        if len(trace) > 0:
            P.plot_trace(trace, legend_txt='%d' % gid)
            all_traces.append(trace)
            all_gids.append(gid)

#    for i_, trace in enumerate(all_traces):
#        P.plot_trace(trace, ax, legend_txt='%d' % all_gids[i_])



if __name__ == '__main__':

    if len(sys.argv) > 1:
        params = utils.load_params(sys.argv[1])
    else:
        param_tool = simulation_parameters.global_parameters()
        params = param_tool.params

    f = file(params['bg_gids_fn'], 'r')
    bg_gids = json.load(f)
    action_gids = bg_gids['action']
    print 'debug', action_gids
    plot_voltage_gids_all_actions(params, action_gids)


    action_gids = bg_gids['d2'][1]
    plot_voltage_gids_for_action(params, action_gids)

#    it_range = (0, 3)
#    P = VoltPlotter(params, it_range)
#    P.n_fig_x = 1
#    P.n_fig_y = 2
#    P.create_fig()
#    bg_gids = params['gids_to_record_bg']
#    mpn_gids = params['gids_to_record_mpn']
#    ax = P.add_subplot(1)
#    all_traces = []
#    all_gids = []
#    for gid in mpn_gids:
#        t_axis, trace = P.get_trace(gid)
#        print 'MPN gid', gid, len(trace), len(t_axis)
#        all_traces.append(trace)
#        all_gids.append(gid)
#    for i_, trace in enumerate(all_traces):
#        P.plot_trace(trace, ax, legend_txt='%d' % all_gids[i_])
#    ax = P.add_subplot(2)
#    all_traces = []
#    all_gids = []
#    for gid in bg_gids:
#        t_axis, trace = P.get_trace(gid, fn_base=params['bg_volt_fn'])
#        print 'BG gid', gid, len(trace), len(t_axis)
#        all_traces.append(trace)
#        all_gids.append(gid)
#    for i_, trace in enumerate(all_traces):
#        P.plot_trace(trace, ax, legend_txt='%d' % all_gids[i_])



        
#    action_idx = 20
#    all_traces = []
#    all_gids = []
#    bg_gids = P.get_recorded_bg_gids_for_action(action_idx)
#    for gid in bg_gids:
#        t_axis, trace = P.get_trace(gid)
#        all_traces.append(trace)
#        all_gids.append(gid)
#        print 'BG gid', gid, len(trace), len(t_axis)
#    for i_, trace in enumerate(all_traces):
#        P.plot_trace(trace, fig_cnt=2, legend_txt='%d' % all_gids[i_])

    
#    output_fn_base = params['figures_folder'] + 'volt_trace_'
    pylab.show()
