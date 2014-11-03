import os, sys, inspect
sys.path.append('../')
sys.path.append('../PlottingScripts/')

import numpy as np

#import os, sys, inspect
# use this if you want to include modules from a subforder
#cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../")))
#if cmd_subfolder not in sys.path:
#    sys.path.insert(0, cmd_subfolder)

#cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../PlottingScripts/")))
#if cmd_subfolder not in sys.path:
#    sys.path.insert(0, cmd_subfolder)


#import test_functions as DT

#import debug_tool as DT

#DT.hello_man()
#H = DT.Hello()

#H = test_functions.Hello()


#"""
from plot_bcpnn_traces import TracePlotter
from test_functions import TestBcpnn
import BCPNN
import pylab

fn = sys.argv[1]
f = file(fn, 'r')

weight_trace = ''
time_info = []
record_weight = False
n = 0
for line in f:

    if record_weight and line.find('END PRINTOUT') == -1:
        weight_trace += line
        n += 1

    if line.find('START PRINTOUT') != -1:
        record_weight = True

    if line.find('END PRINTOUT') != -1:
        record_weight= False
            
fn_out = 'debug_traces.txt'
f_out = file(fn_out, 'w')
f_out.write(weight_trace)
f_out.flush()
f_out.close()


dt = 0.1
t_sim = n * dt

d = np.loadtxt(fn_out)
print 'n_steps from nest output:', d[:, 1].size
#spike_times = np.loadtxt('spikes-4-0.gdf')
#idx_1 = (spike_times[:, 0] == 1).nonzero()[0]
#spike_train_0 = spike_times[idx_1, :]
#idx_2 = (spike_times[:, 0] == 2).nonzero()[0]
#spike_train_1 = spike_times[idx_2, :]
#s_pre = BCPNN.convert_spiketrain_to_trace(spike_train_0, t_sim - 1)
#s_post = BCPNN.convert_spiketrain_to_trace(spike_train_1, t_sim - 1)
#print 'spike pre:', spike_train_0
#print 'spike post:', spike_train_1

#s_pre = np.zeros(n)
#s_post = np.zeros(n)

print 'For Online computation we use these spike trains:'


#bcpnn_traces = [wij, bias, pi, pj, pij, ei, ej, eij, zi, zj, s_pre, s_post]
s_post = d[:, 11]
s_pre = d[:, 12]
#print 'debug s_pre.size', len(s_pre)
bcpnn_traces = [d[:, 9], np.log(d[:, 7]), d[:, 6], d[:, 7], d[:, 8], d[:, 3], d[:, 4], d[:, 5], d[:, 1], d[:, 2], s_pre, s_post]

K_values = (0., 1., 0.)

T = TestBcpnn(K_values[0])
T.K_values = K_values
T.times_K_changed = [0., 0., 1000.]
T.t_sim = t_sim
K_vec = T.get_K_vec()[:-1]
#bcpnn_params = T.syn_params

TP = TracePlotter({})
#TP.plot_trace(bcpnn_traces, bcpnn_params, dt, output_fn=None, info_txt=None, fig=None, \
#        color_pre='b', color_post='g', color_joint='r', style_joint='-')

output_fn = 'online_traces.png'
TP.plot_trace_with_spikes(bcpnn_traces, T.syn_params, dt, output_fn=output_fn, fig=None, \
        color_pre='b', color_post='g', color_joint='r', style_joint='-', K_vec=K_vec, \
        extra_txt='K_vec ONLINE')

print 'output_fn', output_fn

pylab.show()
