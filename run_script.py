import os
import time

t0 = time.time()
#cmd = 'python PlottingScripts/PlotMPNActivity.py'
#cmd = 'python PlottingScripts/PlotBGActivity.py'

folder = 'Training_RBL_+10_titer25_nRF50_nV50_2_nStim1x3_taup600/'
cmd = 'python PlottingScripts/tune_bcpnn_trace_params.py'


for tau_zi in [5., 10., 20., 50., 100., 200., 400.]:
    for tau_zj in [5., 10., 20., 50., 100., 200., 400.]:
        for tau_p in [5., 50., 500., 5000., 50000., 250000.]:
            run_cmd = cmd + ' %s %d %d %d' % (folder, tau_zi, tau_zj, tau_p)
            print run_cmd

            os.system(run_cmd)

t1 = time.time()
print 'Time: %d [sec] %.1f [min]' % (t1 - t0, (t1 - t0)/60.)
#for thing in os.listdir('.'):

#    if thing.find('Test_afterRBL_2_it15__0-2') != -1:
#        run_cmd = cmd + ' ' + thing
#        print run_cmd
#        os.system(run_cmd)


