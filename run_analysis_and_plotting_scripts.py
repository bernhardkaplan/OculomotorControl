import os
import sys
import numpy as np
import utils
from PlottingScripts.PlotBGActivity import run_plot_bg
from PlottingScripts.PlotMPNActivity import MetaAnalysisClass
from PlottingScripts.SuperPlot import PlotEverything
from PlottingScripts.PlotTesting import PlotTesting

if __name__ == '__main__':
    print 'Running analysis...'

#    for folder in sys.argv[1:]:
#        params = utils.load_params(folder)
#        if params['training']:
#            P = PlotEverything(sys.argv, verbose=True)
#        else:
#            P = PlotTesting(sys.argv)
#        run_plot_bg(params, None)
#        MAC = MetaAnalysisClass([params['folder_name']])
#        n_stim = params['n_stim']
#        MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(n_stim)])


    params = utils.load_params(sys.argv[1])
    stim_range = (0, 100)
    for idx in xrange(stim_range[0], stim_range[1]):
        cmd = 'python PlottingScripts/PlotBGActivity.py %s %d %d' % (params['folder_name'], idx, idx + 1)
        print 'run:', cmd
        os.system(cmd)
        cmd = 'python PlottingScripts/PlotMPNActivity.py %s %d %d' % (params['folder_name'], idx, idx + 1)
        print 'run:', cmd
        os.system(cmd)
