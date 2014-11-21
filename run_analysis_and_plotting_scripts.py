import sys
import numpy as np
import utils
from PlottingScripts.PlotBGActivity import run_plot_bg
from PlottingScripts.PlotMPNActivity import MetaAnalysisClass
from PlottingScripts.SuperPlot import PlotEverything
from PlottingScripts.PlotTesting import PlotTesting

if __name__ == '__main__':
    print 'Running analysis...'
    params = utils.load_params(sys.argv[1])
    if params['training']:
        P = PlotEverything(sys.argv, verbose=True)
    else:
        P = PlotTesting(sys.argv)
    run_plot_bg(sys.argv, None)
    MAC = MetaAnalysisClass([params['folder_name']])
    n_stim = params['n_stim']
    MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(n_stim)])

