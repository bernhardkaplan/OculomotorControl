import sys
import numpy as np
from PlottingScripts.PlotBGActivity import run_plot_bg
from PlottingScripts.PlotMPNActivity import MetaAnalysisClass
from PlottingScripts.SuperPlot import PlotEverything

if __name__ == '__main__':
    print 'Running analysis...'
    P = PlotEverything(sys.argv, verbose=True)
    run_plot_bg(params, None)
    MAC = MetaAnalysisClass([params['folder_name']])
    MAC = MetaAnalysisClass(['dummy', params['folder_name'], str(0), str(n_stim)])

