import os

script_name = 'python PlottingScripts/PlotMPNActivity.py '
for thing in os.listdir('.'):

    if thing.find('Test_after') != -1:
        cmd = script_name + ' ' + thing
        print cmd
        os.system(cmd)



