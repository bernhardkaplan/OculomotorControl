import os
import numpy
import sys
import json

class MergeSpikefiles(object):

    def __init__(self, params):
        self.params = params


    def merge_nspike_files(self, merge_pattern, sorted_pattern_output,  sort_idx=1):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = rnd_nr1 + 1
        fn_out = sorted_pattern_output
        print 'MergeSpikeFiles nspike:', fn_out
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        os.system("cat %s_* > %s" % (merge_pattern,  tmp_file))
        # sort according to cell id
        os.system("sort -gk %d %s > %s" % (sort_idx, tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))


    def merge_spiketimes_files(self, merge_pattern, sorted_pattern_output,  sort_idx=1):
        rnd_nr1 = numpy.random.randint(0,10**8)
        rnd_nr2 = numpy.random.randint(0,10**8) + 1
        fn_out = sorted_pattern_output
        print 'MergeSpikeFiles spiketimes:', fn_out#, merge_pattern
        # merge files from different processors
        tmp_file = "tmp_%d" % (rnd_nr2)
        cmd = "cat %s* > %s" % (merge_pattern,  tmp_file)
        print 'debug', cmd
        os.system(cmd)
        # sort according to cell id
        os.system("sort -gk %d %s > %s" % (sort_idx, tmp_file, fn_out))
        os.system("rm %s" % (tmp_file))




if __name__ == '__main__':
#    info_txt = \
#    """
#    Usage:
#        python MergeSpikeFiles.py [FOLDER] [CELLTYPE] 
#        or
#        python MergeSpikeFiles.py [FOLDER] [CELLTYPE] [PATTERN_NUMBER]
#    """
#    assert (len(sys.argv) > 2), 'ERROR: folder and cell_type not given\n' + info_txt

#    fparam = 'Test/Parameters/simulation_parameters.json'
#    f = open(fparam, 'r')
#    params = json.load(f)

    try:
        folder = os.path.abspath(sys.argv[1])
        params_fn = os.path.abspath(folder) + '/Parameters/simulation_parameters.json'
        param_tool = simulation_parameters.global_parameters(params_fn=params_fn)
    except:
        param_tool = simulation_parameters.global_parameters()
    params = param_tool.params

    cell_types = ['d1', 'd2', 'action', 'efference', 'states']
    cell_types_volt = ['d1', 'd2', 'action']
    MS = MergeSpikefiles(params)
    for cell_type in cell_types:
        print 'Merging spiketimes file for %s ' % (cell_type)
        for naction in xrange(params['n_actions']):
            print 'Merging spiketimes file for %d ' % (naction)
            merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] + str(naction)
            output_fn = params['spiketimes_folder'] + params['%s_spikes_fn_merged' % cell_type] + str(naction) + '.dat'
            MS.merge_spiketimes_files(merge_pattern, output_fn)

#            output_fn = params['spiketimes_folder'] + str(naction) + params['%s_spikes_fn_merged' % cell_type] 
#    print 'Merging spiketimes file for states '
#    cell_type = 'states'
#    for nstate in xrange(params['n_states']):
#        print 'Merging spiketimes file for %d ' % (nstate)
#        merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] + str(nstate)
#        output_fn =  params['spiketimes_folder'] + str(nstate) + params['%s_spikes_fn_merged' % cell_type] 
#        MS.merge_spiketimes_files(merge_pattern, output_fn)

#    print 'Merging spiketimes file for rp'
#    cell_type = 'rp'
#    for i in xrange(params['n_states']*params['n_actions']):
#        print 'Merging spiketimes file for %d ' % (i)
#        merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] + str(i)
#        output_fn =  params['spiketimes_folder'] + str(i) + params['%s_spikes_fn_merged' % cell_type] 
#        MS.merge_spiketimes_files(merge_pattern, output_fn)

#    for cell_type in cell_types_volt:
#        print 'Merging voltmeter data file for %s ' % (cell_type)
#        print 'Merging voltmeter recordings file for %s ' % (cell_type)
#        for naction in xrange(params['n_actions']):
#            merge_pattern = params['spiketimes_folder'] + params['%s_volt_fn' % cell_type] + str(naction)
#            output_fn = params['spiketimes_folder'] + str(naction) + params['%s_volt_fn_merged' % cell_type]
#            MS.merge_spiketimes_files(merge_pattern, output_fn)
# need to add merging for rp and rew volt data
#    cell_type = 'rew'
#    print 'Merging spikes recordings file for %s ' % (cell_type)
#    merge_pattern = params['spiketimes_folder'] + params['%s_spikes_fn' % cell_type] 
#    output_fn = params['spiketimes_folder'] + params['%s_spikes_fn_merged' % cell_type]
#    MS.merge_spiketimes_files(merge_pattern, output_fn)
        
