
"""
This script needs to be run on a single core for a set of simulations individually.
It will create the folder structure and print the parameter file,
with which the simulation script NetworkSimModule is to be called.
"""
import os
import simulation_parameters

def clean_up_results_directory(params):
    filenames = [params['mpn_exc_spikes_fn'], \
                params['mpn_inh_spikes_fn'], \
                params['d1_spikes_fn'], \
                params['d1_volt_fn'], \
                params['d2_spikes_fn'], \
                params['d2_volt_fn'], \
                params['actions_spikes_fn'], \
                params['actions_volt_fn'], \
                params['efference_spikes_fn'], \
                params['supervisor_spikes_fn'], \
                params['rew_spikes_fn'], \
                params['rew_volt_fn'], \
                params['rp_spikes_fn'], \
                params['rp_volt_fn'], \
                params['states_spikes_fn_merged'], \
                params['d1_spikes_fn_merged'], \
                params['d2_spikes_fn_merged'], \
                params['actions_spikes_fn_merged'], \
                params['efference_spikes_fn_merged'], \
                params['supervisor_spikes_fn_merged'], \
                params['rew_spikes_fn_merged'], \
                params['rp_spikes_fn_merged'], \
                params['d1_volt_fn_merged'], \
                params['d2_volt_fn_merged'], \
                params['actions_volt_fn_merged'], \
                params['rew_volt_fn_merged'], \
                params['rp_volt_fn_merged'] ]

    for fn in filenames:
        cmd = 'rm %s*' % (fn)
        print 'Removing %s' % (cmd)
        os.system(cmd)


ps = simulation_parameters.global_parameters()
params = ps.params
clean_up_results_directory(params)
ps.set_filenames()
ps.create_folders()
ps.write_parameters_to_file()

print 'Ready for simulation:\n\t%s' % (params['params_fn_json'])
