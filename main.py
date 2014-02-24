import sys
import os
import nest
import BasalGanglia
import Reward
import json
import simulation_parameters
import utils
import numpy as np
import time
import tempfile

os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()
import matplotlib
matplotlib.use('Agg')
import pylab as pl

# Load BCPNN synapse and bias-iaf neuron module                                                                                                
if (not 'bcpnn_synapse' in nest.Models()):
    nest.sr('(/cfs/klemming/nobackup/b/berthet/code/modules/bcpnn_module/share/nest/sli) addpath') #t/tully/sequences/share/nest/sli
    nest.Install('/cfs/klemming/nobackup/b/berthet/code/modules/bcpnn_module/lib/nest/pt_module') #t/tully/sequences/lib/nest/pt_module


try: 
    from mpi4py import MPI
    USE_MPI = True
    comm = MPI.COMM_WORLD
    pc_id, n_proc = comm.rank, comm.size
    print "USE_MPI:", USE_MPI, 'pc_id, n_proc:', pc_id, n_proc
except:
    USE_MPI = False
    pc_id, n_proc, comm = 0, 1, None
    print "MPI not used"



nest.ResetKernel()
# load bcpnn synapse module and iaf neuron with bias
#if (not 'bcpnn_synapse' in nest.Models('synapses')):
#    nest.Install('pt_module')
nest.SetKernelStatus({"overwrite_files": True})



def save_spike_trains(params, iteration, stim_list, gid_list):
    n_units = len(stim_list)
    fn_base = params['input_st_fn_mpn']
    for i_ in xrange(n_units):
        if len(stim_list[i_]) > 0:
            fn = fn_base + '%d_%d.dat' % (iteration, gid_list[i_] - 1)
            np.savetxt(fn, stim_list[i_])


def remove_files_from_folder(folder):
    print 'Removing all files from folder:', folder
    path =  os.path.abspath(folder)
    cmd = 'rm  %s/*' % path
    print cmd
    os.system(cmd)


if __name__ == '__main__':

    t1 = time.time()
    if len(sys.argv) > 1: # re-run an old parameter file
        param_fn = sys.argv[1]
        if os.path.isdir(param_fn): # go to the path containing the json object storing old parameters
            param_fn += '/Parameters/simulation_parameters.json' # hard coded subpath in ParameterContainer
        assert os.path.exists(param_fn), 'ERROR: Can not find %s - please give an existing parameter filename or folder name to re-run a simulation' % (param_fn)
        f = file(param_fn, 'r')
        print 'Loading parameters from', param_fn
        params = json.load(f)
    else: # run a simulation with parameters as set in simulation_parameters
        GP = simulation_parameters.global_parameters()
        GP.write_parameters_to_file() # write_parameters_to_file MUST be called before every simulation
        params = GP.params

    t0 = time.time()

    #weights_sim = {}
    staten2D1 = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
    staten2D2 = [[0. for i_action in range(params['n_actions'])] for j in range(params['n_iterations'])]
	
    action2D1 = [[0. for i_state in range(params['n_states'])] for j in range(params['n_iterations'])]
    action2D2 = [[0. for i_state in range(params['n_states'])] for j in range(params['n_iterations'])]
#    if pc_id == 0:
#        remove_files_from_folder(params['spiketimes_folder_mpn'])
#        remove_files_from_folder(params['input_folder_mpn'])
    
    BG = BasalGanglia.BasalGanglia(params, comm)
    R = Reward.Reward(params)

    actions = np.empty(params['n_iterations']) 
    states  = np.empty(params['n_iterations']) 
    rewards = np.empty(params['n_iterations']) 
#    network_states_net= np.zeros((params['n_iterations'], 4))
    block = 0
    for iteration in xrange(params['n_iterations']):
        
        print 'ITERATION', iteration

        state = iteration % params['n_states']
       
        BG.set_state(state)
        BG.set_gain(1.)
        BG.set_kappa_OFF()
        if comm != None:
            comm.barrier()
        nest.Simulate(params['t_selection'])
        if comm != None:
            comm.barrier()

		# record weights between state 0 and all actions for both d1 and d2 pathways
        for nactions in range(params['n_actions']):
            # conn = nest.FindConnections(source = BG.states[0], target = BG.strD1[nactions][0], synapse_model = 'bcpnn_synapse')
            conn = nest.GetConnections(source = BG.states[0], target = BG.strD1[nactions], synapse_model = 'bcpnn_synapse')
            staten2D1[iteration][nactions] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]) 
            conn = nest.GetConnections(source = BG.states[0], target = BG.strD2[nactions], synapse_model = 'bcpnn_synapse')
            # conn = nest.FindConnections(source = BG.states[0], target = BG.strD2[nactions][0], synapse_model = 'bcpnn_synapse')
            staten2D2[iteration][nactions] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            print 'debug: WEIGHTS CHECK staten2D2:', staten2D2[iteration][nactions] 
            for a in nest.GetStatus(conn):
                print 'source:', a['source'] ,'target:', a['target'], 'TRACESCHECK PI ', a['p_i']
                print 'source:', a['source'] ,'target:', a['target'], 'TRACESCHECK PJ ', a['p_j']
                print 'source:', a['source'] ,'target:', a['target'], 'TRACESCHECK PIJ', a['p_ij']
                print 'DATA pi pj pij', a['p_i'], a['p_j'], a['p_ij'], 'RATIO', a['p_ij']/(a['p_i']*a['p_j']) 
                print 'LOG', np.log( a['p_ij']/(a['p_i']*a['p_j']) ) 


		# record weights between all the states and action 0 for both d1 and d2 pathways
        for nstates in range(params['n_states']):
            conn = nest.GetConnections(source = BG.states[nstates], target = BG.strD1[0], synapse_model = 'bcpnn_synapse')
            # conn = nest.FindConnections(source = BG.states[nstates], target = BG.strD1[0][0], synapse_model = 'bcpnn_synapse')
            action2D1[iteration][nstates] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]) 
            conn = nest.GetConnections(source = BG.states[nstates], target = BG.strD2[0], synapse_model = 'bcpnn_synapse')
            # conn = nest.FindConnections(source = BG.states[nstates], target = BG.strD2[0][0], synapse_model = 'bcpnn_synapse')
            action2D2[iteration][nstates] = np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)])
            print 'debug: WEIGHTS CHECK action2D2 :', np.mean([(np.log(a['p_ij']/(a['p_i']*a['p_j']))) for a in nest.GetStatus(conn)]) 

       # weights_sim[iteration] = BG.get_weights(BG.states[0], BG.strD1[:])
        states[iteration] = state
        actions[iteration] = BG.get_action() # BG returns the selected action
        
        #BG.stop_state()
        #BG.set_gain(1.)
        BG.set_efference_copy(actions[iteration])
        if comm != None:
            comm.barrier()
        nest.Simulate(params['t_efference'])
        if comm != None:
            comm.barrier()

        rew = utils.communicate_reward(comm, R , state, actions[iteration], block )
        
        rewards[iteration] = rew

        print 'REWARD =', rew
        BG.set_gain(0.)
        if rew == 1:
            BG.set_kappa_ON(0.01, states[iteration], actions[iteration])
        else:
            BG.set_kappa_ON(-0.01, states[iteration], actions[iteration])
        BG.set_reward(rew)
        #BG.stop_efference()

        if comm != None:
            comm.barrier()
        nest.Simulate(params['t_reward'])
        if comm != None:
            comm.barrier()

        BG.set_gain(0.)
        BG.set_kappa_OFF()
        BG.set_rest()
        if comm != None:
            comm.barrier()
        
        nest.Simulate(params['t_rest'])
        if comm != None:
            comm.barrier()   
        block = int (iteration / params['block_len'])
    

    # END of SIMULATION LOOP
    if pc_id == 0:
        np.savetxt(params['actions_taken_fn'], actions)
        np.savetxt(params['states_fn'], states)
        np.savetxt(params['rewards_fn'], rewards)
        
        exc_sptimes = nest.GetStatus(BG.recorder_d1[0])[0]['events']['times']
        for i_proc in xrange(1,n_proc ):
            exc_sptimes = np.r_[exc_sptimes, comm.recv(source=i_proc)]
       
    else:
        comm.send(nest.GetStatus(BG.recorder_d1[0])[0]['events']['times'],dest=0)
    if comm != None:
        comm.barrier()
    if pc_id == 0:
        exc_spids = nest.GetStatus(BG.recorder_d1[0])[0]['events']['senders']
        for i_proc in xrange(1, n_proc):
            exc_spids = np.r_[exc_spids, comm.recv(source=i_proc)]
        pl.figure(33)
        pl.scatter(exc_sptimes, exc_spids,s=1.)
        binsize = 10
        bins=np.arange(0, params['t_sim']+1, binsize)
        c_exc,b = np.histogram(exc_sptimes,bins=bins)
        rate_exc = c_exc*(1000./binsize)*(1./params['num_msn_d1'])
        pl.plot(b[0:-1],rate_exc)
        pl.title('firing rate of STR D1 action 0')
        pl.savefig('fig3_firingrate.pdf')
    else:
        comm.send(nest.GetStatus(BG.recorder_d1[0])[0]['events']['senders'],dest=0)
    if comm != None:
        comm.barrier()

    #CC.get_weights(, BG)  implement in BG or utils


#    print 'weight simu ', weights_sim
    if pc_id ==1:	
        t1 = time.time() - t0
        print 'Time: %.2f [sec] %.2f [min]' % (t1, t1 / 60.)

        print 'learning completed'
        pl.figure(1)
        pl.subplot(211)
        pl.title('D1')
        pl.plot(staten2D1)
        pl.ylabel(r'$w_{0j}$')
        pl.subplot(212)
        pl.title('D2')
        pl.plot(staten2D2)
        pl.ylabel(r'$w_{0j}$')
        pl.xlabel('trials')
        pl.suptitle('Computed weights from state 0 to all actions')
        pl.savefig('fig1_allactions.pdf')

        pl.figure(2)
        pl.subplot(211)
        pl.plot(action2D1)
        pl.title('D1')
        pl.ylabel(r'$w_{0j}$')
        pl.subplot(212)
        pl.plot(action2D2)
        pl.ylabel(r'$w_{0j}$')
        pl.title('D2')
        pl.xlabel('trials')
        pl.suptitle('Computed weights from all states to action 0')
        pl.savefig('fig2_allstates.pdf')
#		pl.show()	
#		pl.show()
        print 'States ', states
        print 'Actions ', actions
        print 'Rewards ', rewards
