import numpy as np
import json
import utils
import random

class VisualInput(object):

    def __init__(self, params, comm=None, visual_stim_seed=None):
        """
        Keyword arguments
        params -- dictionary that contains 
        """
        self.params = params
        self.trajectories = []
        self.t_axis = np.arange(0, self.params['t_iteration'], self.params['dt'])
        self.iteration = 0
        self.t_current = 0 # stores the 'current' time
        if visual_stim_seed == None:
            visual_stim_seed = self.params['visual_stim_seed']
        np.random.seed(visual_stim_seed)
        random.seed(visual_stim_seed)
        self.RNG = np.random

        self.supervisor_state = [0., 0.]
        self.tuning_prop_exc = self.set_tuning_prop('exc')
        if self.params['with_inh_mpn']:
            self.tuning_prop_inh = self.set_tuning_prop('inh')

#        self.rf_sizes = self.set_receptive_fields('exc')

        self.comm = comm
        if self.comm != None:
            self.pc_id = comm.rank
            self.n_proc = comm.size
        else:
            self.pc_id = 0
            self.n_proc = 1

        if self.pc_id == 0:
            print 'Saving tuning properties exc to:', self.params['tuning_prop_exc_fn']
            np.savetxt(self.params['tuning_prop_exc_fn'], self.tuning_prop_exc)
            np.savetxt(self.params['receptive_fields_exc_fn'], self.rf_sizes)
            if self.params['with_inh_mpn']:
                print 'Saving tuning properties inh to:', self.params['tuning_prop_inh_fn']
                np.savetxt(self.params['tuning_prop_inh_fn'], self.tuning_prop_inh)
        if self.comm != None:
            self.comm.Barrier()


        self.x0_stim = np.zeros(self.params['n_iterations'])
        self.perceived_states = np.zeros((self.params['n_iterations'], 4))

        self.current_motion_params = list(self.params['initial_state'])
        # store the motion parameters seen on the retina
        self.n_stim_dim = len(self.params['initial_state'])
        self.motion_params = np.zeros((self.params['n_iterations'], self.n_stim_dim + 1))  # + 1 dimension for the time axis

#        self.get_gids_near_stim_trajectory(verbose=self.params['debug_mpn'])


    def create_training_sequence_iteratively(self):
        """
        Training samples are drawn from the tuning properties of the cells, i.e. follow the same distribution
        Returns n_cycles of state vectors, each cycle containing a set of n_training_stim_per_cycle states.
        The set of states is shuffled for each cycle
        """
        mp_training = np.zeros((self.params['n_stim_training'], 4))
        training_states = np.zeros((self.params['n_training_stim_per_cycle'], 4))
        if self.params['n_training_stim_per_cycle'] == self.params['n_exc_mpn']:
            training_states_int = range(0, self.params['n_exc_mpn'])
        else:
#            for i_ in xrange(self.params['n_training_stim_per_cycle']):
#                rnd_ = self.RNG.random_integers(0, self.params['n_exc_mpn'] - 1, 1)
#                print 'RND:', rnd_
#                training_states[i_] = self.tuning_prop_exc[rnd_, :]
            training_states_int = self.RNG.random_integers(0, self.params['n_exc_mpn'] - 1, self.params['n_training_stim_per_cycle'])
#        print 'training_states_nit', training_states_int
        training_states = self.tuning_prop_exc[training_states_int, :]

        if self.params['n_stim_training'] == 1:
            x0 = self.params['initial_state'][0]
            v0 = self.params['initial_state'][2]
            mp_training[0, 0] = x0
            mp_training[0, 1] = .5
            mp_training[0, 2] = v0
        else:
            for i_cycle in xrange(self.params['n_training_cycles']):
                self.RNG.shuffle(training_states)
                i_ = i_cycle * self.params['n_training_stim_per_cycle']
                for i_stim in xrange(self.params['n_training_stim_per_cycle']):
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 0] = (training_states[i_stim][0] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_x'])) % 1.
                    mp_training[i_stim + i_, 1] = .5
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 2] =  training_states[i_stim][2] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_v'])
                    mp_training[i_stim + i_, 3] =  0.

#                i_ = i_cycle * self.params['n_training_stim_per_cycle']
#                j_ = (i_cycle + 1) * self.params['n_training_stim_per_cycle']
#                mp_training[i_:j_, :] = self.tuning_prop_exc[training_states, :]

        np.savetxt(self.params['training_sequence_fn'], mp_training)
        return mp_training 


    def create_training_sequence_from_a_grid(self):
        """
        Training samples are generated in a grid-like manner, i.e. random points from a grid on the tuning property space
        are drawn
         
        Returns n_cycles of state vectors, each cycle containing a set of n_training_stim_per_cycle states.
        The set of states is shuffled for each cycle.
        """
        mp_training = np.zeros((self.params['n_stim_training'], 4))

        x_lim_frac = .9
        v_lim_frac = .8
#        x_lim_frac = 1.
#        v_lim_frac = 1.
#        x_lim = ((1. - x_lim_frac) * (self.params['x_max_tp'] - self.params['x_min_tp']), x_lim_frac * self.params['x_max_tp'])
#        v_lim = (- v_lim_frac * (self.params['v_max_tp'] - self.params['v_min_tp']), v_lim_frac * self.params['v_max_tp'])
        x_lim = ((1. - x_lim_frac) * (np.max(self.tuning_prop_exc[:, 0]) - np.min(self.tuning_prop_exc[:, 0])), x_lim_frac * np.max(self.tuning_prop_exc[:, 0]))
        v_lim = (v_lim_frac * np.min(self.tuning_prop_exc[:, 2]), v_lim_frac * np.max(self.tuning_prop_exc[:, 2]))

        x_grid = np.linspace(x_lim[0], x_lim[1], self.params['n_training_x'])
        v_grid = np.linspace(v_lim[0], v_lim[1], self.params['n_training_v'])
        training_states_x = range(0, self.params['n_training_x'])
        training_states_v = range(0, self.params['n_training_v'])
        training_states = []
        for i_, x in enumerate(training_states_x):
            for j_, v in enumerate(training_states_v):
                training_states.append((x_grid[i_], v_grid[j_]))

        if self.params['n_stim_training'] == 1:
            x0 = self.params['initial_state'][0]
            v0 = self.params['initial_state'][2]
            mp_training[0, 0] = x0
            mp_training[0, 1] = .5
            mp_training[0, 2] = v0
        else:
            for i_cycle in xrange(self.params['n_training_cycles']):
                self.RNG.shuffle(training_states)
#                print 'Cycle %d training_states: ' % (i_cycle), training_states
                i_ = i_cycle * self.params['n_training_stim_per_cycle']
                for i_stim in xrange(self.params['n_training_stim_per_cycle']):
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 0] = (training_states[i_stim][0] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_x'])) % 1.
                    mp_training[i_stim + i_, 1] = .5
                    plus_minus = utils.get_plus_minus(self.RNG)
                    mp_training[i_stim + i_, 2] =  training_states[i_stim][1] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_v'])
                    mp_training[i_stim + i_, 3] =  training_states[i_stim][1] + plus_minus * self.RNG.uniform(0, self.params['training_stim_noise_v'])
        print 'VisualInput saves training sequence parameters to:', self.params['training_sequence_fn']
        np.savetxt(self.params['training_sequence_fn'], mp_training)
        return mp_training 


    def create_training_sequence_around_center(self):

        n_center = int(np.round(self.params['n_stim_training'] * self.params['frac_training_samples_center']))

        mp_center = np.zeros((n_center, 4))
        mp_center[:, 0] = self.RNG.normal(.5, self.params['center_stim_width'], n_center)
        mp_center[:, 2] = self.RNG.uniform(-self.params['v_max_tp'], self.params['v_max_tp'], n_center)
        return mp_center
        

    def create_training_sequence(self):
        """
        When training with multiple stimuli, this function returns the initial motion parameters for 
        the new stimulus
        """
        mp_training = np.zeros((self.params['n_stim_training'], 4))
        stim_params = np.zeros((self.params['n_training_stim_per_cycle'], 4))

        if self.params['n_stim_training'] == 1:
            x0 = self.params['initial_state'][0]
            v0 = self.params['initial_state'][2]
            mp_training[0, 0] = x0
            mp_training[0, 1] = .5
            mp_training[0, 2] = v0
        else:
            i_stim = 0
            for i_stim_in_cycle in xrange(self.params['n_training_stim_per_cycle']):
                plus_minus = utils.get_plus_minus(self.RNG)
                x0 = .5 + plus_minus * .5 * np.random.rand()
#                plus_minus = (-1.)**i_stim
                v0 = self.params['v_max_tp']
                while v0 > .8 * self.params['v_max_tp']: # choose a random cell index and add some noise to the v value 
                    rnd_idx = np.random.randint(0, self.params['n_exc_mpn'])
                    v0 = self.tuning_prop_exc[rnd_idx, 2]
                    v0 *= (.4 * np.random.rand() + .8)
                plus_minus = utils.get_plus_minus(self.RNG)
                v0 *= plus_minus
                stim_params[i_stim_in_cycle, 0] = x0
                stim_params[i_stim_in_cycle, 1] = .5
                stim_params[i_stim_in_cycle, 2] = v0

            # randomize the order for each cycle
            for i_cycle in xrange(self.params['n_training_cycles']):
                offset = i_cycle * self.params['n_training_stim_per_cycle']
                idx_ = xrange(self.params['n_training_stim_per_cycle'])
                idx_rnd = np.random.permutation(xrange(self.params['n_training_stim_per_cycle'])) + offset
                mp_training[idx_rnd, 0] = stim_params[idx_, 0]
                mp_training[idx_rnd, 1] = stim_params[idx_, 1]
                mp_training[idx_rnd, 2] = stim_params[idx_, 2]
        np.savetxt(self.params['training_sequence_fn'], mp_training)
        return mp_training 


    def compute_input(self, local_gids, action_code):
        """
        Integrate the real world trajectory and the eye direction and compute spike trains from that.

        Arguments:
        local_gids -- the GIDS for which the stimulus needs to be computed
        action_code -- a tuple representing the action (direction of eye movement)
        network_state --  perceived motion parameters, as given by the MPN network [x, y, u, v]
        """

#        self.trajectory, supervisor_state = self.update_stimulus_trajectory_new(action_code)
        self.trajectory, supervisor_state = self.update_stimulus_trajectory_static(action_code)
        self.x0_stim[self.iteration] = self.trajectory[0][0]
        local_gids = np.array(local_gids) - 1 # because PyNEST uses 1-aligned GIDS 
        self.create_spike_trains_for_trajectory(local_gids, self.trajectory)
        # update the position of the stimulus regardless of the action
        self.current_motion_params[0] += (self.current_motion_params[2] + action_code[0]) * self.params['t_iteration'] / self.params['t_cross_visual_field'] 
        self.current_motion_params[1] += (self.current_motion_params[3] + action_code[1]) * self.params['t_iteration'] / self.params['t_cross_visual_field'] 
        self.iteration += 1
        return self.stim, supervisor_state


    def get_reward_from_perceived_stim(self, perceived_state):
        """
        Computes the reward based on the internal states of the MPN (motion-perception / prediction network).
        Must be called after a simulation step.
        Also, compute_input increase self.iteration to + 1 (hence an addition -1 is used here)
        perceived_state -- is a 4-element list of the vector-average resembling [x, y, u, v]
        """
        self.perceived_states[self.iteration-1] = perceived_state
        punish_overshoot = .7
        learning_rate = 10.
        if self.iteration < 2:
            return 0
        else:
            x, y, v, u = perceived_state
            dx_i = self.perceived_states[self.iteration - 2][0] - .5 # -2 and -1 because self.iteration is + 1 (because compute_input has been called before)
            dx_j = self.perceived_states[self.iteration - 1][0] - .5
            dx_i_abs = np.abs(dx_i)
            dx_j_abs = np.abs(dx_j)
            diff_dx_abs = dx_j_abs - dx_i_abs # if diff_dx_abs < 0: # improvement
            R = -1 * learning_rate * diff_dx_abs
            if np.sign(dx_i) != np.sign(dx_j): # 'overshoot'
                R *= punish_overshoot
        return R


    def get_reward_from_real_stim_pos(self):
        """
        Computes the reward based on the REAL WORLD coordinates of the stimulus.
        Should be called after compute_input.
        Hence self.iteration in this function is always + 1.
        """
        punish_overshoot = .7
        learning_rate = 30.
        if self.iteration < 2:
            return 0
        else:
            dx_i = self.x0_stim[self.iteration - 2] - .5 # -2 and -1 because self.iteration is + 1 (because compute_input has been called before)
            dx_j = self.x0_stim[self.iteration - 1] - .5
            dx_i_abs = np.abs(dx_i)
            dx_j_abs = np.abs(dx_j)
            diff_dx_abs = dx_j_abs - dx_i_abs # if diff_dx_abs < 0: # improvement
            R = -1 * learning_rate * diff_dx_abs
            if np.sign(dx_i) != np.sign(dx_j): # 'overshoot'
                R *= punish_overshoot
        return R


    def get_supervisor_actions(self, training_stimuli, BG):
        """
        Computes the supervisor action (and index) as in compute_input_open_loop
        """
        time_axis = np.arange(0, self.params['t_iteration'], self.params['dt_input_mpn'])
        supervisor_states = np.zeros((self.params['n_stim'], 2))
        action_indices = np.zeros(self.params['n_stim'], dtype=np.int)
        motion_params_pre = np.zeros((self.params['n_stim'], 4))

        for i_stim in xrange(self.params['n_stim']):
            x_stim = (training_stimuli[i_stim, 2]) * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * training_stimuli[i_stim, 0]
            y_stim = (training_stimuli[i_stim, 3]) * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * training_stimuli[i_stim, 1]
            trajectory = (x_stim, y_stim)
            delta_x = (x_stim[-1] - .5)
            delta_y = (y_stim[-1] - .5)
#            delta_x = (x_stim[0] - .5)
#            delta_y = (y_stim[0] - .5)
#            delta_x = (x_stim[len(x_stim) / 2] - .5)
#            delta_y = (y_stim[len(y_stim) / 2] - .5)

            delta_t = (self.params['t_iteration'] / self.params['t_cross_visual_field'])
            k = self.params['supervisor_amp_param']
            supervisor_states[i_stim, 0] = k * delta_x / delta_t + training_stimuli[i_stim, 2]
            supervisor_states[i_stim, 1] = k * delta_y / delta_t + training_stimuli[i_stim, 3]
            motion_params_pre[i_stim, 0] = x_stim[0]
            motion_params_pre[i_stim, 1] = .5#x_stim[1]
            motion_params_pre[i_stim, 2] = training_stimuli[i_stim, 2]
            motion_params_pre[i_stim, 3] = training_stimuli[i_stim, 3]
            action_indices[i_stim] = BG.map_speed_to_action(supervisor_states[i_stim, 0], xy='x')
        return [supervisor_states, action_indices, motion_params_pre]


#    def get_supervisor_actions_new(self, training_stimuli, BG):
#        """
#        Computes the supervisor action (and index) as in compute_input_open_loop
#        """
#        time_axis = np.arange(0, self.params['t_iteration'], self.params['dt_input_mpn'])
#        supervisor_states = np.zeros((self.params['n_stim'], 2))
#        action_indices = np.zeros(self.params['n_stim'], dtype=np.int)
#        motion_params_pre = np.zeros((self.params['n_stim'], 4))
#        n_steps = self.params['t_iteration'] / self.params['dt_input_mpn']

#        for i_stim in xrange(self.params['n_stim']):

#            x_stim = self.current_motion_params[0] * self.params['t_iteration'] / self.params['t_cross_visual_field'] * np.ones(n_steps) \
#                    + time_axis * self.current_motion_params[2] / self.params['t_cross_visual_field']
#            y_stim = self.current_motion_params[1] * self.params['t_iteration'] / self.params['t_cross_visual_field'] * np.ones(n_steps) \
#                    + time_axis * self.current_motion_params[3] / self.params['t_cross_visual_field']
#            trajectory = (x_stim, y_stim)
#            delta_x_end = (x_stim[-1] - .5)
#            delta_y_end = (y_stim[-1] - .5)

#            delta_t = (self.params['t_iteration'] / self.params['t_cross_visual_field'])
#            k = self.params['supervisor_amp_param']
#            supervisor_states[i_stim, 0] = k * delta_x_end / delta_t + training_stimuli[i_stim, 2]
#            supervisor_states[i_stim, 1] = k * delta_y_end / delta_t + training_stimuli[i_stim, 3]
#            motion_params_pre[i_stim, 0] = x_stim[0]
#            motion_params_pre[i_stim, 1] = .5#x_stim[1]
#            motion_params_pre[i_stim, 2] = training_stimuli[i_stim, 2]
#            motion_params_pre[i_stim, 3] = training_stimuli[i_stim, 3]
#            action_indices[i_stim] = BG.map_speed_to_action(supervisor_states[i_stim, 0], xy='x')
#        return [supervisor_states, action_indices, motion_params_pre]




    def compute_input_open_loop(self, local_gids):
        """
        In contrast to the compute_input function, the open_loop variation does not update the stimulus trajectory
        based on the action that has been taken.
        """
        time_axis = np.arange(0, self.params['t_iteration'], self.params['dt_input_mpn'])

        # calculate where the stimulus will move according to the current_motion_params
        x_stim = (self.current_motion_params[2]) * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[0]
        y_stim = (self.current_motion_params[3]) * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[1]
        trajectory = (x_stim, y_stim)

        # compute the supervisor signal taking into account:
        # - the trajectory position at the end of the iteration
        # - the knowledge about the motion (current_motion_params
        # - and / or some sort of amplification 
        delta_x_end = (x_stim[-1] - .5)
        delta_y_end = (y_stim[-1] - .5)
        delta_t = (self.params['t_iteration'] / self.params['t_cross_visual_field'])
        k = self.params['supervisor_amp_param']

        # omniscient supervisor
        self.supervisor_state[0] = k * delta_x_end / delta_t + self.current_motion_params[2]
        self.supervisor_state[1] = k * delta_y_end / delta_t + self.current_motion_params[3]


        local_gids = np.array(local_gids) - 1 # because PyNEST uses 1-aligned GIDS 
#        print 'debug shape motion_params', self.motion_params.shape, self.iteration
        self.motion_params[self.iteration, -1] = self.t_current
        self.create_spike_trains_for_trajectory(local_gids, trajectory)
        self.motion_params[self.iteration, :self.n_stim_dim] = self.current_motion_params # store the current motion parameters before they get updated

        self.iteration += 1
        return self.stim, self.supervisor_state




    def create_spike_trains_for_trajectory(self, local_gids, trajectory, save_rate_files=False):
        """
        Arguments:
        local_gids -- list of gids for which a stimulus shall be created
        trajectory -- two - dimensional (x, y) array of where the stimulus is positioned over time
        """
        self.stim = [ [] for gid in xrange(len(local_gids))]

        dt = self.params['dt_input_mpn'] # [ms] time step for the non-homogenous Poisson process 

        time = np.arange(0, self.params['t_iteration'], dt)
        n_cells = len(local_gids)
        L_input = np.zeros((n_cells, time.shape[0]))
        for i_time, time_ in enumerate(time):
            x_stim = trajectory[0][i_time]
            y_stim = trajectory[1][i_time]
            motion_params = (x_stim, y_stim, self.current_motion_params[2], self.current_motion_params[3])
            # get the envelope of the Poisson process for this timestep
            L_input[:, i_time] = self.get_input_new(self.tuning_prop_exc[local_gids, :], self.rf_sizes[local_gids, 0], self.rf_sizes[local_gids, 2], motion_params, \
                    self.params['blur_X'], self.params['blur_V']) 
            L_input[:, i_time] *= self.params['f_max_stim']
#            L_input[:, i_time] = self.get_input(self.tuning_prop_exc[local_gids, :], motion_params) 

        input_nspikes = np.zeros((len(local_gids), 2))
        # depending on trajectory and the tp create a spike train
        for i_, gid in enumerate(local_gids):
            rate_of_t = np.array(L_input[i_, :]) 
            n_steps = rate_of_t.size
            st = []
            for i in xrange(n_steps):
                r = np.random.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    st.append(i * dt + self.t_current) 
            input_nspikes[i_, :] = (gid, len(st))
            self.stim[i_] = st

        self.t_current += self.params['t_iteration']
#        if self.params['debug_mpn']:
#            np.savetxt(self.params['input_nspikes_fn_mpn'] + 'it%d_%d.dat' % (self.iteration, self.pc_id), input_nspikes, fmt='%d\t%d')
        return self.stim


    def get_input(self, tuning_prop, motion_params):
        """
        Arguments:
        tuning_prop: the 4-dim tuning properties of local cells
        motion_params: 4-element tuple with the current stimulus position and direction
        """

        # TODO: 
        # iteration over cells, look up tuning width (blur_x/v) for cell_gid
        n_cells = tuning_prop[:, 0].size
        blur_X, blur_V = self.params['blur_X'], self.params['blur_V'] #0.5, 0.5
        x_stim, y_stim, u_stim, v_stim = motion_params[0], motion_params[1], motion_params[2], motion_params[3]
        if self.params['n_grid_dimensions'] == 2:
            d_ij = visual_field_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
            L = np.exp(-.5 * (d_ij)**2 / blur_X**2 
                    -.5 * (tuning_prop[:, 2] - u_stim)**2 / blur_V**2
                    -.5 * (tuning_prop[:, 3] - v_stim)**2 / blur_V**2)
        else:
            d_ij = np.sqrt((tuning_prop[:, 0] - x_stim * np.ones(n_cells))**2)
            L = np.exp(-.5 * (d_ij)**2 / blur_X**2 \
                       -.5 * (tuning_prop[:, 2] - u_stim)**2 / blur_V**2)
        return L


    def get_input_new(self, tuning_prop, rfs_x, rfs_v, motion_params, blur_x, blur_v):
        """
        Arguments:
        tuning_prop: the 4-dim tuning properties of local cells
        rfs_x: the tuning widths (receptive field sizes) of local cells (corresponding to the tuning prop)
        motion_params: 4-element tuple with the current stimulus position and direction
        """

        # TODO: 
        # iteration over cells, look up tuning width (blur_x/v) for cell_gid
        n_cells = tuning_prop[:, 0].size
        x_stim, y_stim, u_stim, v_stim = motion_params[0], motion_params[1], motion_params[2], motion_params[3]
        if self.params['n_grid_dimensions'] == 2:
            d_ij = visual_field_distance2D_vec(tuning_prop[:, 0], x_stim * np.ones(n_cells), tuning_prop[:, 1], y_stim * np.ones(n_cells))
            L = np.exp(-.5 * (d_ij)**2 / (rfs_x * blur_x)**2 \
                    -.5 * (tuning_prop[:, 2] - u_stim)**2 / (np.sqrt(rfs_v * blur_v))**2
                    -.5 * (tuning_prop[:, 3] - v_stim)**2 / (np.sqrt(rfs_v * blur_v))**2)
        else:
            d_ij = np.abs(tuning_prop[:, 0] - x_stim)
            L = np.exp(-.5 * (d_ij)**2 / (np.sqrt(rfs_x * blur_x))**2 \
                       -.5 * (tuning_prop[:, 2] - u_stim)**2 / (np.sqrt(rfs_v * blur_v))**2)
        return L


    def update_stimulus_trajectory_static(self, v_eye):
        """
        During one iteration the stimulus is perceived as static, except for the movement given by the 
        difference between eye (= v_eye) and the stimulus
        """
        n_steps = self.params['t_iteration'] / self.params['dt_input_mpn']
        time_axis = np.arange(0, self.params['t_iteration'], self.params['dt_input_mpn'])
        x_stim = self.current_motion_params[0] - (v_eye[0] * self.params['t_iteration'] * np.ones(n_steps) + time_axis * self.current_motion_params[2]) / self.params['t_cross_visual_field']
        y_stim = self.current_motion_params[1] - (v_eye[1] * self.params['t_iteration'] * np.ones(n_steps) + time_axis * self.current_motion_params[3]) / self.params['t_cross_visual_field']

        trajectory = (x_stim, y_stim)
        self.current_motion_params[0] = x_stim[0]
        self.current_motion_params[1] = y_stim[0]
        # compute the supervisor signal taking into account:
        # - the trajectory position at the end of the iteration
        # - the knowledge about the motion (current_motion_params
        delta_x_end = (x_stim[-1] - .5)
        delta_y_end = (y_stim[-1] - .5)
        delta_t = (self.params['t_iteration'] / self.params['t_cross_visual_field'])
        k = self.params['supervisor_amp_param']

        # omniscient supervisor computes the 'correct' action to take
        self.supervisor_state[0] = k * delta_x_end / delta_t + self.current_motion_params[2]
        self.supervisor_state[1] = k * delta_y_end / delta_t + self.current_motion_params[3]

        self.motion_params[self.iteration, :self.n_stim_dim] = self.current_motion_params # store the current motion parameters before they get updated
        self.motion_params[self.iteration, -1] = self.t_current

        return trajectory, self.supervisor_state



    def update_stimulus_trajectory_new(self, action_code):
        """
        Update the motion parameters based on the action

        Keyword arguments:
        action_code -- a tuple representing the action (direction of eye movement)
        """
        time_axis = np.arange(0, self.params['t_iteration'], self.params['dt_input_mpn'])

        # calculate where the stimulus will move according to the current_motion_params
        x_stim = (self.current_motion_params[2] - action_code[0]) * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[0]
        y_stim = (self.current_motion_params[3] - action_code[1]) * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[1]
        trajectory = (x_stim, y_stim)

        # update the current motion parameters based on the action that was selected for this iteration
        self.current_motion_params[0] = x_stim[-1]
        self.current_motion_params[1] = y_stim[-1]
        # TODO: try this in addition

        # compute the supervisor signal taking into account:
        # - the trajectory position at the end of the iteration
        # - the knowledge about the motion (current_motion_params
        # - and / or the 
        delta_x_end = (x_stim[-1] - .5)
        delta_y_end = (y_stim[-1] - .5)
        delta_t = (self.params['t_iteration'] / self.params['t_cross_visual_field'])
        k = self.params['supervisor_amp_param']

        # omniscient supervisor computes the 'correct' action to take
        self.supervisor_state[0] = k * delta_x_end / delta_t + self.current_motion_params[2]
        self.supervisor_state[1] = k * delta_y_end / delta_t + self.current_motion_params[3]

        self.motion_params[self.iteration, :self.n_stim_dim] = self.current_motion_params # store the current motion parameters before they get updated
        self.motion_params[self.iteration, -1] = self.t_current

        return trajectory, self.supervisor_state


    def update_stimulus_trajectory_OLD(self, action_code, v_eye, network_state):
        """
        Update the motion parameters based on the action

        Keyword arguments:
        action_code -- a tuple representing the action (direction of eye movement)
        v_eye -- [vx, vy] eye velocity 
        network_state -- [x, y, vx, vy] 'prediction' based on sensory neurons
        network_state[2:4]  = v_object
        """

        t_integrate = self.params['t_iteration']
        time_axis = np.arange(0, t_integrate, self.params['dt_input_mpn'])

        # store the motion parameters at the beginning of this iteration
        print 'before update current motion parameters', self.current_motion_params, 'action_code', action_code
        self.current_motion_params[2] -= action_code[0]  # update v_stim_x
        self.current_motion_params[3] -= action_code[1]  # update v_stim_y

#         calculate how the stimulus will move according to these motion parameters
        x_stim = self.current_motion_params[2] * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[0]
        y_stim = self.current_motion_params[3] * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[1]

        # update the retinal position to the position of the stimulus at the end of the iteration
        self.current_motion_params[0] = x_stim[-1]
        self.current_motion_params[1] = y_stim[-1]
        print 'after update current motion parameters', self.current_motion_params
        trajectory = (x_stim, y_stim)
        
        delta_x = (x_stim[-1] - .5)
        delta_y = (y_stim[-1] - .5)
        delta_t = (self.params['t_iteration'] / self.params['t_cross_visual_field'])
        k = self.params['supervisor_amp_param']
        self.supervisor_state[0] = k * np.abs(delta_x) / .5 * delta_x / delta_t + network_state[2] + self.current_motion_params[2]
        self.supervisor_state[1] = k * np.abs(delta_y) / .5 * delta_y / delta_t + network_state[3] + self.current_motion_params[3]
        print 'Supervised_state[%d] = ' % (self.iteration), self.supervisor_state

        print 'self.motion_params', self.motion_params[self.iteration, :]
        self.motion_params[self.iteration, :self.n_stim_dim] = self.current_motion_params # store the current motion parameters before they get updated
        self.motion_params[self.iteration, -1] = self.t_current
#        self.trajectories.append(trajectory) # store for later save 

        return trajectory, self.supervisor_state

    


    def set_receptive_fields(self, cell_type):
        """
        Can be called only after set_tuning_prop.
        Receptive field sizes increase linearly depending on their relative position.
        """
        n_cells = self.params['n_exc_mpn']
        rfs = np.zeros((n_cells, 4))
        rfs[:, 0] = utils.get_receptive_field_sizes_x(self.params, self.tuning_prop_exc[:, 0])
        rfs[:, 1] = utils.get_receptive_field_sizes_x(self.params, self.tuning_prop_exc[:, 1])
        rfs[:, 2] = utils.get_receptive_field_sizes_v(self.params, self.tuning_prop_exc[:, 2])
        rfs[:, 3] = utils.get_receptive_field_sizes_v(self.params, self.tuning_prop_exc[:, 3])
#        rfs[:, 0] = self.params['rf_size_x_gradient'] * np.abs(self.tuning_prop_exc[:, 0] - .5) + self.params['rf_size_x_min']
#        rfs[:, 1] = self.params['rf_size_y_gradient'] * np.abs(self.tuning_prop_exc[:, 1] - .5) + self.params['rf_size_y_min']
#        rfs[:, 2] = self.params['rf_size_vx_gradient'] * np.abs(self.tuning_prop_exc[:, 2]) + self.params['rf_size_vx_min']
#        rfs[:, 3] = self.params['rf_size_vy_gradient'] * np.abs(self.tuning_prop_exc[:, 3]) + self.params['rf_size_vy_min']
        return rfs


    def set_tuning_prop(self, cell_type):

        if self.params['n_grid_dimensions'] == 2:
            return self.set_tuning_prop_2D(mode, cell_type)
        else:
            if self.params['regular_tuning_prop']:
                return self.set_tuning_prop_1D_regular(cell_type)
            else:
                return self.set_tuning_prop_1D_with_const_fovea(cell_type)
#                return self.set_tuning_prop_1D(cell_type)


    def set_tuning_prop_1D_regular(self, cell_type='exc'):
        if cell_type == 'exc':
            n_cells = self.params['n_exc_mpn']
            n_v = self.params['n_v']
            n_rf_x = self.params['n_rf_x']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        else:
            n_cells = self.params['n_inh_mpn']
            n_v = self.params['n_v_inh']
            n_rf_x = self.params['n_rf_x_inh']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']

#        v_rho_half_1 = np.linspace(v_min, v_max, num=n_v/2, endpoint=True)
#        v_rho_half_2 = np.linspace(v_rho_half_1[1], v_max, num=n_v/2, endpoint=True)
#        v_rho = np.zeros(n_v)
#        v_rho[:n_v/2] = -v_rho_half_1
#        v_rho[n_v/2:] = v_rho_half_2
        v_rho = np.linspace(-v_max, v_max, num=n_v, endpoint=True)
        RF = np.linspace(self.params['x_min_tp'], self.params['x_max_tp'], n_rf_x, endpoint=True)
        index = 0
        tuning_prop = np.zeros((n_cells, 4))

        for i_RF in xrange(n_rf_x):
            for i_v_rho, rho in enumerate(v_rho):
                for i_in_mc in xrange(self.params['n_exc_per_state']):
#                    tuning_prop[index, 0] = RF[index]
                    tuning_prop[index, 0] = RF[i_RF]
                    tuning_prop[index, 1] = 0.5 
                    tuning_prop[index, 2] = rho
                    tuning_prop[index, 3] = 0. 
                    index += 1
        assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
        return tuning_prop


    def set_tuning_prop_1D_with_const_fovea(self, cell_type='exc'):
        np.random.seed(self.params['tuning_prop_seed'])
        if cell_type == 'exc':
            n_cells = self.params['n_exc_mpn']
            n_v = self.params['n_v']
            n_rf_x = self.params['n_rf_x']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        else:
            n_cells = self.params['n_inh_mpn']
            n_v = self.params['n_v_inh']
            n_rf_x = self.params['n_rf_x_inh']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        if self.params['log_scale']==1:
            v_rho_half = np.linspace(v_min, v_max, num=n_v/2, endpoint=True)
        else:
            v_rho_half = np.logspace(np.log(v_min)/np.log(self.params['log_scale']),
                            np.log(v_max)/np.log(self.params['log_scale']), num=n_v/2,
                            endpoint=True, base=self.params['log_scale'])
        self.rf_sizes = np.zeros((n_cells, 4))
        v_rho = np.zeros(n_v)
        v_rho[:n_v/2] = -v_rho_half
        v_rho[n_v/2:] = v_rho_half
        
        n_rf_x_log = self.params['n_rf_x'] - self.params['n_rf_x_fovea']
        RF_x_log = utils.get_xpos_log_distr(self.params['log_scale'], n_rf_x_log, x_min=self.params['x_min_tp'], x_max=self.params['x_max_tp'])
        RF_x_const = np.linspace(.5 - self.params['x_min_tp'], .5 + self.params['x_min_tp'], self.params['n_rf_x_fovea'])
        RF_x = np.zeros(n_rf_x)
        idx_upper = n_rf_x_log / 2 + self.params['n_rf_x_fovea']
        RF_x[:n_rf_x_log / 2] = RF_x_log[:n_rf_x_log / 2]
        RF_x[idx_upper:] = RF_x_log[n_rf_x_log / 2:]
        RF_x[n_rf_x_log / 2 : n_rf_x_log / 2 + self.params['n_rf_x_fovea']] = RF_x_const



#        print '------------------------------\nDEBUG'
#        print 'n_rf_x: ', n_rf_x
#        print 'n_rf_x_log: ', n_rf_x_log
#        print 'n_rf_x_fovea: ', self.params['n_rf_x_fovea']
#        print 'RF_x_const:', RF_x_const
#        print 'RF_x_log:', RF_x_log
#        print 'RF_x:', RF_x

        index = 0
        tuning_prop = np.zeros((n_cells, 4))
        rf_sizes_x = utils.get_receptive_field_sizes_x(self.params, RF_x)
        rf_sizes_v = utils.get_receptive_field_sizes_v(self.params, v_rho)
        for i_RF in xrange(n_rf_x):
            for i_v_rho, rho in enumerate(v_rho):
                for i_in_mc in xrange(self.params['n_exc_per_state']):
                    x = RF_x[i_RF]
#                    tuning_prop[index, 0] = (x + np.abs(x - .5) / .5 * self.RNG.uniform(-self.params['sigma_rf_pos'] , self.params['sigma_rf_pos'])) % 1.
                    tuning_prop[index, 0] = RF_x[i_RF]
                    tuning_prop[index, 0] += self.RNG.normal(.0, self.params['sigma_rf_pos'] / 2) # add some extra noise to the neurons representing the fovea (because if their noise is only a percentage of their distance from the center, it's too small
                    tuning_prop[index, 0] = tuning_prop[index, 0] % 1.0
                    tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
                    tuning_prop[index, 2] = (-1)**(i_v_rho % 2) * rho * (1. + self.params['sigma_rf_speed'] * np.random.randn())
                    tuning_prop[index, 3] = 0. 
                    self.rf_sizes[index, 0] = rf_sizes_x[i_RF]
                    self.rf_sizes[index, 2] = rf_sizes_v[i_v_rho]
                    index += 1



        assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
#        exit(1)
        return tuning_prop




    def set_tuning_prop_1D(self, cell_type='exc'):

        np.random.seed(self.params['tuning_prop_seed'])
        if cell_type == 'exc':
            n_cells = self.params['n_exc_mpn']
            n_v = self.params['n_v']
            n_rf_x = self.params['n_rf_x']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        else:
            n_cells = self.params['n_inh_mpn']
            n_v = self.params['n_v_inh']
            n_rf_x = self.params['n_rf_x_inh']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        if self.params['log_scale']==1:
            v_rho_half = np.linspace(v_min, v_max, num=n_v/2, endpoint=True)
        else:
            v_rho_half = np.logspace(np.log(v_min)/np.log(self.params['log_scale']),
                            np.log(v_max)/np.log(self.params['log_scale']), num=n_v/2,
                            endpoint=True, base=self.params['log_scale'])

#        n_cells = self.params['n_exc_mpn']
#        self.rf_sizes = self.set_receptive_fields('exc')
        self.rf_sizes = np.zeros((n_cells, 4))

        v_rho = np.zeros(n_v)
        v_rho[:n_v/2] = -v_rho_half
        v_rho[n_v/2:] = v_rho_half
#        RF = np.random.normal(0.5, self.params['sigma_rf_pos'], n_cells)
        RF_x = utils.get_xpos_log_distr(self.params['logscale'], n_rf_x, x_min=self.params['x_min_tp'], x_max=self.params['x_max_tp'])
        RF_x = RF_x % self.params['visual_field_width']
        index = 0
        tuning_prop = np.zeros((n_cells, 4))
        rf_sizes_x = utils.get_receptive_field_sizes_x(self.params, RF_x)
        rf_sizes_v = utils.get_receptive_field_sizes_v(self.params, v_rho)

        for i_RF in xrange(n_rf_x):
            for i_v_rho, rho in enumerate(v_rho):
                for i_in_mc in xrange(self.params['n_exc_per_state']):
                    x = RF_x[i_RF]
                    tuning_prop[index, 0] = (x + np.abs(x - .5) / .5 * self.RNG.uniform(-self.params['sigma_rf_pos'] , self.params['sigma_rf_pos'])) % 1.
                    tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
                    tuning_prop[index, 2] = (-1)**(i_v_rho % 2) * rho * (1. + self.params['sigma_rf_speed'] * np.random.randn())
                    tuning_prop[index, 3] = 0. 
                    self.rf_sizes[index, 0] = rf_sizes_x[i_RF]
                    self.rf_sizes[index, 2] = rf_sizes_v[i_v_rho]
                    index += 1
        assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
#        exit(1)
        return tuning_prop




    def set_tuning_prop_2D(self, cell_type='exc'):
        """
        Place n_exc excitatory cells in a 4-dimensional space.
        The position of each cell represents its excitability to a given a 4-dim stimulus.
        The radius of their receptive field is assumed to be constant.
        return value:
            tp = set_tuning_prop(self.params)
            tp[:, 0] : x-position
            tp[:, 1] : y-position
            tp[:, 2] : u-position (speed in x-direction)
            tp[:, 3] : v-position (speed in y-direction)
        All x-y values are in range [0..1].         By convention, velocity is such that V=(1,0) corresponds to one horizontal spatial period in one temporal period.
        All u, v values are in the range -params[v_max_tp] .. params['v_max_tp']
        """

        np.random.seed(self.params['tuning_prop_seed'])
        if cell_type == 'exc':
            n_cells = self.params['n_exc_mpn']
            n_v = self.params['n_v']
            n_rf_x = self.params['n_rf_x']
            n_rf_y = self.params['n_rf_y']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
        else:
            n_cells = self.params['n_inh_mpn']
            n_v = self.params['n_v_inh']
            n_rf_x = self.params['n_rf_x_inh']
            v_max = self.params['v_max_tp']
            v_min = self.params['v_min_tp']
            n_rf_x = self.params['n_rf_x_inh']
            n_rf_y = self.params['n_rf_y_inh']
            if n_v == 1:
                v_min = self.params['v_min_tp'] + .5 * (self.params['v_max_tp'] - self.params['v_min_tp'])
                v_max = v_min
            else:
                v_max = self.params['v_max_tp']
                v_min = self.params['v_min_tp']

        n_theta = self.params['n_theta'] 

        tuning_prop = np.zeros((n_cells, 4))
        # distribution of speed vectors (= length of the preferred direction vectors)
        if self.params['log_scale']==1:
            v_rho = np.linspace(v_min, v_max, num=n_v, endpoint=True)
        else:
            v_rho = np.logspace(np.log(v_min)/np.log(self.params['log_scale']),
                            np.log(v_max)/np.log(self.params['log_scale']), num=n_v,
                            endpoint=True, base=self.params['log_scale'])


        parity = np.arange(self.params['n_v']) % 2

        # wrapping up:
        index = 0
        if self.params['n_grid_dimensions'] == 1:
            random_rotation = np.zeros(n_cells)
            v_theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        else:
            random_rotation = 2*np.pi*np.random.rand(n_rf_x * n_rf_y * n_v * n_theta) * self.params['sigma_rf_direction']
            v_theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)

        for i_RF in xrange(n_rf_x * n_rf_y):
            for i_v_rho, rho in enumerate(v_rho):
                for i_theta, theta in enumerate(v_theta):
                    for i_ in xrange(self.params['n_exc_per_state']):
                        tuning_prop[index, 0] = np.random.uniform()
                        tuning_prop[index, 1] = np.random.uniform()
                        tuning_prop[index, 2] = np.cos(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                                * rho * (1. + self.params['sigma_rf_speed'] * np.random.randn())
                        tuning_prop[index, 3] = np.sin(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta) \
                                * rho * (1. + self.params['sigma_rf_speed'] * np.random.randn())
                        index += 1
        assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
        if self.params['n_grid_dimensions'] == 1:
            tuning_prop[:, 1] = .5
            tuning_prop[:, 3] = .0

        return tuning_prop


    def get_gids_near_stim_trajectory(self, verbose=False):

        self.gids_to_record_exc, distances = utils.sort_gids_by_distance_to_stimulus(self.tuning_prop_exc, self.current_motion_params, \
                self.t_current, self.t_current + self.params['t_iteration'], self.params['t_cross_visual_field'])
        if verbose:
            print 'Motion parameters', self.current_motion_params
            print 'GID\tdist_to_stim\tx\ty\tu\tv\t\t'
            for i in xrange(self.params['n_exc_mpn']):
                gid = self.gids_to_record_exc[i]
                print gid, '\t', distances[i], self.tuning_prop_exc[gid, :]


        return self.gids_to_record_exc


    def set_empty_input(self, local_gids):
        """
        At the last iteration for each stimulus return an empty spike train
        """

        local_gids = np.array(local_gids)
        for i_ in xrange(len(local_gids)):
            self.stim[i_] = []
        self.motion_params[self.iteration, -1] = self.t_current
        self.t_current += self.params['t_iteration']
        self.iteration += 1
        self.supervisor_state = [0., 0.]
        return self.stim, self.supervisor_state


    def set_pc_id(self, pc_id):
        self.pc_id = pc_id


    def create_dummy_stim(self, local_gids, action_code=0):
        """
        Keyword arguments:
        local_gids -- list of gids for which a stimulus shall be created
        action_code -- a tuple representing the action (direction of eye movement)
        """
        t_integrate = self.params['t_iteration']
        print 'Creating dummy spike trains', self.t_current
        stim = [ [] for gid in xrange(len(local_gids))]

        for i_, gid in enumerate(local_gids):
            # get the cell from the list of populations
            mc_idx = (gid - 1) / self.params['n_exc_per_mc']
            idx_in_pop = (gid - 1) - mc_idx * self.params['n_exc_per_mc']
            if mc_idx == action_code:
                n_spikes = np.random.randint(20, 50)
                stim[i_] = np.around(np.random.rand(n_spikes) * t_integrate + self.t_current, decimals=1)
                stim[i_] = np.sort(stim[i_])
        self.t_current += t_integrate
        self.iteration += 1
        return stim

