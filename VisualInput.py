import numpy as np
import json
import utils

class VisualInput(object):

    def __init__(self, params, pc_id=0):
        """
        Keyword arguments
        params -- dictionary that contains 
        """
        self.params = params
        self.pc_id = pc_id
        self.trajectories = []
        self.t_axis = np.arange(0, self.params['t_iteration'], self.params['dt'])
        self.iteration = 0
        self.t_current = 0 # stores the 'current' time
        np.random.seed(self.params['visual_stim_seed'])

        self.tuning_prop_exc = self.set_tuning_prop('exc')
        self.tuning_prop_inh = self.set_tuning_prop('inh')
        print 'Saving tuning properties exc to:', self.params['tuning_prop_exc_fn']
        print 'Saving tuning properties inh to:', self.params['tuning_prop_inh_fn']
        np.savetxt(self.params['tuning_prop_exc_fn'], self.tuning_prop_exc)
        np.savetxt(self.params['tuning_prop_inh_fn'], self.tuning_prop_inh)

        self.current_motion_params = list(self.params['initial_state'])
        # store the motion parameters seen on the retina
        self.n_stim_dim = len(self.params['initial_state'])
        self.motion_params = np.zeros((self.params['n_iterations'], self.n_stim_dim + 1))  # + 1 dimension for the time axis

#        self.get_gids_near_stim_trajectory(verbose=self.params['debug_mpn'])



    def compute_input(self, local_gids, action_code, dummy=False):
        """
        Integrate the real world trajectory and the eye direction and compute spike trains from that.

        Keyword arguments:
        local_gids -- the GIDS for which the stimulus needs to be computed
        action_code -- a tuple representing the action (direction of eye movement)
        """

        trajectory = self.update_stimulus_trajectory(action_code)
        local_gids = np.array(local_gids) - 1 # because PyNEST uses 1-aligned GIDS --> grrrrr :(
        self.create_spike_trains_for_trajectory(local_gids, trajectory)

        self.iteration += 1

#        supervisor_state =  (self.trajectories[-1][0][-1], self.trajectories[-1][1][-1], self.current_motion_params[2], self.current_motion_params[3])
        supervisor_state = (trajectory[0][-1], trajectory[1][-1], self.current_motion_params[2], self.current_motion_params[3])
        return self.stim, supervisor_state


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
            if (i_time % 1000 == 0) and (self.pc_id == 0):
                print "t:", time_
            x_stim = trajectory[0][i_time]
            y_stim = trajectory[1][i_time]
            motion_params = (x_stim, y_stim, self.current_motion_params[2], self.current_motion_params[3])
            # get the envelope of the Poisson process for this timestep
            L_input[:, i_time] = self.get_input(self.tuning_prop_exc[local_gids, :], motion_params) 
            L_input[:, i_time] *= self.params['f_max_stim']

        input_nspikes = np.zeros((len(local_gids), 2))
        # depending on trajectory and the tp create a spike train
        for i_, gid in enumerate(local_gids):
            rate_of_t = np.array(L_input[i_, :]) 
            output_fn = self.params['input_rate_fn_mpn'] + str(gid) + '.dat'
            n_steps = rate_of_t.size
            st = []
            for i in xrange(n_steps):
                r = np.random.rand()
                if (r <= ((rate_of_t[i]/1000.) * dt)): # rate is given in Hz -> 1/1000.
                    st.append(i * dt + self.t_current) 
            input_nspikes[i_, :] = (gid, len(st))
            self.stim[i_] = st

        self.t_current += self.params['t_iteration']
        np.savetxt(self.params['input_nspikes_fn_mpn'] + 'it%d_%d.dat' % (self.iteration, self.pc_id), input_nspikes, fmt='%d\t%d')
        return self.stim


    def get_input(self, tuning_prop, motion_params):
        """
        Arguments:
        tuning_prop: the 4-dim tuning properties of local cells
        motion_params: 4-element tuple with the current stimulus position and direction
        """

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


    def update_stimulus_trajectory(self, action_code):
        """
        Keyword arguments:
        action_code -- a tuple representing the action (direction of eye movement)
        """
        t_integrate = self.params['t_iteration']
        time_axis = np.arange(0, t_integrate, self.params['dt_input_mpn'])
        # update the motion parameters based on the action

        # store the motion parameters at the beginning of this iteration
        self.motion_params[self.iteration, :self.n_stim_dim] = self.current_motion_params # store the current motion parameters before they get updated
        self.motion_params[self.iteration, -1] = self.t_current

        print 'before update cur mot p', self.current_motion_params
        self.current_motion_params[0] -= action_code[0] # shift x-position by moving according to vx
        self.current_motion_params[1] -= action_code[1] # shift y-position by moving according to vy
        self.current_motion_params[2] -= action_code[0]  # update v_stim_x
        self.current_motion_params[3] -= action_code[1]  # update v_stim_y

        # calculate how the stimulus will move according to these motion parameters
        x_stim = self.current_motion_params[2] * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[0]
        y_stim = self.current_motion_params[3] * time_axis / self.params['t_cross_visual_field'] + np.ones(time_axis.size) * self.current_motion_params[1]
        
        # update the retinal position to the position of the stimulus at the end of the iteration
        self.current_motion_params[0] = x_stim[-1]
        self.current_motion_params[1] = y_stim[-1]
        print 'after update cur mot p', self.current_motion_params
        trajectory = (x_stim, y_stim)
#        self.trajectories.append(trajectory) # store for later save 

        return trajectory

    


    def set_tuning_prop(self, cell_type):

#        return self.set_tuning_prop_2D(cell_type)
        if self.params['n_grid_dimensions'] == 2:
            return self.set_tuning_prop_2D(mode, cell_type)
        else:
            return self.set_tuning_prop_1D(cell_type)


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

        v_rho = np.zeros(n_v)
        v_rho[:n_v/2] = -v_rho_half
        v_rho[n_v/2:] = v_rho_half
        RF = np.linspace(0, self.params['visual_field_width'], n_rf_x, endpoint=False)
        index = 0
        random_rotation_for_orientation = np.pi*np.random.rand(self.params['n_exc_per_mc'] * n_rf_x * n_v) * self.params['sigma_rf_orientation']

        tuning_prop = np.zeros((n_cells, 4))

        for i_RF in xrange(n_rf_x):
            for i_v_rho, rho in enumerate(v_rho):
                for i_in_mc in xrange(self.params['n_exc_per_state']):
                    tuning_prop[index, 0] = (RF[i_RF] + self.params['sigma_rf_pos'] * np.random.randn()) % self.params['visual_field_width']
                    tuning_prop[index, 1] = 0.5 # i_RF / float(n_rf_x) # y-pos 
                    tuning_prop[index, 2] = rho * (1. + self.params['sigma_rf_speed'] * np.random.randn())
                    tuning_prop[index, 3] = 0. 
                    index += 1
        print 'debug', n_v, n_rf_x, n_v * n_rf_x, self.params['n_exc_per_state'], cell_type
        assert (index == n_cells), 'ERROR, index != n_cells, %d, %d' % (index, n_cells)
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
#        print 'debug parity', parity

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
                        print 'debug', np.cos(theta + random_rotation[index] + parity[i_v_rho] * np.pi / n_theta)
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
#        stim = [ [] for unit in xrange(self.params['n_exc_per_mc'])]
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

