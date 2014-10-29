import numpy as np
import pylab
import utils
import simulation_parameters
import BasalGanglia
import utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm


def plot_reward_schedule(x, R):

    
    fig = pylab.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    N = len(x)
    for i_ in xrange(N):
        ax1.plot(i_, x[i_], 'o', markersize=10, c='k')
    ax1.plot(range(N), x, '-', ls='-', c='k', lw=2)

    ms_min = 2
    ms_max = 20
    for i_ in xrange(N - 1):
        if R[i_] > 0:
            c = 'r'
            s = '^'
            fillstyle = 'full'
        else:
            c = 'b'
            s = 'v'
            fillstyle = 'full' #'none'
        ms = np.abs(np.round(R[i_] / np.max(R) * ms_max)) + ms_min
        print 'R i_ markersize', R[i_], i_, ms
        ax2.plot(i_, R[i_], s, markersize=ms, c=c, fillstyle=fillstyle)

    ax1.plot((-.5, N + .5), (.5, .5), '--', c='k')
    ax1.set_xlim(-.5, N + .5)
    ax1.set_ylim((-.1, 1.1))

    fs = 18
    ax1.set_title('Reward schedule for random actions & positions', fontsize=fs)
    ax1.set_xlabel('Iteration', fontsize=fs)
    ax1.set_ylabel('Position', fontsize=fs)
    ax2.set_ylabel('Reward', fontsize=fs)


def get_reward(dx, dx_abs, dj_di_abs):
    n_it = dx.size
    R = np.zeros(n_it - 1)
    A = 1# amplify the improvement / worsening linearly
    B = .7  # punish overshoot, 0 <= B <= 1.
    for i_ in xrange(1, n_it):
        print 'i_ R, dj_di_abs', i_, R.size, dj_di_abs.size, i_ - 1
#        R[i_-1] = -1 * A * dj_di_abs[i_-1] * np.abs(dj_di_abs[i_-1])
        R[i_-1] = -2. * A * dj_di_abs[i_-1]
        if np.sign(dx[i_-1]) != np.sign(dx[i_]):
            R[i_-1] *= B
    return R


def test_random_placements():
    np.random.seed(0)
    n_iterations = 50
    x = np.zeros(n_iterations)
    for i_ in xrange(n_iterations):
        x[i_] = np.random.rand()
    dx = x - .5
    dx_abs = np.abs(dx)
    dj_di_abs = dx_abs[1:] - dx_abs[:-1]
    #R = get_reward(dx, dx_abs, dj_di_abs)
    R = np.zeros(n_iterations)
    for i_ in xrange(n_iterations-1):
        R[i_] = utils.get_reward_from_perceived_states(x[i_], x[i_+1])
    print 'Iteration x\tdx\tdx_abs\tdj_di_abs\tReward'
    for i_ in xrange(1, n_iterations):
        print '%2d\t%.2f\t%.2f\t%.2f' % (i_-1, x[i_-1], dx[i_-1], dx_abs[i_-1])
        print '%2d\t%.2f\t%.2f\t%.2f\t%.2f\t\t%.2f' % (i_, x[i_], dx[i_], dx_abs[i_], dj_di_abs[i_-1], R[i_-1])
        print '\n'
    plot_reward_schedule(x, R)
    pylab.savefig('Reward_schedule_with_random_positions_and_random_actions.png', dpi=200)


def get_rewards_for_all_stimuli_and_actions(n_pos=20, n_v=20):

    v_min = 0.
    v_max =  2.
    GP = simulation_parameters.global_parameters()
    params = GP.params
    BG = BasalGanglia.BasalGanglia(params, dummy=True)
    speeds = BG.action_bins_x

    x_pos = np.linspace(0., 1., n_pos)
    vx = np.linspace(v_min, v_max, n_v)
    print 'x_pos:', x_pos

#    R = np.zeros((n_pos, n_pos, n_v, params['n_actions']))
    rewards = np.zeros((n_pos, n_v, params['n_actions']))
    new_positions = np.zeros((n_pos, n_v, params['n_actions']))
    stimuli_and_rewards = np.zeros((n_pos * n_v * params['n_actions'], 6))
    idx = 0
    for i_x in xrange(n_pos):
        x_old = x_pos[i_x]
        for i_v in xrange(n_v):
            stim_params = [x_pos[i_x], .5, vx[i_v], 0.]
            for i_a in xrange(params['n_actions']):
                x_new = utils.get_next_stim(params, stim_params, speeds[i_a])[0]
                R = utils.get_reward_from_perceived_states(x_old, x_new)
#                new_positions[i_x, i_v, i_a] = x_new
#                rewards[i_x, i_v, i_a] = R
                stimuli_and_rewards[idx, 0] = x_old
                stimuli_and_rewards[idx, 1] = x_new
                stimuli_and_rewards[idx, 2] = vx[i_v]
                stimuli_and_rewards[idx, 3] = i_a
                stimuli_and_rewards[idx, 4] = speeds[i_a]
                stimuli_and_rewards[idx, 5] = R
                idx += 1


    return stimuli_and_rewards

#    print "BG.action_bins_x:", BG.action_bins_x


def plot_4d(d):

    fig = pylab.figure()
    ax = Axes3D(fig)
    pylab.rcParams['lines.markeredgewidth'] = 0

    min_size = 0
    max_size = 25
    color_code_axis = 5
    code = d[:, color_code_axis]
    norm = matplotlib.colors.Normalize(vmin=code.min(), vmax=code.max())
    m = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.jet)
    rgba_colors = m.to_rgba(code)
    marker_sizes = np.round(utils.linear_transformation(code, min_size, max_size))
    p = ax.scatter(d[:, 0], d[:, 1], d[:, 2], c=np.array(rgba_colors), marker='o', linewidth='0', edgecolor=rgba_colors, s=marker_sizes)#, cmap='seismic')
    m.set_array(code)
    fig.colorbar(m)

    ax.set_xlabel('Stimulus position')
    ax.set_ylabel('Position after action')
    ax.set_zlabel('Stimulus speed')


if __name__ == '__main__':

#    test_random_placements()
    all_data = get_rewards_for_all_stimuli_and_actions(n_pos=20, n_v=4)
    plot_4d(all_data)
    pylab.show()
    #    print '%2d\t%.2f\t%.2f\t%.2f\t%.2f' % (i_, x[i_], dx[i_], dx_abs[i_], dj_di_abs[i_])
