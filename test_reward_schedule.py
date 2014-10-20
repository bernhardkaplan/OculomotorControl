import numpy as np
import pylab
import utils


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


if __name__ == '__main__':

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
    pylab.show()
    #    print '%2d\t%.2f\t%.2f\t%.2f\t%.2f' % (i_, x[i_], dx[i_], dx_abs[i_], dj_di_abs[i_])
