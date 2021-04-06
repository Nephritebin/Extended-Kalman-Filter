import numpy as np
from matplotlib import pyplot as plt
import math

# Remove some frames in animations
remove_step = 5

# Simulation parameters: noise matrix
# Q is for motion [Vx Vy psi] and R is for observation [x y yaw]
noiseQ = np.diag([0.1, 0, math.radians(10)]) ** 2
noiseR = np.diag([0.5, 0.5, math.radians(5)]) ** 2
# Considering the independence, calculate the conv matrix Q and R
convQ = np.matmul(noiseQ, noiseQ)
convR = np.matmul(noiseR, noiseR)


class Estimation:
    def __init__(self, time, v, ekf_now, truth, odom, gps):
        self.time = time
        self.u = v
        self.gps = gps
        self.odom = odom
        self.truth = truth

        self.ekf_pre = ekf_now
        self.conv = np.zeros(3)

    def error(self, conv):
        self.conv = conv


def robot_control(time, end_time):
    t = 10  # [sec]
    x = 1.0  # [m/s]
    y = 0.2  # [m/s]
    psi = 5  # [deg/s]

    if time > end_time / 2:
        psi = -5

    v = [x*(1 - math.exp(-time/t)), y*(1 - math.exp(-time/t)), math.radians(psi)*(1 - math.exp(-time/t))]
    v = np.array(v).reshape(3, 1)
    return v


def state_init():
    # State vector [x, y, yaw]'
    ekf_now = np.zeros((3, 1))
    # Ground true state
    truth = ekf_now
    # Odometry only
    odom = ekf_now
    # Observation vector[x, y, yaw]'
    gps = np.zeros((3, 1))
    v = robot_control(0, np.inf)
    state = Estimation(0, v, ekf_now, truth, odom, gps)

    return state


def do_motion(dt, pos_old, v):
    matrix = np.array([[math.cos(pos_old[2][0]), -math.sin(pos_old[2][0]), 0.],
                       [math.sin(pos_old[2][0]),  math.cos(pos_old[2][0]), 0.],
                       [0., 0., 1.]])

    pos_new = pos_old + np.matmul(matrix, v) * dt

    return pos_new


def do_observation(pos):
    obs = np.matmul(np.eye(3), pos) + np.matmul(noiseR, np.random.randn(3, 1))

    return obs


def prepare(state, v, dt):
    truth = do_motion(dt, state.truth, v)
    v = v + np.matmul(noiseQ, np.random.randn(3, 1))
    odom = do_motion(dt, state.odom, v)
    gps = do_observation(truth)  # truth + np.matmul(noiseR, np.random.randn(3, 1))

    return gps, truth, odom, v


def jacob_f(pos, v, dt):
    F = np.array([[1., 0., -(math.sin(pos[2][0]) * v[0][0] + math.cos(pos[2][0]) * v[1][0])],
                  [0., 1.,  (math.cos(pos[2][0]) * v[0][0] - math.sin(pos[2][0]) * v[1][0])],
                  [0., 0., 1.]])

    return F


def jacob_h(pos):
    H = np.eye(3)
    return H


def kalman_filter(state_old, gps, v, dt):
    pos_pre = do_motion(dt, state_old.ekf_pre, v)

    F = jacob_f(pos_pre, v, dt)
    H = jacob_h(pos_pre)

    # Transfer the error from control space to state space
    matrix_trans = np.array([[math.cos(state_old.ekf_pre[2]), -math.sin(state_old.ekf_pre[2]), 0.],
                             [math.sin(state_old.ekf_pre[2]),  math.cos(state_old.ekf_pre[2]), 0.],
                             [0., 0., 1.]])  # * dt
    conQ_to_pos = np.matmul(np.matmul(matrix_trans, convQ), np.transpose(matrix_trans))
    conv_pre = np.matmul(np.matmul(F, state_old.conv), np.transpose(F)) + conQ_to_pos

    obs = do_observation(pos_pre)

    K = np.linalg.inv(np.matmul(np.matmul(H, conv_pre), np.transpose(H)) + convR)
    K = np.matmul(np.matmul(conv_pre, np.transpose(H)), K)

    ekf_now = pos_pre + np.matmul(K, gps - obs)
    conv_now = np.matmul((np.eye(3) - np.matmul(K, H)), conv_pre)

    return ekf_now, conv_now


def error_calculate(x1, y1, x2, y2):
    error_sum = 0.
    for i in range(len(x1)):
        error = math.sqrt((x2[i] - x1[i]) ** 2 + (y2[i] - y1[i]) ** 2)
        error_sum = error_sum + error
    error_mean = error_sum / len(x1)

    return error_mean


def final_plot(save_state, save_dir):
    plot_true_x = []
    plot_true_y = []
    plot_ekf_x = []
    plot_ekf_y = []
    plot_gps_x = []
    plot_gps_y = []
    plot_odom_x = []
    plot_odom_y = []
    for i in range(len(save_state)):
        plot_true_x.append(save_state[i].truth[0])
        plot_true_y.append(save_state[i].truth[1])
        plot_gps_x.append(save_state[i].gps[0])
        plot_gps_y.append(save_state[i].gps[1])
        plot_odom_x.append(save_state[i].odom[0])
        plot_odom_y.append(save_state[i].odom[1])
        plot_ekf_x.append(save_state[i].ekf_pre[0])
        plot_ekf_y.append(save_state[i].ekf_pre[1])

    error1 = error_calculate(plot_odom_x, plot_odom_y, plot_true_x, plot_true_y)
    error2 = error_calculate(plot_ekf_x, plot_ekf_y, plot_true_x, plot_true_y)
    print("Only odometer error %f" % error1)
    print("EKF estimation error %f" % error2)

    plt.figure()
    plt.title('Result Plot')
    plt.xlabel('X (meter)')
    plt.ylabel('Y (meter)')

    plt.scatter(plot_gps_x, plot_gps_y, 4, alpha=0.5)
    plt.scatter(plot_odom_x, plot_odom_y, 4, alpha=0.5)
    plt.scatter(plot_ekf_x, plot_ekf_y, 4, alpha=0.4)
    plt.scatter(plot_true_x, plot_true_y, 4, alpha=0.4)
    plt.legend(['GPS Observations', 'Odometry Only', 'EKF Localization', 'Ground Truth'])

    plt.grid()
    plt.savefig(save_dir)
    plt.show()

