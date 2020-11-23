import numpy as np
from matplotlib import pyplot as plt
from scipy import io
import math
import ekf_alg as ekf


# Beginning time and ending time, [sec]
time = 0
end_time = 60
# An iteration step duration, [sec]
dt = 0.1
# Steps number
steps = (end_time - time) / dt
steps = math.ceil(steps)
# Save the states in steps
save_state = []
save_data_dir = 'data.mat'
save_image_dir = 'result.jpg'

# todo initialize the states
state = ekf.state_init()
save_state.append(state)

print("EKF begin!")

for i in range(steps):
    time = time + dt
    v = ekf.robot_control(time, end_time)
    gps, truth, odom, v = ekf.prepare(save_state[-1], v, dt)

    # todo Kalman Filter
    ekf_now, conv_now = ekf.kalman_filter(save_state[-1], gps, v, dt)

    state = ekf.Estimation(time, v, ekf_now, truth, odom, gps)
    state.error(conv_now)
    save_state.append(state)

print('EKF have already finished!')

# todo error estimation and plot
ekf.final_plot(save_state, save_image_dir)
io.savemat(save_data_dir, {'data': save_state})


