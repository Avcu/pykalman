import numpy as np
import matplotlib.pyplot as plt

from unscentedKF_withInput import AdditiveUnscentedKalmanFilter
from unscentedKF_withInput import AdditiveUnscentedKalmanFilter_withInput
from nonlinear_model_methods import nonlinear_state_space_model

## settings of the nonlinear state space model
settings_ss = {}
settings_ss['num_samples_trial'] = 200
settings_ss['dim_u'] = 1
settings_ss['dim_z'] = 1
settings_ss['dim_y'] = 3
settings_ss['noise_Q'] = 1e-1
settings_ss['noise_R'] = 1e-2

ss = nonlinear_state_space_model(settings_ss)
ss.generate_u()
ss.generate_z()
ss.generate_y()

## Plot input and states
# ss.plot_u()
# ss.plot_z()
# ss.plot_y()

## Plot the manifold
ss.plot_y_manifold()

## Linear kalman filter
z_filtered = np.empty(np.shape(ss.z))
print('Start filtering')

unscentedKF = AdditiveUnscentedKalmanFilter(transition_functions=ss.A_fn, observation_functions=ss.C_fn,
                                                                transition_covariance=ss.Q, observation_covariance=ss.R,
                                                                initial_state_mean=np.zeros(
                                                                    (ss.dim_z,), dtype='float32'),
                                                                initial_state_covariance=0.01 * np.eye(ss.dim_z, dtype='float32'))

unscentedKF_withInput = AdditiveUnscentedKalmanFilter_withInput(transition_functions=ss.A_fn, observation_functions=ss.C_fn,
                                                   transition_covariance=ss.Q, observation_covariance=ss.R,
                                                   initial_state_mean=np.zeros((ss.dim_z,), dtype='float32'),
                                                   initial_state_covariance=0.01 * np.eye(ss.dim_z, dtype='float32'))
state_offset = ss.B @ np.transpose(ss.u)
z_filtered, _ = unscentedKF.filter(ss.y)
z_filtered_withInput, _ = unscentedKF_withInput.filter(ss.y, state_offset=state_offset.T)


## Plot the true states and filtered ones
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(ss.z.shape[0]), ss.z[:, 0], label='z')
ax.plot(range(ss.z.shape[0]), z_filtered[:, 0], label='filtered z without input')
ax.plot(range(ss.z.shape[0]), z_filtered_withInput[:, 0], label='filtered z with input')
ax.legend()
plt.pause(0.05)
plt.show()
