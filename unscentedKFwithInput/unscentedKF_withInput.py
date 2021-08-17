import os, sys
import numpy as np 
from pykalman import AdditiveUnscentedKalmanFilter
from pykalman.unscented import *
from pykalman.utils import array1d, array2d, check_random_state, get_params, preprocess_arguments, check_random_state
from pykalman.standard import _last_dims, _determine_dimensionality, _arg_or_default


def additive_unscented_filter_withInput(mu_0, sigma_0, f, g, Q, R, Z, state_offset):
    '''Apply the Unscented Kalman Filter with additive noise

        Parameters
        ----------
        mu_0 : [n_dim_state] array
            mean of initial state distribution
        sigma_0 : [n_dim_state, n_dim_state] array
            covariance of initial state distribution
        f : function or [T-1] array of functions
            state transition function(s). Takes in an the current state and outputs
            the next.
        g : function or [T] array of functions
            observation function(s). Takes in the current state and outputs the
            current observation.
        Q : [n_dim_state, n_dim_state] array
            transition covariance matrix
        R : [n_dim_state, n_dim_state] array
            observation covariance matrix

        Returns
        -------
        mu_filt : [T, n_dim_state] array
            mu_filt[t] = mean of state at time t given observations from times [0,
            t]
        sigma_filt : [T, n_dim_state, n_dim_state] array
            sigma_filt[t] = covariance of state at time t given observations from
            times [0, t]
        '''
    # extract size of key components
    T = Z.shape[0]
    n_dim_state = Q.shape[-1]
    n_dim_obs = R.shape[-1]

    # construct container for results
    mu_filt = np.zeros((T, n_dim_state))
    sigma_filt = np.zeros((T, n_dim_state, n_dim_state))

    for t in range(T):
        # Calculate sigma points for P(x_{t-1} | z_{0:t-1})
        if t == 0:
            mu, sigma = mu_0, sigma_0
        else:
            mu, sigma = mu_filt[t - 1], sigma_filt[t - 1]
        points_state = moments2points(Moments(mu, sigma))

        # Calculate E[x_t | z_{0:t-1}], Var(x_t | z_{0:t-1})
        if t == 0:
            points_pred = points_state
            moments_pred = points2moments(points_pred)
        else:
            transition_function = _last_dims(f, t - 1, ndims=1)[0]
            (_, moments_pred) = (
                unscented_filter_predict(
                    transition_function, points_state, sigma_transition=Q
                )
            )
            if state_offset is not None:
                (cur_moment_mean, cur_moment_cov) = moments_pred
                moments_pred = Moments(cur_moment_mean + state_offset[t, :], cur_moment_cov)
            points_pred = moments2points(moments_pred)

        # Calculate E[x_t | z_{0:t}], Var(x_t | z_{0:t})
        observation_function = _last_dims(g, t, ndims=1)[0]
        mu_filt[t], sigma_filt[t] = (
            unscented_filter_correct(
                observation_function, moments_pred, points_pred,
                Z[t], sigma_observation=R
            )
        )

    return (mu_filt, sigma_filt)

class AdditiveUnscentedKalmanFilter_withInput(AdditiveUnscentedKalmanFilter):

    ## For now only additive unscented filtering is modified
    def filter(self, Z, state_offset=None):
        '''Run Unscented Kalman Filter

        Parameters
        ----------
        Z : [n_timesteps, n_dim_state] array
            Z[t] = observation at time t.  If Z is a masked array and any of
            Z[t]'s elements are masked, the observation is assumed missing and
            ignored.

        Returns
        -------
        filtered_state_means : [n_timesteps, n_dim_state] array
            filtered_state_means[t] = mean of state distribution at time t given
            observations from times [0, t]
        filtered_state_covariances : [n_timesteps, n_dim_state, n_dim_state] array
            filtered_state_covariances[t] = covariance of state distribution at
            time t given observations from times [0, t]
        '''
        Z = self._parse_observations(Z)

        (transition_functions, observation_functions,
            transition_covariance, observation_covariance,
            initial_state_mean, initial_state_covariance) = (
            self._initialize_parameters()
        )

        (filtered_state_means, filtered_state_covariances) = (
            additive_unscented_filter_withInput(
                initial_state_mean, initial_state_covariance,
                transition_functions, observation_functions,
                transition_covariance, observation_covariance,
                Z, state_offset
            )
        )

        return (filtered_state_means, filtered_state_covariances)

    
