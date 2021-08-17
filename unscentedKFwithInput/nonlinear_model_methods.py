import sys, os, time
import matplotlib.pyplot as plt
import numpy as np
import scipy

class nonlinear_state_space_model(object):
    '''
    This class simulates data using nonlinear state space model

    z_t = A_fn( z_{t-1} ) + B*u_t + q_t,        cov(q_t) = Q
    y_t = C_fn( a_t ) + r_t,                    cov(r_t) = R

    '''
    def __init__(self, settings):
        self.dim_u = settings['dim_u']
        self.dim_z = settings['dim_z']
        self.dim_y = settings['dim_y']
        self.num_samples_trial = settings['num_samples_trial']
        noise_Q = settings['noise_Q']
        noise_R = settings['noise_R']

        # Print the main features of the data that will be generated
        print('#### Model Details ####')
        print('Dimensions: input(u): {}, state(z): {}, observation: {}'.format(self.dim_u, self.dim_z, self.dim_y))
        print('Trial length: {}'.format(self.num_samples_trial))
        print('Noise levels: Q: {}, R:{}'.format(noise_Q, noise_R))

        print('---> Initializing A_fn')
        self.eigens = np.array([0.99])
        self.A_fn, _ = self._generate_A_function(self.eigens)

        print('---> Initializing B')
        self.B = np.ones((self.dim_z, self.dim_u))
        print('B: {}'.format(self.B))

        print('---> Initializing C_fn')
        self.C_fn = lambda x: self._generate_C_function(x, max_a=300)

        # Noise Covariances
        self.range_var_Q = (noise_Q, noise_Q)
        self.range_var_R = (noise_R, noise_R)
        self.Q = self._generate_Q_matrix(self.range_var_Q)
        self.R = self._generate_R_matrix(self.range_var_R)
        return

    def _generate_A_function(self, eigens):
        A = np.diag(eigens)
        W, V = np.linalg.eig(A)
        W_real, v_real =  scipy.linalg.cdf2rdf(W, V) # transforms into real block diagonal form
        A = W_real
        self.A = A
        print('A: {}'.format(self.A))
        A_fn = lambda x: A @ x
        return A_fn, A

    def _generate_Q_matrix(self,range_var_Q):
        Q_diag = (range_var_Q[1] - range_var_Q[0]) * np.random.random_sample((self.dim_z)) + range_var_Q[0]
        Q = np.diag(Q_diag)
        return Q

    def _generate_R_matrix(self,range_var_R):
        R_diag = (range_var_R[1] - range_var_R[0]) * np.random.random_sample((self.dim_y)) + range_var_R[0]
        R = np.diag(R_diag)
        return R
    
    def _generate_C_function(self,a,max_a):
        a_this = a[0]
        m = np.asarray([np.cos(2 * (np.pi/max_a) * a_this), np.sin(4 * (np.pi/max_a) * a_this),
                        np.sin(2 * (np.pi/max_a) * a_this)], dtype="float32")
        return m
 
    '''
    Following are the model functions
    '''
    def generate_z(self):
        z = np.empty((self.num_samples_trial, self.dim_z))
        z[:] = np.nan

        init_z = np.zeros((self.dim_z))
        z[0,:] = init_z

        for t in range(1,self.num_samples_trial):
            z[t,:] =  self.A_fn( z[t-1,:] ) + self.B @ self.u[t,:] + np.random.multivariate_normal(np.zeros(self.dim_z) , self.Q)

        self.z = z
        print('States (z) are initilized as a vector by shape {}'.format( self.z.shape ))
        return

    def generate_y(self):
        z = self.z
        length_trial, _ = z.shape
        y = np.empty((length_trial, self.dim_y))

        for t in range(0,length_trial):
            y[t, :] = self.C_fn(z[t, :]) + np.random.multivariate_normal(np.zeros(self.dim_y), self.R)

        self.y = y
        print('Observations (y) are initilized as a vector by shape {}\n'.format(self.y.shape))
        return y

    def generate_u(self):
        self.u = 10*np.random.rand(self.num_samples_trial, self.dim_u)
        return

    '''
    Following are the plot functions
    '''
    def plot_u(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(self.num_samples_trial), self.u[:, :])
        plt.suptitle('States (z)')
        plt.pause(0.05)
        plt.show()
        return

    def plot_z(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(self.num_samples_trial), self.z[:, :])
        plt.suptitle('States (z)')
        plt.pause(0.05)
        plt.show()
        return
        
    def plot_y(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(self.num_samples_trial),self.y[:,:])
        plt.suptitle('Observation (y)')
        plt.pause(0.05)
        plt.show()
        return

    def plot_y_manifold(self):
        y = self.y
        length_trial,_ = y.shape
 
        #colors = cm.rainbow( np.linspace(0, 1, settings_manif['num_samples_trial'] ) )
        color_map = plt.cm.get_cmap('viridis')
        fig_3d = plt.figure()
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        # plot the main manifold
        self.plot_main_manifold(fig_3d,ax_3d)

        color_index = range(length_trial)
        a_m = ax_3d.scatter(y[:,0], y[:,1], y[:,2], c=color_index, vmin=0, vmax=length_trial, s=35, cmap=color_map)

        ax_3d.set_xlabel('dim 1')
        ax_3d.set_ylabel('dim 2')
        ax_3d.set_zlabel('dim 3')
        fig_3d.colorbar(a_m)
        plt.show()    
        return 

    def plot_main_manifold(self,fig,ax):
        n = 2000 # num points
        z = np.random.uniform(0, 1, n)
        z = np.expand_dims(z,axis=1)
        m = np.empty(shape=(n, 3), dtype='float32')
        c = lambda z: np.asarray( [np.cos(2*np.pi*z[0]), np.sin(4*np.pi*z[0]), np.sin(2*np.pi*z[0])], dtype="float32")     

        for i in range(n):
            m[i,:] = c(z[i,:])
        ax.scatter(m[:,0],m[:,1],m[:,2],c='#C0C0C0',alpha=0.1)       
        return
