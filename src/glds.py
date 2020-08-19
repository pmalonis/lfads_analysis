'''
Fits and simulates a Gaussian Linear Dynamical System, following Hinton and Gharamani
'''

import numpy as np
from numpy.random import multivariate_normal
from sklearn.decomposition import FactorAnalysis

class glds(object):
    def __init__(self, d_obs, d_latent):
        self.d_obs = d_obs
        self.d_latent = d_latent
        
        #model parameters
        self.A = None
        self.C = None
        self.Q = None
        self.R = None

        #observations
        self.y = None

    def set_params(self, A, C, Q, R):
        '''
        Set model parameters
        A: State transition matrix
        C: Observation matrix
        Q: Process noise covariance
        R: Observtion noise covariance
        '''
        assert(A.shape == (self.d_latent, self.d_latent))
        assert(C.shape == (self.d_obs, self.d_latent))
        assert(Q.shape == (self.d_latent, self.d_latent))
        assert(R.shape == (self.d_obs, self.d_obs))

        self.A = A
        self.C = C
        self.Q = Q
        self.R = R

    def simulate(self, init, n_steps):
        '''
        Simulate model, from initial state
        init: initial state vector

        returns:
        x: States, shape is (d_latent, n_steps)
        y: Observations, shape is (d_obs, n_steps)
        '''

        if any([param is None for param in (self.A, self.C, self.Q, self.R)]):
            raise Exception('Must specify or initialize model parameters')
        
        x = np.zeros((self.d_latent, n_steps))
        x[:,0] = init
        y = np.zeros((self.d_obs, n_steps))
        for i in range(1, n_steps):
            w = multivariate_normal(np.zeros(self.d_latent), self.Q)
            x[:,i] = self.A.dot(x[:,i-1]) + w
        
        v = multivariate_normal(np.zeros(self.d_obs), self.R, size=n_steps).T
        y = (self.C.dot(x) + v).T
        x = x.T

        return x, y

    def set_obs(self, obs):
        '''
        Add observations to model. If data is continuous observation, then pass as
        2D array of shape (n_obs, n_neurons). If data is split into trials, then pass as
        list of length n_trials or 3D array with the size of dimension 0 equal to n_trials
        '''
        if type(obs) == list:
            self.obs = obs
        elif type(obs) == np.ndarray:
            if obs.ndim == 2:
                self.obs = np.array([obs])
            elif obs.ndim == 3:
                self.obs = obs
            else:
                raise ValueError('Observations must be a 2d array, 3d array, or list of 2d arrays')
        else:
            raise ValueError('Observations must be a 2d array, 3d array, or list of 2d arrays')

    def initialize(self, obs=None):
        '''Initialize data using factor analysis and ordinary least squares'''
        if obs is not None:
            self.set_obs(obs)

        fa = FactorAnalysis(n_components=self.d_latent)
        fa.fit(np.concatenate(self.obs))
        R = np.diag(fa.noise_variance_)
        Q = np.eye(self.d_latent) #there should be a better way to initialize this
        C = fa.components_.T
        xt = np.concatenate([fa.transform(y[:-1,:]) for y in self.obs])
        xtp1 = np.concatenate([fa.transform(y[1:,:]) for y in self.obs])
        A = np.linalg.lstsq(xt, xtp1)[0]

        self.set_params(A, C, Q, R)

    def em_step(self):
        pass

    def fit(self, n_steps, obs=None):
        pass
