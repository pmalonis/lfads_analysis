'''
Fits and simulates a Gaussian Linear Dynamical System, following Hinton and Gharamani
'''

import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt

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

        self.log_likelihood = [] #to be appended on EM step

    def set_params(self, A, C, Q, R, pi_0=None, V_0=None):
        '''
        Set model parameters
        A: State transition matrix
        C: Observation matrix
        Q: Process noise covariance
        R: Observtion noise covariance
        pi_0: initial state mean
        V_0: initial state covariance
        '''
        assert(A.shape == (self.d_latent, self.d_latent))
        assert(C.shape == (self.d_obs, self.d_latent))
        assert(Q.shape == (self.d_latent, self.d_latent))
        assert(R.shape == (self.d_obs, self.d_obs))

        self.A = A
        self.C = C
        self.Q = Q
        self.R = R
        self.pi_0 = pi_0
        self.V_0 = V_0

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

    def initialize(self, obs):
        '''Initialize data using factor analysis and ordinary least squares'''
        
        self.set_obs(obs)
        fa = FactorAnalysis(n_components=self.d_latent)
        fa.fit(np.concatenate(self.obs))
        R = np.diag(fa.noise_variance_)
        Q = np.eye(self.d_latent) #there should be a better way to initialize this
        C = fa.components_.T
        xt = np.concatenate([fa.transform(y[:-1,:]) for y in self.obs])
        xtp1 = np.concatenate([fa.transform(y[1:,:]) for y in self.obs])
        #A = np.linalg.lstsq(xt, xtp1)[0].T
        A = np.random.randn(self.d_latent, self.d_latent)
        A = (A - A.T)/2 #random skew symmetric matrix
        N = np.sum(y.shape[0] for y in obs)
        pi_0 = np.zeros((len(obs), self.d_latent))
        V_0 = np.array([np.eye(self.d_latent) for i in range(len(obs))])
        self.set_params(A, C, Q, R, pi_0, V_0)

    def filter(self, y, pi_0, V_0):
        '''Kalman smoothing of observations
        parameters
        y: single trial sequence of spike counts
        pi_0: initial state mean
        V_0: initial state covariance'''

        T = y.shape[0] #length of time series
        x = np.zeros((T, self.d_latent)) # will contain E(x_t|y_1,...,y_t)
        V = np.zeros((T, x.shape[1], x.shape[1])) # will contain Var(x_t|y_1,...,y_t)
        all_bel_V = np.zeros_like(V) #will store all bel_V (in case needed for smoothing)
        bel_x = pi_0
        bel_V = V_0
        all_K = np.zeros((T, self.d_latent, self.d_obs))
        for i in range(T):
            try:
                K = bel_V.dot(self.C.T).dot(inv(self.C.dot(bel_V).dot(self.C.T) + self.R))
            except np.linalg.LinAlgError:
                eps = np.eye(self.d_obs) * 1e-10
                K = bel_V.dot(self.C.T).dot(inv(self.C.dot(bel_V).dot(self.C.T) + self.R + eps))
            try:
                x[i,:] = bel_x + K.dot(y[i] - self.C.dot(bel_x))
            except:
                import pdb;pdb.set_trace()

            V[i,:,:] = bel_V - K.dot(self.C).dot(bel_V)

            all_bel_V[i,:,:] = bel_V
            #update beliefs for next sample
            if i < T - 1:
                bel_x = self.A.dot(x[i])
                bel_V = self.A.dot(V[i,:,:]).dot(self.A.T) + self.Q

        y_filtered = self.C.dot(x.T).T    
            
        return y_filtered, x, V, all_bel_V, K #return final Kalman gain for EM step
    # def smooth(self, y):
    #     '''Kalman smoothing of observations
    #     parameters
    #     y: single trial sequence of spike counts'''

        #return smoothed

    def em_step(self):
        '''Perform one step of EM algorithm'''

        #E step
        N = np.sum(y.shape[0] for y in self.obs)
        x_T = np.zeros((N,self.d_latent)) # will contain E(x_t|y_1,...,y_T) where T is length of y
        V_T = np.zeros((N, self.d_latent, self.d_latent)) # will contain Var(x_t|y_1,...,y_T) where T is length of y
        V_T_ttm1 = np.zeros((N-len(self.obs), self.d_latent, self.d_latent))
        P_t = np.zeros_like(V_T)
        P_ttm1 = np.zeros_like(V_T_ttm1)
        n = 0 #time point counter
        #loop over intervals
        for trial, y in enumerate(self.obs):
            T = y.shape[0] #length of time series
            #forward iteration
            _, x, V, bel_V, K_T = self.filter(y, self.pi_0[trial], self.V_0[trial])
            #initialize for backwards iteration
            x_T[n+T-1,:] = x[-1,:]
            V_T[n+T-1,:,:] = V[-1,:,:]
            V_T_ttm1[n+T-trial-2,:,:] = (np.eye(self.d_latent) - K_T.dot(self.C)).dot(self.A).dot(bel_V[-2,:,:])
            P_t[n+T-1,:,:] = V_T[-1,:,:] + np.outer(x_T[n+T-1], x_T[n+T-1])
            P_ttm1[n+T-trial-2,:,:] = V_T_ttm1[n+T-trial-2,:,:] + np.outer(x_T[n+T-1], x_T[n+T-2])
            #backward iteration
            for i in reversed(range(n,n+T-1)):
                J = V[i-n,:,:].dot(self.A.T).dot(inv(bel_V[i+1-n,:,:]))
                x_T[i] = x[i-n] + J.dot(x_T[i+1-n] - self.A.dot(x[i-n]))
                V_T[i] = V[i-n,:,:] + J.dot(V_T[i+1,:,:] - bel_V[i-n,:,:]).dot(J.T)
                P_t[i,:,:] = V_T[i] + np.outer(x_T[i], x_T[i])
                if i < n + T-2:
                    V_T_ttm1[i-trial,:,:] = V[i+1-n,:,:].dot(J.T) + J_tp1.dot(V_T_ttm1[i+1-trial,:,:] - self.A.dot(V[i+1-n,:,:])).dot(J.T)
                    P_ttm1[i-trial,:,:] = V_T_ttm1[i-trial,:,:] + np.outer(x_T[i+1,:], x_T[i,:])
                
                J_tp1 = np.copy(J)
            n = n + T

        #M step
        all_y = np.concatenate(self.obs)
        T = np.sum(y.shape[0] for y in self.obs)
        self.C = sum(np.outer(all_y[i], x_T[i]) for i in range(T)).dot(inv(np.sum(P_t,axis=0)))
        self.R = 1/T * sum(np.outer(all_y[i],all_y[i]) - 
                            self.C.dot(np.outer(x_T[i], all_y[i])) for i in range(T))
        t0 = np.cumsum([y.shape[0] for y in self.obs]) #start of each trial
        t0 = np.insert(t0[:-1], 0, 0)
        sum_P_t = np.sum(P_t, axis=0) - np.sum(P_t[t0,:,:], axis=0)
        self.A = np.sum(P_ttm1, axis=0).dot(inv(sum_P_t))
        self.Q = 1/(T-1) * (sum_P_t - self.A.dot(np.sum(P_ttm1, axis=0).T))
        self.pi = np.array([x_T[i] for i in t0])
        self.V_0 = np.array([V_T[i] for i in t0])

        #p = self.joint_prob(x_T, t0)

        # return p
        #ll = - np.sum(1/2*(y-self.C.dot(x_T)))
        #self.log_likelihood.append(ll)

    def joint_prob(self, x, i_x0):
        '''Get joint probability of hidden states
        x: hidden states
        i_x0: indices of x that represent the first hidden state of each
        trial in self.obs
        '''
        assert(len(self.obs) == len(i_x0))
        y = np.concatenate(self.obs)
        T = y.shape[0]
        x_t = np.delete(x, i_x0, axis=0)
        x_tm1 = np.delete(x, i_x0[1:]-1, axis=0)
        x_tm1 = x_tm1[:-1]
        V_0 = np.mean(self.V_0, axis=0) #averaging over trial initial state covariances
        p = (-np.sum(0.5*(y.T - self.C.dot(x.T)).T.dot(inv(self.R)).dot(y.T - self.C.dot(x.T)))
            - T/2 * np.log(np.linalg.det(self.R))
            - np.sum(0.5*(x_t.T - self.A.dot(x_tm1.T)).T.dot(inv(self.Q)).dot(x_t.T - self.A.dot(x_tm1.T)).T)
            - (T-1)/2 * np.log(np.linalg.det(self.Q))
            - np.sum([0.5 * (x[i_x0[i]] - self.pi_0[i]).dot(inv(V_0)).dot((x[i_x0[i]] - self.pi_0[i]).T) for i in range(len(i_x0))])
            - 0.5 * np.log(np.linalg.det(V_0)) - T*(self.d_latent + self.d_obs)/2 * np.log(2*np.pi))

        return p

def fit_dataset(dataset, nsteps, binsize=.05, d_latent=10):
    from evaluate_all_datasets import center_out_gen_counts
    counts = center_out_gen_counts(dataset, .05)
    d_obs = counts[0].shape[1]
    model = glds(d_obs, d_latent)
    model.initialize(counts[:150])
    p = []
    for i in range(nsteps):
        p.append(model.em_step())
        print("EM step %d completed"%i)

    return model

    def fit(self, n_steps, obs=None):
        pass
