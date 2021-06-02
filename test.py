import numpy as np
import math
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.colors as mcolors
import scipy.optimize as opt
import os
import sys

# Constants
Pdmax = 0.8 # Max range sensor
dmax = 4 # Max distance
sigma = 0.7 # Sensor spread (standard deviation)
delta = 0.5 # Constant displacement

exercise = 2

def bivariate_gaussian(w, h, mu, Sigma, X, Y):
    COV_1 = np.linalg.inv(Sigma) # Compute the inverse Sigma^-1
    p = np.zeros(w*h) # probability map
    d = np.array([X.ravel()-mu[0],Y.ravel()-mu[1]]) # distances (x- \mu)
    # Compute probability of each point
    for i in range(0,w*h):
        p[i] = np.exp(- 0.5 * d[:,i].dot(COV_1).dot(d[:,i].transpose()))
    p = p/np.sum(p) # normalize to sum to 1
    P = p.reshape((w, h)) # rank 2 square matrix 
    return P

def sensor_pnd(d, dmax, Pdmax, sigma):
    return 1 - Pdmax*math.exp(-sigma*(d/dmax)**2)

def distance(xi, xj, i, j):
    return math.sqrt((i - xi)**2 + (j - xj)**2)

class Agent:
    def __init__(self, i, bk, env, state=np.array([0.,0.])):
        self.id = i  # Create an instance variable
        self.bk = bk
        self.env = env

        ## Agent params
        self.x = state
        self.track = self.x.T # Stores all positions. Aux variable for visualization of the agent path
        self.height_plot = 0.1
        
        # add extra agents params
        
        ## Sensor params
        
    # get id
    def get_id(self):
        return self.id

    # compute discrete forward states 1 step ahead
    def forward(self):
        return np.random.permutation(
            [[a+self.x[0], b+self.x[1]] for [a, b] in self.env.mat if a+self.x[0] >= 0 
             and a+self.x[0] < 40 and b+self.x[1] >= 0 and b+self.x[1] < 40])
        
    # computes utility of forward state
#     def nodetect_observation(self, x):
#         pnd = []
#         return pnd
    
    # computes utility of states
    def utility(self, fs):
        # compute cost funtion J of all potential forward states (brute force method)
        J = []
        for state in fs:
            utility = 0
            for x in range(self.env.width):
                for y in range(self.env.height):
                    d = distance(state[0], state[1], x, y)
                    utility += sensor_pnd(d, dmax, Pdmax, sigma)*self.bk[x, y]
            J.append(utility)
        return J
    
    def next_best_state(self):
        fs = self.forward()
        J = self.utility(fs)
        return fs[J.index(min(J))]
      
    # simulate agent next state
    def next(self, state):
        self.x = state
        self.track = np.vstack((self.track, self.x))

    # update belief with observation at state self.x
    def update_belief(self):
        for x in range(self.env.width):
            for y in range(self.env.height):
                d = distance(self.x[0], self.x[1], x, y)
                self.bk[x, y] = sensor_pnd(d, dmax, Pdmax, sigma)*self.bk[x, y]
        self.bk = self.bk/np.sum(self.bk)
      
    def plot(self, ax):
        # Reshape belief for plotting
        # bkx = self.bk.reshape((self.env.width, self.env.height))
        ax.cla() # clear axis plot
        
        # plot contour of the P(\tau) -> self.bk
        cset = ax.contourf(self.env.Y, self.env.X, self.bk, zdir='z', offset=-0.005, cmap=cm.viridis)

        # plot agent trajectory, self.track
        ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-', linewidth=2);
        ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]], [self.height_plot, 0], 'ko-', linewidth=2);

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('P')

        # Adjust the limits, ticks and view angle
        ax.set_zlim(0, 0.12*self.get_id())
        ax.view_init(27, -21)
    
    
class Environment:
    # Use it to store parameters and for ploting functions
    def __init__(self, common_bk = np.array([]), map_size = np.array([40, 40]), sigma = np.array([[40,0],[0,60]])):
        self.width = map_size[0]
        self.height = map_size[1]
        self.X, self.Y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.mu = np.array([self.width/2., self.height/2.]) # center point
        self.sigma = sigma # Bimodal covariance with no dependence.
        self.mat = [[0, 0], [-delta, 0], [-delta, delta], [0, delta], [delta, delta], [delta, 0], [delta, -delta], [0, -delta], [-delta, -delta]]
        self.common_bk = common_bk
        
print('-------------------------------------------------\n');
print('> M-Agents search 2D (1 n-step ahead no comm)\n')

nagents = 2
# Create environment
env = Environment()

# Create bivariate gaussian distribution
belief = bivariate_gaussian(env.width, env.height, env.mu, env.sigma, env.X, env.Y)
env.common_bk = belief

# Create agents
agents = []
a0 = Agent(0, belief, env, np.array([5,5]))
agents.append(a0)
a1 = Agent(1, belief, env, np.array([30,30]))
agents.append(a1)
a2 = Agent(2, belief, env, np.array([0, 15]))
agents.append(a2)

# Global plot for animations
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.ion()

# Start algorithm
ite = 0  # iteration count
nite = 200  # number of iterations
found = 0  # target found

## exercise 1
# if exercise == 1:
#     while not found and ite < nite:

#         for a in agents:
#             a.next(a.next_best_state())
#             a.update_belief()
#             a.plot(ax)

#         # plot
#         plt.draw()
#         plt.pause(0.01)  # animation

#         # iteration count
#         ite += 1

## exercise 2
if exercise == 2:
    while not found and ite < nite:
        for a in agents:
            a.bk = env.common_bk
            a.next(a.next_best_state())
            a.update_belief()
            env.common_bk = a.bk

        for a in agents:
            a.plot(ax)

        # plot
        plt.draw()
        plt.pause(0.01)  # animation

        # iteration count
        ite += 1
