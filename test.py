import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as opt
import copy
import warnings

# Disable warnings
warnings.filterwarnings("ignore")

# Exercise variable, either 1, 2 or 3
exercise = 3


# ========================    IMPORTANT FUNCTIONS     ========================

# Returns a bivariate guassian distribution
def bivariate_gaussian(w, h, mu, Sigma, X, Y):
    COV_1 = np.linalg.inv(Sigma)                    # Compute the inverse Sigma^-1
    p = np.zeros(w*h)                               # Probability map
    d = np.array([X.ravel()-mu[0],Y.ravel()-mu[1]]) # Distances (x- \mu)
    # Compute probability of each point
    for i in range(0,w*h):
        p[i] = np.exp(- 0.5 * d[:,i].dot(COV_1).dot(d[:,i].transpose()))
    p = p/np.sum(p)                                 # Normalize to sum to 1
    P = p.reshape((w, h))                           # Rank 2 square matrix 
    return P

# Returns the probability of non-detecting given the distance to the target
def sensor_pnd(d, dmax, Pdmax, sigma):
    return 1 - Pdmax*math.exp(-sigma*(d/dmax)**2)

# Returns the distance between (xi, xj) and (i,j)
def distance(xi, xj, i, j):
    return math.sqrt((i - xi)**2 + (j - xj)**2)

# Computes the multi utility
def multi_utility(uk, agents, N, bk):
    # Reshape the turning rates back to an N by M (len(agents)) matrix
    uk = uk.reshape(N, len(agents))
    value = 1
    for i in range(len(agents)):
        a = copy.deepcopy(agents[i])
        for j in range(N):
            a.next(uk[j, i])
            for x in range(env.width):
                for y in range(env.height):
                    d = distance(a.x[0], a.x[1], x, y)
                    value += sensor_pnd(d, dmax, Pdmax, sigma)*bk[x, y]
    return value


# ========================    CLASS DEFINITIONS     ========================

class Agent:
    # Constructor
    def __init__(self, bk, env, state=np.array([0.,0.])):
        self.bk = bk            # The belief of the agent
        self.env = env          # The environment
        self.x = state          # The state of the agent (tuple of floats)
        self.track = self.x.T   # Stores all positions. Aux variable for visualization of the agent path
        self.height_plot = 0.1

    # Computes discrete forward states 1 step ahead
    def forward(self):
        return np.random.permutation(
            [[a+self.x[0], b+self.x[1]] for [a, b] in self.env.mat if a+self.x[0] >= 0 
             and a+self.x[0] < 40 and b+self.x[1] >= 0 and b+self.x[1] < 40])
    
    # Computes utility of states
    def utility(self, fs):
        J = []
        for state in fs:
            utility = 0
            for x in range(self.env.width):
                for y in range(self.env.height):
                    d = distance(state[0], state[1], x, y)
                    utility += sensor_pnd(d, dmax, Pdmax, sigma)*self.bk[x, y]
            J.append(utility)
        return J
    
    # Returns the next best state
    def next_best_state(self):
        fs = self.forward()
        J = self.utility(fs)
        return fs[J.index(min(J))]
      
    # Simulates agent next state
    def next(self, state):
        self.x = state
        self.track = np.vstack((self.track, self.x))

    # Updates belief with observation at state self.x
    def update_belief(self):
        for x in range(self.env.width):
            for y in range(self.env.height):
                d = distance(self.x[0], self.x[1], x, y)
                self.bk[x, y] = sensor_pnd(d, dmax, Pdmax, sigma)*self.bk[x, y]
        self.bk = self.bk/np.sum(self.bk)
      
    # Plots agent trajectory, self.track
    def plot(self, ax):
        ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-', linewidth=2);
        ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]], [self.height_plot, 0], 'ko-', linewidth=2);


class AgentContinuous():
    # Constructor
    def __init__(self, state=np.array([0.,0.,0.])):
        self.V = 2                  # Velocity of the agent
        self.dt = 1                 # Interval for euler integration (continuous case)
        self.max_turn_change = 0.2  # Max angle turn (action bounds)
        self.x = state
        self.track = self.x.T       # Stores all positions
        self.height_plot = 0.1
        
    # Sets next state
    def next(self, vk):
        # Singular case u = 0 -> the integral changes
        if vk == 0:
            self.x[0] = self.x[0] + self.dt * self.V * np.cos(self.x[2])
            self.x[1] = self.x[1] + self.dt * self.V * np.sin(self.x[2])
            self.x[2]= self.x[2]
        else:
            desp = self.V / vk
            if np.isinf(desp) or np.isnan(desp):
                print('forwardstates:V/u -> error')
            self.x[0] = self.x[0] + desp * (np.sin(self.x[2] + vk * self.dt) - np.sin(self.x[2]))
            self.x[1] = self.x[1] + desp * (-np.cos(self.x[2] + vk * self.dt) + np.cos(self.x[2]))
            self.x[2] = self.x[2] + vk * self.dt

        self.track = np.vstack((self.track, self.x))

    # Plots the trajectory of the agent
    def plot(self, ax):
        ax.plot(self.track[:, 0], self.track[:, 1], np.ones(self.track.shape[0]) * self.height_plot, 'r-', linewidth=2)
        ax.plot([self.track[-1, 0], self.track[-1, 0]], [self.track[-1, 1], self.track[-1, 1]], [self.height_plot, 0], 'ko-', linewidth=2)


class Optimizer:
    # Constructor
    def __init__(self):
        self.method = 'trust-constr'    # Optimization method
        self.jac = "2-point"            # Automatic Jacobian finite differenciation
        self.hess =  opt.SR1()          # opt.BFGS() method for the hessian computation
        self.ul = np.pi / 4             # Max turn constraint for our problem (action limits). How much the vehicle can turn

    # fun - function to optimize, 
    # x0 - variables initialization (velocities at each instant), 
    # agents - structure with the information of each agent
    # N - steps ahead
    # bk - current belief of the environment. Probability of finding the target.
    def optimize(self, fun, x0, agents, N, bk):
        n = x0.shape[0]
        bounds = opt.Bounds(np.ones(n) * (-self.ul), np.ones(n) * self.ul) 
        res = opt.minimize(fun, x0, args=(agents, N, bk), method=self.method, jac=self.jac, hess=self.hess, bounds=bounds)
        return res
    
class Environment:
    # Constructor 
    def __init__(self, common_bk = np.array([]), map_size = np.array([40, 40]), sigma = np.array([[40,0],[0,60]])):
        self.width = map_size[0]
        self.height = map_size[1]
        self.X, self.Y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.mu = np.array([self.width/2., self.height/2.])     # Center point
        self.sigma = sigma                                      # Bimodal covariance with no dependence.
        self.mat = [[0, 0], [-delta, 0], [-delta, delta], [0, delta], [delta, delta], [delta, 0], [delta, -delta], [0, -delta], [-delta, -delta]]
        self.common_bk = common_bk

    # Plots belief
    def plot(self, ax, belief):
        ax.cla() # Clear axis plot
        ax.contourf(self.Y, self.X, belief, zdir='z', offset=-0.005, cmap=cm.viridis)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('P')
        ax.set_zlim(0, 0.12)
        ax.view_init(27, -21)

    # Updates common belief of continuous agents
    def update_common_belief(self, state):
        for x in range(self.width):
            for y in range(self.height):
                d = distance(state[0], state[1], x, y)
                self.common_bk[x, y] = sensor_pnd(d, dmax, Pdmax, sigma)*self.common_bk[x, y]
        self.common_bk = self.common_bk/np.sum(self.common_bk)
        


## ========================    CONSTANTS/SETTINGS     ========================

Pdmax = 0.8     # Max range sensor
dmax = 4        # Max distance
sigma = 0.7     # Sensor spread (standard deviation)
delta = 0.5     # Constant displacement

ite = 0         # Iteration count
nite = 50       # Number of iterations
found = 0       # Target found

# The number of agents
nagents = 2

# The number of steps ahead (only relevant for exercise 3)
N = 3

# ========================    INITIALIZATIONS FOR THE ALGORITHM     ========================
# Create environment
env = Environment()

# Create bivariate gaussian distribution
belief = bivariate_gaussian(env.width, env.height, env.mu, env.sigma, env.X, env.Y)
env.common_bk = belief

# Create agents
agents = []
positions = [np.array([5,5]),np.array([30,30])]
for i in range(nagents):
    agents.append(Agent(belief, env, positions[i]))

# Create continuous agents
agents_c = []
states = [np.array([0., 0., 0.]),np.array([20., 0., 0.])]
for i in range(nagents):
    agents_c.append(AgentContinuous(states[i]))


# ========================    START ALGORITHM     ========================
print('-------------------------------------------------\n')


## Exercise 1
if exercise == 1:
    print('> M-Agents search 2D (1-step ahead, no common belief)\n')

    # Prepare plot figures
    figs = []
    axs = []
    plt.ion()
    for i in range(nagents):
        figs.append(plt.figure())
        axs.append(figs[i].gca(projection='3d'))

    # Compute the next state and update belief accordingly
    while not found and ite < nite:
        for a in agents:
            a.next(a.next_best_state())
            a.update_belief()

        # Plot
        for i in range(nagents):
            env.plot(axs[i], agents[i].bk)
            agents[i].plot(axs[i])
        plt.draw()
        plt.pause(0.01)  # Animation

        # Iteration count
        ite += 1


## Exercise 2
if exercise == 2:
    print('> M-Agents search 2D (1-step ahead, with common belief)\n')

    # Prepare plot figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.ion()

    # Compute the next state and update belief accordingly
    while not found and ite < nite:
        for a in agents:
            a.bk = env.common_bk
            a.next(a.next_best_state())
            a.update_belief()
            env.common_bk = a.bk

        # Plot
        env.plot(ax, env.common_bk)
        for a in agents:
            a.plot(ax)
        plt.draw()
        plt.pause(0.01)  # Animation

        # Iteration count
        ite += 1


## Exercise 3
if exercise == 3:
    print('> M-Agents search 2D (N-steps ahead, with common belief)\n')

    # Prepare plot figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.ion()
    opti = Optimizer()
    
    # Compute the next state and update belief accordingly
    while not found and ite < nite:
        x0 = np.full((N, nagents), 0.001)
        turnrates = opti.optimize(multi_utility, x0.flatten(), copy.deepcopy(agents_c), N, copy.deepcopy(env.common_bk)).x.reshape(N, nagents)
        for i in range(nagents):
            agents_c[i].next(turnrates[0, i])
            env.update_common_belief(agents_c[i].x)

        # Plot
        env.plot(ax, env.common_bk)
        for a in agents_c:
            a.plot(ax)
        plt.draw()
        plt.pause(0.01)  # Animation

        # Iteration count
        ite += 1
