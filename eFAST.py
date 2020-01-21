from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

# b3 = contact radius cubed - DEFAULT 8000
# D = number of DCs - DEFAULT 720
# F = T cell free path - DEFAULT 25
# f = T cell free path sd - DEFAULT 3
# n = T cell act threshold - DEFAULT 20
# P = first DC arrival - DEFAULT 1080 mins
# p = DC arrival duration - DEFAULT 360
#

problem = {
    'num_vars': 9,
    'names': ['b3', 'D', 'F', 'f', 'n', 'P', 'p', 'R3', 'V'],
    'bounds': [[5, 40],
               [200, 1000],
               [15, 40],
               [],
               [],
               [],
               [],
               [],
               []]
}

param_values = saltelli.sample(problem, 100)
# Y is a vector which will hold the outcome - 1 or 0 - of the simulation
Y = np.zeros([param_values.shape[0]])
for i, X in enumerate(param_values):
    Y[i] = simulate(X)
