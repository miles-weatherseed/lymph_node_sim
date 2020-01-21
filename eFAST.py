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
# R3 = LN radius cubed - DEFAULT 125000000
# V = T-cell velocity - DEFAULT 2.26567314561

problem = {
    'num_vars': 9,
    'names': ['b3', 'D', 'F', 'f', 'n', 'P', 'p', 'R3', 'V'],
    'bounds': [[4000, 16000],
               [360, 1440],
               [12.5, 50],
               [1.5, 6],
               [10, 40],
               [540, 2160],
               [180, 720],
               [62500000, 250000000],
               [1.1328, 4.533]]
}

param_values = saltelli.sample(problem, 100)
# Y is a vector which will hold the outcome - 1 or 0 - of the simulation
Y = np.zeros([param_values.shape[0]])
for i, X in enumerate(param_values):
    Y[i] = simulate(X)
