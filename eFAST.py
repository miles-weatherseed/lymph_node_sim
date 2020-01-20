from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[],
               [],
               []]
}

param_values = saltelli.sample(problem, 1000)
# Y is a vector which will hold the outcome - 1 or 0 - of the simulation
Y = np.zeros([param_values.shape[0]])
for i, X in enumerate(param_values):
    Y[i] = simulate(X)
