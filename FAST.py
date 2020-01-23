from SALib.sample import fast_sampler
from SALib.analyze import fast
import numpy as np
from simulation import Simulation
import sys

# b3 = contact radius cubed - DEFAULT 8000
# D = number of DCs - DEFAULT 720
# F = T cell free path - DEFAULT 25
# f = T cell free path sd - DEFAULT 3
# n = T cell act threshold - DEFAULT 20
# P = first DC arrival - DEFAULT 1080 mins
# p = DC arrival duration - DEFAULT 360
# R3 = LN radius cubed - DEFAULT 125000000
# V = T-cell velocity - DEFAULT 2.26567314561
# Ad = cognate antigen mean in dermis - DEFAULT 0.2
# k = antigen off rate - DEFAULT 0.00138728
# N = antigen in contact area - DEFAULT 500

problem = {
    'num_vars': 12,
    'names': ['b3', 'D', 'F', 'f', 'n', 'P', 'p', 'R3', 'V', 'Ad', 'k', 'N'],
    'bounds': [[6000, 10000],
               [540, 900],
               [18.75, 31.25],
               [2.25, 3.75],
               [15, 25],
               [810, 1350],
               [270, 450],
               [93750000, 156250000],
               [1.69925485921, 2.83209143201],
               [0.15, 0.25],
               [0.000375, 0.000625],
               [375, 625]]
}

param_values = fast_sampler.sample(problem, 100)
Y = np.zeros(param_values.shape[0])
for i, vals in enumerate(param_values):
    X = Simulation(contactRadius = vals[0]**(1/3), numDCells=int(vals[1]), freePathMean=vals[2], freePathStDev=vals[3], tCellActivationThreshold= int(vals[4]),
                   firstDCArrival=vals[5], DCArrivalDuration=vals[6], radius=vals[7]**(1/3), tGammaShape=vals[8],
                   cogAgInDermis=vals[9], antigenDecayRate=vals[10], numAntigenInContactArea=int(vals[11]))
    Y[i] = X.simulate()

Si = fast.analyze(problem, Y)
log = open("SensitivityAnalysis.log", "a")
sys.stdout = log
print(Si)