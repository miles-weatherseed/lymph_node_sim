# We need to bring in values of all parameters from a txt file. Don't let user do this yet.
# Then need to work out amount of antigen on DC arrival in lymph node
# Also need to import probabilities table

import json
import numpy as np

with open("config/parameters.json") as json_file:
    data = json.load(json_file)

locals().update(data)

# defining parameters that need more care

TOTAL_TNUM = T_CELL_DENSITY*4.0*0.3333*3.1415926 * DEFAULT_RADIUS**3 * 1E-9
ANTIGEN_OOB_TOLERANCE = 0.1*DEFAULT_DENDNUM
POSITION_OOB_TOLERANCE = 100*DEFAULT_DENDNUM

def T_GAMMA_SCALE(v):
    return v/DEFAULT_T_VELOCITY_MEAN*4.40287799651

def NUM_TIMESTEPS_HOUR(ts):
    return 60/ts

def NUM_TIMESTEPS_DAY(ts):
    return 1440/ts

def NUM_TIMESTEPS_2DAY(ts):
    return 2880/ts

# now creating our local parameters and giving them the default values or calculating them

timeStep = DEFAULT_TIMESTEP
numTimeSteps = int(NUM_TIMESTEPS_2DAY(timeStep))
numTimeStepsPerWave = numTimeSteps
numTimeMeasurements = DEFAULT_NUM_TIME_MEASUREMENTS
radius = DEFAULT_RADIUS
contactRadius = DEFAULT_CONTACT_RADIUS
tCellAgSpecFreq = DEFAULT_AGSPEC_FREQ
numTCells = int(TOTAL_TNUM * tCellAgSpecFreq)
numDCells = DEFAULT_DENDNUM
numAntigenInContactArea = DEFAULT_NUM_ANTIGEN_IN_CONTACT_AREA
numAntigenOnDC = int(numAntigenInContactArea*2000.0/7.8)
tCellActivationThreshold = DEFAULT_TCELL_ACTIVATION_THRESHOLD
cogAgInDermis = DEFAULT_COGNATE_RATIO_MEAN_AT_DERMIS
firstDCArrival = DEFAULT_FIRST_DC_ARRIVAL
DCArrivalDuration = DEFAULT_DC_ARRIVAL_DURATION
tVelocityMean = DEFAULT_T_VELOCITY_MEAN
tVelocityStDev = DEFAULT_T_VELOCITY_STDEV
tGammaShape = T_GAMMA_SHAPE
tGammaScale = T_GAMMA_SCALE(DEFAULT_T_VELOCITY_MEAN)
freePathMean = DEFAULT_T_FREE_PATH_MEAN
freePathStDev = DEFAULT_T_FREE_PATH_STDEV
numRepeats = DEFAULT_NUM_REPEATS
antigenDecayRate = DEFAULT_ANTIGEN_DECAY_RATE
cogAgOnArrival = cogAgInDermis*np.exp(-antigenDecayRate*firstDCArrival)
introRate = numDCells/DCArrivalDuration
noTimeLimit = False
animateStatus = True

# Intialize positions of DCs on a grid

occupiedPositionsArraySize = 0
contractGridReductionFactor = 2.0
status = True

while status:
    if contractGridReductionFactor < MINIMUM_CONTACT_GRID_MEMORY_REDUCTION_FACTOR:
        raise ValueError("Could not produce a discrete position grid of small enough footprint. Check parameters.")
    else:
        contractGridReductionFactor /= 2
        numPositions = int(2 * contractGridReductionFactor * radius / contactRadius + 1)
        numPositionsSq = numPositions ** 2
        occupiedPositionsArraySize = int(numPositions ** 3)
    status = occupiedPositionsArraySize * 10 * 4 > 2e9

cellSide = contactRadius / contractGridReductionFactor