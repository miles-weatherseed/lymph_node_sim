from classesAndFunctions import *
import numpy as np
import json
import hypergeometric
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

# IMPORT STAGE

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

class ProbabilitiesTable:
    def __init__(self, numAntigenOnDC, numAntigenInContactArea, tCellActivationThreshold):
        """
        When we initiate the class, check if a table exists with the right parameters. If not, make one
        """
        with open("config/ProbabilitiesTable.dat") as datFile:
            if (int(datFile.readline()[18:24]) == numAntigenOnDC) and (
                    int(datFile.readline()[28:31]) == numAntigenInContactArea) \
                    and (int(datFile.readline()[27:39]) == tCellActivationThreshold):
                pass
            else:
                generateLookupTable(numAntigenOnDC, numAntigenInContactArea, tCellActivationThreshold)
        self.ProbabilitiesTable = np.loadtxt("config/ProbabilitiesTable.dat", delimiter="\t", skiprows=3)

    def getProbability(self, cogAgRatio):
        return self.ProbabilitiesTable[
            np.where(self.ProbabilitiesTable[:, 0] == round((1E-5) * int(cogAgRatio / (1E-5)), 5))[0][
                0]][1]

class Simulation(dCell, tCell, ProbabilitiesTable):
    def __init__(self, timeStep=DEFAULT_TIMESTEP, numTimeSteps = int(NUM_TIMESTEPS_2DAY(timeStep)),
                 numTimeMeasurements = DEFAULT_NUM_TIME_MEASUREMENTS, radius = DEFAULT_RADIUS,
                 contactRadius = DEFAULT_CONTACT_RADIUS, tCellAgSpecFreq = DEFAULT_AGSPEC_FREQ,
                 numTCells = int(TOTAL_TNUM * tCellAgSpecFreq), numDCells = DEFAULT_DENDNUM,
                 numAntigenInContactArea = DEFAULT_NUM_ANTIGEN_IN_CONTACT_AREA,
                 numAntigenOnDC = int(numAntigenInContactArea*2000.0/7.8),
                 tCellActivationThreshold = DEFAULT_TCELL_ACTIVATION_THRESHOLD,
                 cogAgInDermis = DEFAULT_COGNATE_RATIO_MEAN_AT_DERMIS, firstDCArrival = DEFAULT_FIRST_DC_ARRIVAL,
                 DCArrivalDuration = DEFAULT_DC_ARRIVAL_DURATION, tVelocityMean = DEFAULT_T_VELOCITY_MEAN,
                 tVelocityStDev = DEFAULT_T_VELOCITY_STDEV, tGammaShape = T_GAMMA_SHAPE,
                 tGammaScale = T_GAMMA_SCALE(DEFAULT_T_VELOCITY_MEAN), freePathMean = DEFAULT_T_FREE_PATH_MEAN,
                 freePathStDev = DEFAULT_T_FREE_PATH_STDEV, numRepeats = DEFAULT_NUM_REPEATS,
                 antigenDecayRate = DEFAULT_ANTIGEN_DECAY_RATE,
                 cogAgOnArrival = cogAgInDermis*np.exp(-antigenDecayRate*firstDCArrival),
                 introRate = numDCells/DCArrivalDuration, noTimeLimit = False, animateStatus = True,
                 successfulActivations = 0, occupiedPositionsArraySize=occupiedPositionsArraySize, cellSide=cellSide,
                 numPositions=numPositions, numPositionsSq=numPositionsSq):

        self.timeStep = timeStep
        self.numTimeSteps = numTimeSteps
        self.numTimeStepsPerWave = numTimeStepsPerWave
        self.numTimeMeasurements = numTimeMeasurements
        self.radius = radius
        self.contactRadius = contactRadius
        self.tCellAgSpecFreq = tCellAgSpecFreq
        self.numTCells = numTCells
        self.numDCells = numDCells
        self.numAntigenInContactArea = numAntigenInContactArea
        self.numAntigenOnDC = numAntigenOnDC
        self.tCellActivationThreshold = tCellActivationThreshold
        self.cogAgInDermis = cogAgInDermis
        self.firstDCArrival = firstDCArrival
        self.DCArrivalDuration = DCArrivalDuration
        self.tVelocityMean = tVelocityMean
        self.tVelocityStDev = tVelocityStDev
        self.tGammaShape = tGammaShape
        self.tGammaScale = tGammaScale
        self.freePathMean = freePathMean
        self.freePathStDev = freePathStDev
        self.numRepeats = numRepeats
        self.antigenDecayRate = antigenDecayRate
        self.cogAgOnArrival = cogAgOnArrival
        self.introRate = introRate
        self.noTimeLimit = noTimeLimit
        self.animateStatus = animateStatus
        self.successfulActivations = successfulActivations
        self.occupiedPositionsArraySize = occupiedPositionsArraySize
        self.cellSide = cellSide
        self.numPositions = numPositions
        self.numPositionsSq = numPositionsSq

    def simulate(self):
        ProbTable = ProbabilitiesTable(self.numAntigenOnDC, self.numAntigenInContactArea, self.tCellActivationThreshold)
        for rep in range(self.numRepeats):

            presentSuccess = 0
            occupiedPositions = list(np.zeros(
                self.occupiedPositionsArraySize))  # this is a 1d representation of 3d space showing where the cells we've already put are
            cellMovementOrder = np.arange(self.numTCells)
            freePathRemaining = np.empty(self.numTCells)

            # create list of DCs

            dCellList = []

            # create list of T cells

            tCellList = []
            for i in range(self.numTCells):
                tCellList.append(tCell(-1 * np.ones(3), np.zeros(3)))

            # time to get the t cells in there, away from the DCs

            for i in range(self.numTCells):
                PLACE_T_ON_GRID(i, tCellList, dCellList, occupiedPositions, freePathRemaining)

            # finally initializing a few more data holding variables

            this_numInteractions = 0
            this_numUniqueInteractions = 0
            this_numActivated = 0

            # let's make some containers to hold the positions of our T Cells at each time step

            self.tCellsX = np.zeros((self.numTCells, int(self.numTimeSteps)))
            self.tCellsY = np.zeros((self.numTCells, int(self.numTimeSteps)))
            self.tCellsZ = np.zeros((self.numTCells, int(self.numTimeSteps)))

            # Time to start simulating the movement of the tCells

            for time in range(int(self.numTimeSteps)):
                # convert time to minutes
                mins = self.timeStep * time
                # check if we should be introducing a dendritic cell
                if (mins < self.DCArrivalDuration) and (len(dCellList) < mins * self.introRate + 1):
                    dCellList.append(dCell(-1 * np.ones(3)))
                    PLACE_DC_ON_GRID(len(dCellList) - 1, dCellList, occupiedPositions)
                    dCellList[-1].cogAgRatio = self.cogAgOnArrival * np.exp(-self.antigenDecayRate * time * self.timeStep)
                    dCellList[-1].timeAntigenCountLastUpdated = time
                # clear  cellsToDelete from previous step
                cellsToDelete = []
                # break if we've exhausted all the tcells
                if (len(cellMovementOrder) == 0):
                    break
                # check if we should even bother simulating or whether DCs have too little antigen to keep going
                if dCellList[0].cogAgRatio * np.exp(
                        -self.antigenDecayRate * (time - dCellList[0].timeAntigenCountLastUpdated) * self.timeStep) < 0.00285:
                    break
                for i in range(len(cellMovementOrder)):
                    tCellNum = cellMovementOrder[i]
                    cell = tCellList[tCellNum]
                    current_posn = cell.posn
                    initial_posn = cell.initial_posn
                    disp = current_posn - initial_posn
                    # time to actually move this cell
                    velocity = cell.vel
                    dvel = timeStep * velocity
                    new_posn_vec = current_posn + dvel
                    freePathRemaining[tCellNum] -= dvel.dot(dvel)

                    # check if we've left the sphere or if we've reached the end of the mean free path
                    if not inside_sphere(new_posn_vec, radius):
                        # regenerate velocity to move t cell away from sphere surface
                        vmag = np.random.gamma(self.tGammaShape, self.tGammaScale)
                        theta = np.arccos(np.random.uniform(0, 1))
                        phi = 2 * np.pi * np.random.uniform(0, 1)
                        current_posn /= MAGNITUDE(current_posn)
                        velocity = -vmag * current_posn
                        velocity = arbitraryAxisRotation(current_posn[2], current_posn[1], -1 * current_posn[0],
                                                         velocity[0], velocity[1], velocity[2], theta)
                        velocity = arbitraryAxisRotation(current_posn[0], current_posn[1], current_posn[2], velocity[0],
                                                         velocity[1], velocity[2], phi)
                        cell.vel = velocity
                        freePathRemaining[tCellNum] = np.random.normal(self.freePathMean, self.freePathStDev)
                        new_posn_vec += self.timeStep * velocity - dvel
                    elif freePathRemaining[tCellNum] <= 0:
                        vmag = np.random.gamma(self.tGammaShape, self.tGammaScale)
                        theta = np.arccos(np.random.uniform(0, 1))
                        phi = 2 * np.pi * np.random.uniform(0, 1)
                        cell.vel = np.asarray([vmag * np.sin(theta) * np.cos(phi), vmag * np.sin(theta) * np.sin(phi),
                                               vmag * np.cos(theta)])
                        freePathRemaining[tCellNum] = np.random.normal(self.freePathMean, self.freePathStDev)

                    # now we just need to check for DCs nearby to see if they activate the tcell
                    inContact = 0
                    inContact = CHECK_CONTACT_WITH_DENDRITES_T(new_posn_vec, inContact, dCellList, tCellList, tCellNum,
                                                               occupiedPositions)
                    if inContact == 1:
                        # this t cell encountered some dendrites. Let's see if the interaction was successful
                        # we need to manually get the corresponding DC's coordinates and number!
                        [dend_vec, dCellNum] = GET_CONTACTING_DC_COORDS_AND_NUM(new_posn_vec, dCellList, tCellList,
                                                                                tCellNum, occupiedPositions)
                        # should update the DC antigen count and corresponding activation probability
                        dCellList[dCellNum].cogAgRatio = dCellList[dCellNum].cogAgRatio * np.exp(
                            -self.antigenDecayRate * (time - dCellList[dCellNum].timeAntigenCountLastUpdated) * self.timeStep)
                        dCellList[dCellNum].probActivation = ProbTable.getProbability(dCellList[dCellNum].cogAgRatio)
                        dCellList[dCellNum].timeAntigenCountLastUpdated = time
                        dend_coord_vec = set_coordinates(dend_vec, radius, cellSide)
                        this_numInteractions += 1
                        if cell.increment_num_interactions == 1:
                            this_numUniqueInteractions += 1
                        if (dCellList[dCellNum].cannot_activate_t_cells) and not (self.noTimeLimit):
                            # no point simulating this anymore if it'll never be able to activate a tCell
                            REMOVE_DC_FROM_DISCRETE_GRID(dend_coord_vec, dCellNum, occupiedPositions)
                        # now see if t cell has been activated
                        elif np.random.uniform(0, 1) < dCellList[dCellNum].probActivation:
                            # activation successful! let's make a note to remove this cell once we've finished cycling through the tcells
                            # print("activation at time ", timeStep*time)
                            if presentSuccess == 0:
                                self.successfulActivations += 1
                            presentSuccess = 1
                            cellsToDelete.append(tCellNum)
                            this_numActivated += 1
                            # some waffle about timeUntilFirstActivation which I'll ignore atm
                        else:
                            # activation has failed, mark this tCell-DC pair so we don't repeatedly try to interact
                            tCellList[tCellNum].failed_interaction_ID = dCellNum
                            # we regenerate a velocity to move the T cell away from the DC after interaction
                            vmag = np.random.gamma(self.tGammaShape, self.tGammaScale)
                            theta = np.arccos(np.random.uniform(0, 1))
                            phi = 2 * np.pi * np.random.uniform(0, 1)
                            posn_vec = new_posn_vec - dend_vec
                            magsq = MAGNITUDE(posn_vec)
                            posn_vec /= magsq
                            velocity = vmag * posn_vec
                            velocity = arbitraryAxisRotation(posn_vec[2], posn_vec[1], -posn_vec[0], velocity[0],
                                                             velocity[1], velocity[2], theta)
                            velocity = arbitraryAxisRotation(posn_vec[0], posn_vec[1], posn_vec[2], velocity[0],
                                                             velocity[1], velocity[2], phi)
                            tCellList[tCellNum].vel = velocity
                            freePathRemaining[tCellNum] = np.random.normal(self.freePathMean, self.freePathStDev)
                    else:
                        pass
                    tCellList[tCellNum].posn = new_posn_vec
                    # only want to update this every minute
                    if mins % 1 == 0:
                        self.tCellsX[tCellNum, int(mins)] = new_posn_vec[0]
                        self.tCellsY[tCellNum, int(mins)] = new_posn_vec[1]
                        self.tCellsZ[tCellNum, int(mins)] = new_posn_vec[2]

                # end of t cell loop. Just need to delete the cells which successfully interacted
                for i in range(len(cellsToDelete)):
                    cellNo = cellsToDelete[i]
                    # remove from cellMovementOrder and fix them in their final location for rest of animation
                    self.tCellsX[cellNo, int(mins):] = tCellList[cellNo].posn[0]
                    self.tCellsY[cellNo, int(mins):] = tCellList[cellNo].posn[1]
                    self.tCellsZ[cellNo, int(mins):] = tCellList[cellNo].posn[2]
                    cellMovementOrder = np.delete(cellMovementOrder, np.where(cellMovementOrder == cellNo)[0][0])

        print("Final outcome: ", self.successfulActivations, " successes from ", self.numRepeats, " trials")
        print("This represents a probability of success of ", round(self.successfulActivations / self.numRepeats, 5))
        return round(self.successfulActivations / self.numRepeats, 5)


