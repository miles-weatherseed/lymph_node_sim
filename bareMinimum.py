from initialization import *
from minimumFunctions import *
import numpy as np


class dCell:

    def __init__(self, posn):

        self.posn = posn
        self.cogAgRatio = 0
        self.probActivation = 0
        self.timeAntigenCountLastUpdated = 0

    @property
    def posn(self):
        return self.__posn

    @posn.setter
    def posn(self, value):
        self.__posn = value

    @property
    def cogAgRatio(self):
        return self.__cogAgRatio

    @cogAgRatio.setter
    def cogAgRatio(self, value):
        self.__cogAgRatio = value

    @property
    def probActivation(self):
        return self.__probActivation

    @probActivation.setter
    def probActivation(self, value):
        self.__probActivation = value

    @property
    def timeAntigenCountLastUpdated(self):
        return self.__timeAntigenCountLastUpdated

    @timeAntigenCountLastUpdated.setter
    def timeAntigenCountLastUpdated(self, value):
        self.__timeAntigenCountLastUpdated = value

    @property
    def cannot_activate_t_cells(self):
        return (self.probActivation<PROBABILITY_TOLERANCE)

class tCell:

    def __init__(self, posn, vel):
        self.posn = posn
        self.initial_posn = posn
        self.vel = vel
        self.failed_interaction_ID = -1
        self.interactions = 0

    @property
    def posn(self):
        return self.__posn

    @posn.setter
    def posn(self, value):
        self.__posn = value

    @property
    def initial_posn(self):
        return self.__initial_posn

    @initial_posn.setter
    def initial_posn(self, value):
        self.__initial_posn = value

    @property
    def vel(self):
        return self.__vel

    @vel.setter
    def vel(self, value):
        self.__vel = value

    @property
    def failed_interaction_ID(self):
        return self.__failed_interaction_ID

    @failed_interaction_ID.setter
    def failed_interaction_ID(self, value):
        self.__failed_interaction_ID = value

    @property
    def interactions(self):
        return self.__interactions

    @interactions.setter
    def interactions(self, value):
        self.__interactions = value

    @property
    def increment_num_interactions(self):
        return self.__interactions + 1


# create probabilities table if it doesn't already exist

with open("config/ProbabilitiesTable.dat") as datFile:
    if (int(datFile.readline()[18:24]) == numAntigenOnDC) and (int(datFile.readline()[28:31]) == numAntigenInContactArea) \
        and (int(datFile.readline()[27:39]) == tCellActivationThreshold):
        pass
    else:
        generateLookupTable(numAntigenOnDC, numAntigenInContactArea, tCellActivationThreshold)

# now we import this table and store it

ProbabilitiesTable = np.loadtxt("config/ProbabilitiesTable.dat", delimiter="\t", skiprows=3)

# create variable to determine proportion of successful activations

successfulActivations = 0

# this is where each repeat starts

for rep in range(numRepeats):

    presentSuccess = 0
    occupiedPositions = list(np.zeros(occupiedPositionsArraySize))  # this is a 1d representation of 3d space showing where the cells we've already put are
    cellMovementOrder = np.arange(numTCells)
    freePathRemaining = np.empty(numTCells)

    # create list of DCs

    dCellList = []

    # create list of T cells

    tCellList = []
    for i in range(numTCells):
        tCellList.append(tCell(-1*np.ones(3), np.zeros(3)))


    # time to get the t cells in there, away from the DCs

    for i in range(numTCells):
        PLACE_T_ON_GRID(i, tCellList, dCellList, occupiedPositions, freePathRemaining)

    # finally initializing a few more data holding variables

    this_numInteractions = 0
    this_numUniqueInteractions = 0
    this_numActivated = 0

    # let's make some containers to hold the positions of our T Cells at each time step

    tCellsX = np.zeros((numTCells,int(numTimeSteps)))
    tCellsY = np.zeros((numTCells,int(numTimeSteps)))
    tCellsZ = np.zeros((numTCells,int(numTimeSteps)))

    # Time to start simulating the movement of the tCells

    for time in range(int(numTimeSteps)):
        # convert time to minutes
        mins = timeStep * time
        # check if we should be introducing a dendritic cell
        if (mins < DCArrivalDuration) and (len(dCellList) < mins*introRate + 1):
            dCellList.append(dCell(-1 * np.ones(3)))
            PLACE_DC_ON_GRID(len(dCellList)-1, dCellList, occupiedPositions)
            dCellList[-1].cogAgRatio = cogAgOnArrival*np.exp(-antigenDecayRate*time*timeStep)
            dCellList[-1].timeAntigenCountLastUpdated = time
        # clear  cellsToDelete from previous step
        cellsToDelete = []
        # first record number of tCells that got activated at end of previous loop (i.e. start of this step)
        if (time % (numTimeSteps / numTimeMeasurements) == 0):
            pass
            # print this to the terminal
        # break if we've exhausted all the tcells
        if (len(cellMovementOrder)==0):
            print("early exit at time ", mins, " minutes")
            break
        # check if we should even bother simulating or whether DCs have too little antigen to keep going
        if dCellList[0].cogAgRatio*np.exp(-antigenDecayRate*(time - dCellList[0].timeAntigenCountLastUpdated)*timeStep) < 0.00285:
            print("early exit at time ", mins, " minutes")
            break
        for i in range(len(cellMovementOrder)):
            tCellNum = cellMovementOrder[i]
            cell = tCellList[tCellNum]
            current_posn = cell.posn
            initial_posn = cell.initial_posn
            disp = current_posn - initial_posn
            if (time%(numTimeSteps/numTimeMeasurements)==0):
                dist = np.sqrt(disp.dot(disp))
                # print this to the terminal
            # time to actually move this cell
            velocity = cell.vel
            dvel = timeStep*velocity
            new_posn_vec = current_posn + dvel
            freePathRemaining[tCellNum] -= dvel.dot(dvel)

            # check if we've left the sphere or if we've reached the end of the mean free path
            if not INSIDE_SPHERE(new_posn_vec):
                # regenerate velocity to move t cell away from sphere surface
                vmag = np.random.gamma(tGammaShape, tGammaScale)
                theta = np.arccos(np.random.uniform(0,1))
                phi = 2*np.pi*np.random.uniform(0, 1)
                current_posn /= MAGNITUDE(current_posn)
                velocity =-vmag*current_posn
                #print(velocity)
                velocity = arbitraryAxisRotation(current_posn[2], current_posn[1], -1*current_posn[0], velocity[0], velocity[1], velocity[2], theta)
                velocity = arbitraryAxisRotation(current_posn[0], current_posn[1], current_posn[2], velocity[0], velocity[1], velocity[2], phi)
                cell.vel = velocity
                freePathRemaining[tCellNum] = np.random.normal(freePathMean, freePathStDev)
                new_posn_vec += timeStep*velocity - dvel
            elif freePathRemaining[tCellNum] <= 0:
                vmag = np.random.gamma(tGammaShape, tGammaScale)
                theta = np.arccos(np.random.uniform(0, 1))
                phi = 2 * np.pi * np.random.uniform(0, 1)
                cell.vel = np.asarray([vmag*np.sin(theta)*np.cos(phi), vmag*np.sin(theta)*np.sin(phi), vmag*np.cos(theta)])
                freePathRemaining[tCellNum] = np.random.normal(freePathMean, freePathStDev)

            # now we just need to check for DCs nearby to see if they activate the tcell
            inContact = 0
            inContact = CHECK_CONTACT_WITH_DENDRITES_T(new_posn_vec, inContact, dCellList, tCellList, tCellNum, occupiedPositions)
            if inContact == 1:
                # this t cell encountered some dendrites. Let's see if the interaction was successful
                # we need to manually get the corresponding DC's coordinates and number!
                [dend_vec, dCellNum] = GET_CONTACTING_DC_COORDS_AND_NUM(new_posn_vec, dCellList, tCellList,
                                                                                   tCellNum, occupiedPositions)
                # should update the DC antigen count and corresponding activation probability
                dCellList[dCellNum].cogAgRatio = dCellList[dCellNum].cogAgRatio*np.exp(-antigenDecayRate*(time - dCellList[dCellNum].timeAntigenCountLastUpdated)*timeStep)
                dCellList[dCellNum].probActivation = ProbabilitiesTable[np.where(ProbabilitiesTable[:, 0]==round((1E-5)*int(dCellList[dCellNum].cogAgRatio/(1E-5)), 5))[0][0]][1]
                dCellList[dCellNum].timeAntigenCountLastUpdated = time
                dend_coord_vec = SET_COORDINATES(dend_vec)
                this_numInteractions += 1
                if cell.increment_num_interactions==1:
                    this_numUniqueInteractions += 1
                if (dCellList[dCellNum].cannot_activate_t_cells) and not (noTimeLimit):
                    # no point simulating this anymore if it'll never be able to activate a tCell
                    REMOVE_DC_FROM_DISCRETE_GRID(dend_coord_vec, dCellNum, occupiedPositions)
                # now see if t cell has been activated
                elif np.random.uniform(0, 1) < dCellList[dCellNum].probActivation:
                    #activation successful! let's make a note to remove this cell once we've finished cycling through the tcells
                    #print("activation at time ", timeStep*time)
                    if presentSuccess == 0:
                        successfulActivations += 1
                    presentSuccess = 1
                    cellsToDelete.append(tCellNum)
                    this_numActivated += 1
                    # some waffle about timeUntilFirstActivation which I'll ignore atm
                else:
                    # activation has failed, mark this tCell-DC pair so we don't repeatedly try to interact
                    tCellList[tCellNum].failed_interaction_ID = dCellNum
                    # we regenerate a velocity to move the T cell away from the DC after interaction
                    vmag = np.random.gamma(tGammaShape, tGammaScale)
                    theta = np.arccos(np.random.uniform(0, 1))
                    phi = 2 * np.pi * np.random.uniform(0, 1)
                    posn_vec = new_posn_vec - dend_vec
                    magsq = MAGNITUDE(posn_vec)
                    posn_vec /= magsq
                    velocity = vmag*posn_vec
                    velocity = arbitraryAxisRotation(posn_vec[2], posn_vec[1], -posn_vec[0], velocity[0],
                                                     velocity[1], velocity[2], theta)
                    velocity = arbitraryAxisRotation(posn_vec[0], posn_vec[1], posn_vec[2], velocity[0],
                                                     velocity[1], velocity[2], phi)
                    tCellList[tCellNum].vel = velocity
                    freePathRemaining[tCellNum] = np.random.normal(freePathMean, freePathStDev)
            else:
                pass
            tCellList[tCellNum].posn = new_posn_vec
            # only want to update this every minute
            if mins%1==0:
                tCellsX[tCellNum, int(mins)] = new_posn_vec[0]
                tCellsY[tCellNum, int(mins)] = new_posn_vec[1]
                tCellsZ[tCellNum, int(mins)] = new_posn_vec[2]

        # end of t cell loop. Just need to delete the cells which successfully interacted
        for i in range(len(cellsToDelete)):
            cellNo = cellsToDelete[i]
            # remove from cellMovementOrder and fix them in their final location for rest of animation
            tCellsX[cellNo, int(mins):] = tCellList[cellNo].posn[0]
            tCellsY[cellNo, int(mins):] = tCellList[cellNo].posn[1]
            tCellsZ[cellNo, int(mins):] = tCellList[cellNo].posn[2]
            cellMovementOrder = np.delete(cellMovementOrder, np.where(cellMovementOrder == cellNo)[0][0])


    # end of simulation, time to plot and animate the outcome

    if not animateStatus:
        PLOT_ANIMATION(dCellList, tCellsX, tCellsY, tCellsZ, arrivalTimes)

print("Final outcome: ", successfulActivations, " successes from ", numRepeats, " trials")
print("This represents a probability of success of ", round(successfulActivations/numRepeats, 5))