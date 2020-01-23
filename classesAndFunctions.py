from parameters import *
import hypergeometric
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from numba import jit
from LNfunctions import inside_sphere, threed_to_1d, set_coordinates

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

def generateLookupTable(numAntigenAttachedToDC, numAntigenInContactArea, tCellActivationThreshold):
    # first write properties of table at top so that we know that we have the right one
    with open("config/ProbabilitiesTable.dat", "w") as outFile:
        outFile.write("NUM_ANTIGEN_ON_DC")
        outFile.write("\t")
        outFile.write(str(numAntigenAttachedToDC))
        outFile.write("\n")
        outFile.write("NUM_ANTIGEN_IN_CONTACT_AREA")
        outFile.write("\t")
        outFile.write(str(numAntigenInContactArea))
        outFile.write("\n")
        outFile.write("TCELL_ACTIVATION_THRESHOLD")
        outFile.write("\t")
        outFile.write(str(tCellActivationThreshold))
        outFile.write("\n")
        precision = 1E-5
        numSteps = int(1E5)
        reached_one = False
        for i in range(numSteps):
            cogAgR = float(i)*precision
            if not reached_one:
                prob = hypergeometric.hyperGeometricFormula(numAntigenAttachedToDC, numAntigenInContactArea, tCellActivationThreshold, cogAgR)
                if 1.0-prob < PROBABILITY_TOLERANCE:
                    reached_one = True
            else:
                prob = 1.0
            outFile.write(str(round(i*precision, 5)))
            outFile.write("\t")
            outFile.write(str(prob))
            outFile.write("\n")

def PLACE_DC_ON_GRID(i, dCellList, occupiedPositions):
    cell = dCellList[i]
    cellStatus = False
    while (cellStatus == False):
        # let's generate potential coordinates
        posn_vec = np.random.uniform(-1, 1, 3) * radius
        if inside_sphere(posn_vec, radius):
            # we need to check it's not touching any other DCs now
            inContact = 0
            inContact = CHECK_CONTACT_WITH_DENDRITES_DC(posn_vec, inContact, dCellList, occupiedPositions)
            if inContact == 1:
                continue
            else:
                cell.posn = posn_vec
                cellStatus = True
        else:
            continue
    # better update the occupied positions list
    coord_vec = set_coordinates(posn_vec, radius, cellSide)
    if occupiedPositions[threed_to_1d(coord_vec, numPositions)] == 0.0:
        occupiedPositions[threed_to_1d(coord_vec, numPositions)] = [i]
    else:
        occupiedPositions[threed_to_1d(coord_vec, numPositions)].append(i)

def PLACE_T_ON_GRID(i, tCellList, dCellList, occupiedPositions, freePathRemaining):
    cellStatus = False
    while (cellStatus == False):
        # let's generate potential coordinates
        posn_vec = np.random.uniform(-1, 1, 3) * radius
        if inside_sphere(posn_vec, radius):
            # we need to check it's not touching any other DCs now
            inContact = 0
            inContact = CHECK_CONTACT_WITH_DENDRITES_T(posn_vec, inContact, dCellList, tCellList, i, occupiedPositions)
            if inContact == 1:
                continue
            else:
                tCellList[i].posn = posn_vec
                tCellList[i].initial_posn = posn_vec
                freePathRemaining[i] = 0
                tCellList[i].failed_interaction_ID = -1
                tCellList[i].interactions = 0
                cellStatus = True
        else:
            continue

def CHECK_CONTACT_WITH_DENDRITES_DC(posn_vec, inContact, dCellList, occupiedPositions):
    coord_vec = set_coordinates(posn_vec, radius, cellSide)
    nearbyDCs = occupiedPositions[threed_to_1d(coord_vec, numPositions)]
    if isinstance(nearbyDCs, float):
        pass
    else:
        for j in nearbyDCs:
            cell = dCellList[int(j)]
            dend_vec = cell.posn
            diff_vec = posn_vec - dend_vec
            if diff_vec.dot(diff_vec) <= contactRadius**2:
                inContact = 1
                break
    return inContact

# this function is similar to above but also checks to see whether an interaction happens

def CHECK_CONTACT_WITH_DENDRITES_T(posn_vec, inContact, dCellList, tCellList, tCellNum, occupiedPositions):
    coord_vec = set_coordinates(posn_vec, radius, cellSide)
    nearbyDCs = occupiedPositions[threed_to_1d(coord_vec, numPositions)]
    if isinstance(nearbyDCs, float):
        pass
    else:
        for j in nearbyDCs:
            cell = dCellList[int(j)]
            if (HAS_CONTACTED_DC(tCellNum, j, tCellList)):
                continue
            dend_vec = cell.posn
            diff_vec = posn_vec - dend_vec
            if diff_vec.dot(diff_vec) <= contactRadius**2:
                inContact = 1
                break
    return inContact

# function doesn't exist in original code but python isn't so slick so we must fetch them

def GET_CONTACTING_DC_COORDS_AND_NUM(posn_vec, dCellList, tCellList, tCellNum, occupiedPositions):
    coord_vec = set_coordinates(posn_vec, radius, cellSide)
    nearbyDCs = occupiedPositions[threed_to_1d(coord_vec, numPositions)]
    if isinstance(nearbyDCs, float):
        pass
    else:
        for j in nearbyDCs:
            cell = dCellList[int(j)]
            if (HAS_CONTACTED_DC(tCellNum, j, tCellList)):
                continue
            dend_vec = cell.posn
            diff_vec = posn_vec - dend_vec
            if diff_vec.dot(diff_vec) <= contactRadius**2:
                cont_vec = dend_vec
                contNum = j
                break
    return [cont_vec, contNum]

def REMOVE_DC_FROM_DISCRETE_GRID(dcoord_vec, dCellNum, occupiedPositions):
    for p in range(3):
        p -= 1
        if (dcoord_vec[0] + p >= numPositions):
            break
        elif dcoord_vec[0] + p < 0:
            continue
        for q in range(3):
            q -= 1
            if dcoord_vec[1] + q >= numPositions:
                break
            elif dcoord_vec[1] + q < 0:
                continue
            for r in range(3):
                r -= 1
                if dcoord_vec[2] + r >= numPositions:
                    break
                elif dcoord_vec[2] + r < 0:
                    continue
                # now clear this particular dCellNum from the position in the occupiedPositions array
                nearbyDCs = occupiedPositions[threed_to_1d(dcoord_vec + np.asarray([p, q, r]), numPositions)]
                nearbyDCs = np.delete(nearbyDCs, np.where(nearbyDCs==dCellNum))
                occupiedPositions[threed_to_1d(dcoord_vec + np.asarray([p, q, r]), numPositions)] = nearbyDCs

# GEOMETRY

# this function rotates the path of a cell that has left the spherical domain

@jit
def arbitraryAxisRotation(axX, axY, axZ, vecX, vecY, vecZ, angle):
    rotationMatrix = [[np.cos(angle) + axX*axX*(1-np.cos(angle)), axX*axY*(1-np.cos(angle))-axZ*np.sin(angle), axX*axZ*(1-np.cos(angle)) + axY*np.sin(angle)],
                      [axX*axY*(1-np.cos(angle)) + axZ*np.sin(angle), np.cos(angle) + axY*axY*(1-np.cos(angle)), axY*axZ*(1-np.cos(angle))-axX*np.sin(angle)],
                      [axX*axZ*(1-np.cos(angle)) - axY*np.sin(angle), axY*axZ*(1-np.cos(angle))+axX*np.sin(angle), np.cos(angle)+axZ*axZ*(1-np.cos(angle))]]
    return np.asarray([vecX * rotationMatrix[0][0] + vecY * rotationMatrix[0][1] + vecZ * rotationMatrix[0][2],
            vecX * rotationMatrix[1][0] + vecY * rotationMatrix[1][1] + vecZ * rotationMatrix[1][2],
            vecX * rotationMatrix[2][0] + vecY * rotationMatrix[2][1] + vecZ * rotationMatrix[2][2]])

@jit
def MAGNITUDE(vec):
    return np.sqrt(vec.dot(vec))


# PLOTTING FUNCTION - NEED TO ALTER TO INTRODUCE DCS 1 BY 1 AT CORRESPONDING TIMES

def PLOT_ANIMATION(dCellList, tCellsX, tCellsY, tCellsZ, arrivalTimes, save=False):

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    lines = [ax.plot([], [], [], 'o', c='r')[0] for _ in range(numTCells)]  # lines to animate
    time_text = ax.text(0.02, 0.95, 0.95, 'time = 0', transform=ax.transAxes)
    lines.append(time_text)

    # Set the axes properties
    ax.set_xlim3d([-500, 500])
    ax.set_xlabel('X')

    ax.set_ylim3d([-500, 500])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-500, 500])
    ax.set_zlabel('Z')

    ax.set_title('Lymph Node Visualization')

    def init():
        for i in range(numDCells):
            coords = dCellList[i].posn
            ax.scatter(coords[0], coords[1], coords[2], c='b')
        for line in lines[:-1]:
            line.set_data([], [])
            line.set_3d_properties([])
        return lines

    def update_lines(num):
        x = 0
        for line in lines[:-1]:
            if tCellsX[x, num] == tCellsX[x, num - 1]:
                # cell must have been activates
                line.set_color("black")
            line.set_data(np.vstack((tCellsX[x, num], tCellsY[x, num])))
            line.set_3d_properties(tCellsZ[x, num])
            x += 1
        lines[-1].set_text("time = " + str(num+firstDCArrival))
        return lines

    ani = animation.FuncAnimation(fig, update_lines, int(numTimeSteps - firstDCArrival/timeStep), init_func=init,
                                  interval=50, blit=True, repeat=False)

    if save:
        ani.save('animation.gif', writer='imagemagick', fps=30)

    plt.show()

def HAS_CONTACTED_DC(tCellNum, dCellNum, tCellList):
    return tCellList[tCellNum].failed_interaction_ID==dCellNum