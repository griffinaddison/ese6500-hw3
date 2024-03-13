import numpy as np


import matplotlib.pyplot as plt


def clamp(n, minVal, maxVal): 
    return max(min(n, maxVal), minVal)

def keepInGrid(n):
    return clamp(n, 0, 9)


class GridWorld:
    def __init__(self, xdim=10, ydim=10):
        # Initialize zero grid
        self.grid = np.zeros((10, 10))
        # Wall obstacles
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        # L-shape obstacle
        self.grid[[4, 5], 7], self.grid[4, [4, 5, 6]] = 1, 1
        # Short obstacle
        self.grid[7, [4, 5]] = 1
        # Long obstacle
        self.grid[[3, 4, 5, 6], 2] = 1

        self.start_pos = np.array([3, 6])
        self.goal_pos = np.array([8, 1])

        self.reachedGoal = False

    def get_transition_matrix(self, control):
        
        '''Tells us the probability of ending up in each state,
        given the previous state and a certain action'''

        
        T = np.zeros((100, 100))


        if control == "north":

            for x in range(10):
                for y in range(10):

                    
                    next_state_probs_map = np.zeros((10, 10))
                    
                    # Probability of:
                    #   moving north (desired)
                    next_state_probs_map[x, keepInGrid(y+1)] += 0.7
                    #   moving west
                    next_state_probs_map[keepInGrid(x-1), y] += 0.1
                    #   moving east
                    next_state_probs_map[keepInGrid(x+1), y] += 0.1
                    #   staying in place
                    next_state_probs_map[x, y] += 0.1

                    T[x + y*10, :] = next_state_probs_map.reshape((100, 1)) 




        if control == "south":

            for x in range(10):
                for y in range(10):

                    
                    next_state_probs_map = np.zeros((10, 10))
                    
                    # Probability of:
                    #   moving south (desired)
                    next_state_probs_map[x, keepInGrid(y-1)] += 0.7
                    #   moving west
                    next_state_probs_map[keepInGrid(x-1), y] += 0.1
                    #   moving east
                    next_state_probs_map[keepInGrid(x+1), y] += 0.1
                    #   staying in place
                    next_state_probs_map[x, y] += 0.1

                    T[x + y*10, :] = next_state_probs_map.reshape((100, 1)) 


                
        if control == "east":

            for x in range(10):
                for y in range(10):

                    
                    next_state_probs_map = np.zeros((10, 10))
                    
                    # Probability of:
                    #   moving east (desired)
                    next_state_probs_map[keepInGrid(x+1), y] += 0.7
                    #   moving north
                    next_state_probs_map[x, keepInGrid(y+1)] += 0.1
                    #   moving south
                    next_state_probs_map[x, keepInGrid(y-1)] += 0.1
                    #   staying in place
                    next_state_probs_map[x, y] += 0.1

                    T[x + y*10, :] = next_state_probs_map.reshape((100, 1)) 


        if control == "west":

            for x in range(10):
                for y in range(10):

                    
                    next_state_probs_map = np.zeros((10, 10))
                    
                    next_state_probs_map[keepInGrid(x-1), y] += 0.7
                    next_state_probs_map[x, keepInGrid(y+1)] += 0.1
                    next_state_probs_map[x, keepInGrid(y-1)] += 0.1
                    next_state_probs_map[x, y] += 0.1

                    T[x + y*10, :] = next_state_probs_map.reshape((100, 1)) 

           
                

        return T



def get_runtime_cost(self, state, control):
    
    cost = 0
    
    if np.any(control):
        # Cost due to control (is always 1)
        cost += 1
    
    # Cost due to state
    inObstacleCell = self.grid[state[0], state[1]] == 1
    inGoalCell = state == self.goal_pos
    if inObstacleCell:
        cost += 10
    elif inGoalCell and not self.reachedGoal:
        cost -= 10
        self.reachedGoal = True
    else:
        cost += 0


    return cost



def get_terminal_cost(self, state, control=np.zeros((2,))):
    return self.get_runtime_cost(state, control)
    





    def print(self):
        print("\n Grid: \n", self.grid)
        print("\n PLEASE NOTE: grid is rotated cuz thats how np indexing works.")
        print("\n Please use plotGrid() for correct orientaiton.\n")



    def plotGrid(self):

        myplot = plt.figure()
        plt.xlim([-0.5, 9.5])
        plt.ylim([-0.5, 9.5])
        plt.axis('square')
        plt.rcParams['lines.marker'] = "s"
        plt.rcParams['lines.markersize'] = 45

        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):

                # # Plot starting position (1) as blue
                # if self.grid[x, y] == 'S':
                #     plt.plot(x, y, color="lightblue")

                # Plot obstacles (2) as grey
                if self.grid[x, y] == 1:
                    plt.plot(x, y, color="lightgrey")

                plt.plot(self.start_pos[0], self.start_pos[1], color="lightblue")
                plt.plot(self.goal_pos[0], self.goal_pos[1], color="lightgreen")

                # # Plot goal position (3) as green
                # if self.grid[x, y] == "G":
                #     plt.plot(x, y, color="lightgreen")

                

        plt.show()

# ## Create the grid
# gridWorld = np.zeros((10, 10))
# # Fill it in
# gridWorld[-1, :] = 1    


mygrid = GridWorld()


mygrid.print()
mygrid.plotGrid()




