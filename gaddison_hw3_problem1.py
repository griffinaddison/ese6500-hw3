import numpy as np


import matplotlib.pyplot as plt


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

        # self.start_pos = np.array([3, 6])
        # self.goal_pos = np.array([8, 1])

        self.start_pos = (3, 6) 
        self.goal_pos = (5, 1) 

        self.reachedGoal = False
        

        # Initialize policy to all east
        self.policy = np.full((10, 10), 'E', dtype=str)
        # Should always not move from goal (maybe theres a better way to do this)
        self.policy[self.goal_pos[0], self.goal_pos[1]] = 'X'
        
        self.curr_values = np.zeros((10, 10))

    def get_transition_matrix(self, control):
        
        '''Tells us the probability of ending up in each state,
        given the previous state and a certain action'''

        
        T = np.zeros((100, 100))

        
        if control == "X":

                next_state_probs_map = np.eye(10)
                

        if control == "N":

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

                    T[x + y*10, :] = next_state_probs_map.flatten()




        if control == "S":

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

                    T[x + y*10, :] = next_state_probs_map.flatten()


                
        if control == "E":

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

                    T[x + y*10, :] = next_state_probs_map.flatten()


        if control == "W":

            for x in range(10):
                for y in range(10):

                    
                    next_state_probs_map = np.zeros((10, 10))
                    
                    next_state_probs_map[keepInGrid(x-1), y] += 0.7
                    next_state_probs_map[x, keepInGrid(y+1)] += 0.1
                    next_state_probs_map[x, keepInGrid(y-1)] += 0.1
                    next_state_probs_map[x, y] += 0.1

                    T[x + y*10, :] = next_state_probs_map.flatten()

           
                

        return T

    def evaluate_policy(self, discount=0.9, threshold=0.0001):
        counter = 0
        # Iteratively, until values are ~stable
        while True:
            counter += 1
            prev_values = np.copy(self.curr_values)
            # For each possible state
            # for x in range(self.grid.shape[0]-1, -1, -1):
            #     for y in range(self.grid.shape[1]-1, -1, -1):

            for x in range(self.grid.shape[0]):
                for y in range(self.grid.shape[1]):
                    # Rename local variables for clarity
                    curr_state = (x, y)
                    inObstacleCell = self.grid[curr_state] == 1
                    inGoalCell = curr_state == self.goal_pos

                    control = self.policy[curr_state]
                    trans_probs = self.get_transition_matrix(control) 

                    # # Don't bother evaluating terminal states 
                    # # (states from which you cannot move)
                    # if inObstacleCell or inGoalCell:
                    #     continue
                    self.reachedGoal = False

                    # Calculate the expected value of this state
                    runtime_value = self.get_runtime_value(curr_state, control)
                    if curr_state == (8, 1):
                        print("curr_state: ", curr_state)
                        print("runtime_value: ", runtime_value)
                    # curr_state_linIdx = curr_state[0] * 10 + curr_state[1]
                    curr_state_linIdx = curr_state[1] * 10 + curr_state[0]
                    expected_future_value = \
                        trans_probs[curr_state_linIdx].dot(self.curr_values.flatten())
                    expected_value = runtime_value + discount * expected_future_value
                    print("expected_future_value: ", expected_future_value)
                    print("expected_value: ", expected_value)
                    self.curr_values[curr_state] = np.copy(expected_value)
           
            delta = np.max(np.abs(self.curr_values - prev_values))
            if delta < threshold:
                print("Policy evaluation converged after ", counter, \
                      " steps w/ delta ", delta)
                break

    def get_runtime_value(self, state, control):
        
        value = 0
        
        # if np.any(control):
        #     # Cost due to control (is always 1)
        #     cost += 1
       
        value -= 1 # Cost for any control is -1

        # Cost due to state
        inObstacleCell = self.grid[state[0], state[1]] == 1
        print(" state: ", state)
        print("self.goal_pos: ", self.goal_pos)
        inGoalCell = state == self.goal_pos
        print("inGoalCell: ", inGoalCell)
        if inObstacleCell:
            # print("in obstacle cell at x, y: ", state )
            value -= 10
        elif inGoalCell: # and not self.reachedGoal:
            print(" in goal cell at x, y: ", state)
            value += 10
            # self.reachedGoal = True
        else:
            value += 0

        print("return value", value)
        return value


    def get_terminal_cost(self, state, control):
        return self.get_runtime_cost(state, control=np.zeros((2,)))
        





    def print(self):
        print("\n Grid: \n", self.grid)
        print("\n PLEASE NOTE: printed grid is rotated cuz thats how np indexing works.")
        print("\n Please use plotGrid() to see correct orientaiton.\n")



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



                plt.text(x, y-0.2, str(self.policy[x, y]))
                plt.text(x, y+0.2, round(self.curr_values[x, y], 2))
                alpha = np.abs(np.copy(self.curr_values[x, y]) / np.max(np.abs(self.curr_values)))
                plt.plot(x, y, color=(0.5, 0.5, 0.0, alpha))
                # # Plot goal position (3) as green
                # if self.grid[x, y] == "G":
                #     plt.plot(x, y, color="lightgreen")

                

        plt.show()

# ## Create the grid
# gridWorld = np.zeros((10, 10))
# # Fill it in
# gridWorld[-1, :] = 1    .


mygrid = GridWorld()


mygrid.print()


mygrid.evaluate_policy()


mygrid.plotGrid()
