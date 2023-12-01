# Siyi Qian 200012298709
# Chenting Zhang 200205146202
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random



# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100
    MINOTAUR_REWARD = -100


    def __init__(self, maze, minotaur_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map, self.exit_states = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.minotaur_stay = minotaur_stay
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()


    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1, 0)
        actions[self.MOVE_DOWN]  = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        exit_states = []
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] != 1:
                    for k in range(self.maze.shape[0]):
                        for l in range(self.maze.shape[1]):
                            states[s] = (i, j, k, l)
                            map[(i, j, k, l)] = s
                            s += 1
                            if self.maze[i, j] == 2:
                                # print(i, j, 'exit coordinate')
                                exit_states.append(map[(i, j, k, l)])
        # print(exit_states)
        return states, map, exit_states

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        next_s = []
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row, col] == 1)
        if hitting_maze_walls:
            row = self.states[state][0]
            col = self.states[state][1]
        # Minotaur move
        starting = 0 if self.minotaur_stay else 1
        for i in range(starting, len(self.actions)):
            row_m = self.states[state][2] + self.actions[i][0]
            col_m = self.states[state][3] + self.actions[i][1]
            hitting_maze_walls_m = (row_m == -1) or (row_m == self.maze.shape[0]) or \
                                   (col_m == -1) or (col_m == self.maze.shape[1])
            if not hitting_maze_walls_m:
                next_s.append(self.map[(row, col, row_m, col_m)])
        return next_s

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. NS/1
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)
                if (self.states[s][:2]==self.states[s][2:]):    # player and minotaur are in the same grid, player die
                    for i in range(len(next_s)):
                        transition_probabilities[next_s[i], s, a] = 0
                    continue
                prob_next = 1/len(next_s)
                for i in range(len(next_s)):
                    transition_probabilities[next_s[i], s, a] = prob_next

        return transition_probabilities

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        # if weights is None:
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)
                for ns in next_s:
                    # player stays and being eaten at next state
                    player_stay = (self.states[s][0] == self.states[ns][0]) and (self.states[s][1] == self.states[ns][1])
                    player_eaten = (self.states[ns][0] == self.states[ns][2]) and (self.states[ns][1] == self.states[ns][3])
                    # situations of being eaten:
                    # 1. being eaten somewhere other than the exit
                    # 2. player and minotaur arrived at the exit at the same time
                    # 3. minotaur arrived the exit after the player (in this case player still wins)
                    if player_eaten and s not in self.exit_states and ns not in self.exit_states:
                        rewards[s, a] += self.MINOTAUR_REWARD

                    elif player_eaten and s not in self.exit_states and ns in self.exit_states:
                        rewards[s, a] += self.MINOTAUR_REWARD

                    elif player_eaten and s in self.exit_states and ns in self.exit_states:
                        rewards[s, a] += self.GOAL_REWARD

                    # Reward for hitting a wall
                    elif player_stay and a != self.STAY:
                        rewards[s, a] += self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                    # elif player_stay and self.maze[self.states[ns][:2]] == 2:
                    elif player_stay and ns in self.exit_states:
                        rewards[s, a] += self.GOAL_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] += self.STEP_REWARD
                rewards[s, a] = rewards[s, a]/len(next_s)

        # # If the weights are descrobed by a weight matrix
        # else:
        #     for s in range(self.n_states):
        #          for a in range(self.n_actions):
        #              next_s = self.__move(s, a)
        #              i, j, k, l = self.states[next_s]
        #              # Simply put the reward as the weights o the next state.
        #              rewards[s, a] = weights[i][j][k][l]

        return rewards

    # def __rewards(self):
    #     """ Computes the reward function for every state action pair.
    #     """
    #     rewards = np.zeros((self.n_states, self.n_actions))
    #
    #     for s in range(self.n_states):
    #         for a in range(self.n_actions):
    #             prob_states = self.__move(s, a)
    #             # Compute rewards for each probable next state
    #             for next_s in prob_states:
    #                 # Reward for hitting a wall
    #                 if self.states[s][:2] == self.states[next_s][:2] and a != self.STAY:
    #                     rewards[s,a] += self.IMPOSSIBLE_REWARD
    #                 # Reward for reaching the exit
    #                 elif self.states[s][:2] == self.states[next_s][:2] and self.maze[self.states[next_s][:2]] == 2:
    #                     rewards[s,a] += self.GOAL_REWARD
    #                 # Reward for encountering the minotaur
    #                 elif self.states[next_s][:2] == self.states[next_s][2:]:
    #                     rewards[s,a] += self.IMPOSSIBLE_REWARD
    #                 # Reward for taking a step to an empty cell that is not the exit
    #                 else:
    #                     rewards[s,a] += self.STEP_REWARD
    #
    #             rewards[s,a] /= len(prob_states)
    #
    #     return rewards

    def simulate(self, start, policy, method, gamma=29/30):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s, policy[s, t]))
                # Add the position in the maze corresponding to the next state
                # to the path
                # Check exit states
                if s in self.exit_states and next_s not in self.exit_states:
                    print('wrong decision')

                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = random.choice(self.__move(s, policy[s]))
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # needs to be modified
            # Loop while state is not the goal state
            while (self.maze[self.states[next_s][:2]] != 2 and random.random() < gamma):
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = random.choice(self.__move(s,policy[s]));
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
        return path



    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic programming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy

def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma;

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V);
        BV = np.max(Q, 1);
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1);
    # Return the obtained policy
    return V, policy;

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols    = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols    = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0, 0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[path[i][:2]].get_text().set_text('Player')
        grid.get_celld()[path[i][2:]].get_text().set_text('Minotaur')

        if i > 0:
            # update the color of the previous player position
            # if path[i-1][:2] != path[i][:2]:
            #     grid.get_celld()[path[i-1][:2]].set_facecolor(col_map[maze[path[i-1][:2]]])
            #     grid.get_celld()[path[i-1][:2]].get_text().set_text('')

            # check if the player has arrived at the goal
            # update the color of the previous minotaur position
            if path[i-1][2:] != path[i][2:]:
                grid.get_celld()[path[i-1][2:]].set_facecolor(col_map[maze[path[i-1][2:]]])
                grid.get_celld()[path[i-1][2:]].get_text().set_text('')

        if maze[path[i][:2]] == 2:
            grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_GREEN)
            grid.get_celld()[path[i][:2]].get_text().set_text('Player winned ^_^')
        else:
            grid.get_celld()[path[i][:2]].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[path[i][:2]].get_text().set_text(str(i)+'th time')

        grid.get_celld()[path[i][2:]].set_facecolor(LIGHT_PURPLE)

        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)

def success_rate(maze, minotaur_stay=False, method='DynProg', horizon=30, gamma=29/30, epsilon=0.0001):

    simulate_time = 10000
    success_time = 0
    env = Maze(maze, minotaur_stay)
    rate = list()
    x_axis = list()
    start = (0, 0, 6, 5)
    if method == 'DynProg':
        for T in range(1, horizon+1):
            success_time = 0
            x_axis.append(T)
            V, policy = dynamic_programming(env, T)
            for _ in range(simulate_time):
                # V, policy = dynamic_programming(env, T)
                path = env.simulate(start, policy, method, gamma)
                if maze[path[-1][:2]] == 2:
                    success_time += 1
            rate.append(success_time/simulate_time)
            print(str(T)+'th', success_time/simulate_time)
        # display
        plt.figure()
        plt.scatter(x_axis, rate, marker='.')
        plt.xlabel('T')
        plt.ylabel('success rate')
        plt.title('Probability that the player wins')
        plt.show()
    if method == 'ValIter':
        success_time = 0
        V, policy = value_iteration(env, gamma=gamma, epsilon=epsilon)
        for _ in range(simulate_time):
            path = env.simulate(start, policy, method, gamma)
            if maze[path[-1][:2]] == 2:
                success_time += 1
        print(success_time/simulate_time)




def survival_probability(maze, minotaur_stay=False, start=(0, 0, 6, 5), goal=(6, 5),
                         num_simulations=10000, method='DynProg', **kwargs):
    # create the Maze environment
    env = Maze(maze, minotaur_stay)

    if method == 'DynProg':
        sp = []
        T = kwargs.get('T', 20)
        for horizon in range(1, T+1):
            win_counter = 0
            V, policy = dynamic_programming(env, horizon)

            for simulation_idx in range(num_simulations):
                path = env.simulate(start, policy, method)
                reached_goal = path[-1][:2] == goal
                if reached_goal:
                    win_counter += 1
            sp.append(win_counter / num_simulations)

        plt.figure()
        plt.plot(range(1, T+1), sp, 'o')
        plt.title('Survival probability for horizon T')
        plt.xlabel('T')
        plt.ylabel('Probability')
        plt.grid()
        plt.xticks(np.arange(0, T+5, 5))
        plt.yticks(np.arange(0, 1.1, 0.1))

    if method == 'ValIter':
        win_counter = 0
        gamma = kwargs.get('gamma', 29/30)
        epsilon = kwargs.get('epsilon', 0.0001)
        V, policy = value_iteration(env, gamma, epsilon)

        for simulation_idx in range(num_simulations):
            path = env.simulate(start, policy, method, gamma)
            reached_goal = path[-1][:2] == goal
            if reached_goal:
                win_counter += 1

        sp = win_counter / num_simulations

    return sp


