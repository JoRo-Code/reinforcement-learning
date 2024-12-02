"""
Computer Lab 1 
Writers: Melker Haglund 

"""
# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
from tqdm import tqdm

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'

IS_POISON = True

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
    STEP_REWARD = -1          #TODO
    GOAL_REWARD =  100         #TODO
    IMPOSSIBLE_REWARD = -1    #TODO, change this if we want to punish hitting wall/going out of bounds
    MINOTAUR_REWARD =  -100     #TODO

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()
        

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:#Only if the agents coordinates are in a wall we don't create a state
                            states[s] = ((i,j), (k,l))
                            map[((i,j), (k,l))] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1

        states[s] = 'Poison'
        map['Poison'] = s
        s += 1
        
        states[s] = 'Win'#weird to not have self.maze[i,j] == 2: as the condition for 'Win'
        map['Win'] = s
        
        return states, map

    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """
        p = random.random()
        if self.states[state] == 'Eaten' or self.states[state] == 'Win' or self.states[state] == 'Poison': # In these states, the game is over
            return [self.states[state]]
        elif p <= 1/30 and IS_POISON:#Don't think it matters in what order minotaur or poison
            return ['Poison']
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position (int)
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position (int)
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = ((row_player,col_player),(6,5)) in self.map  #TODO
            #Should work since the states that are created for the player does not include walls or anything outside the maze
        
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            rows_minotaur, cols_minotaur = [], []
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = (self.states[state][1][0] + actions_minotaur[i][0] == -1) or \
                                             (self.states[state][1][0] + actions_minotaur[i][0] == self.maze.shape[0]) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == -1) or \
                                             (self.states[state][1][1] + actions_minotaur[i][1] == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
          

            # Based on the impossiblity check return the next possible states.
            if not impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if  (self.states[state][0][0], self.states[state][0][1]) == (rows_minotaur[i], cols_minotaur[i]):# TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.states[state][0][0], self.states[state][0][1]) == (6,5):# TODO: We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if  (row_player,col_player)==(rows_minotaur[i], cols_minotaur[i]):# TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (row_player,col_player) == (6,5):# TODO:We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))
              
                return states
        
        
        

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # TODO: Compute the transition probabilities.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Get the list of possible next states for this (state, action) pair
                next_states = self.__move(s, a)
                # If we're in a terminal state (Eaten or Win), stay there with probability 1
                if self.states[s] == 'Eaten' or self.states[s] == 'Win' or self.states[s] == 'Poison':
                    transition_probabilities[s, s, a] = 1.0
                else:
                    # For non-terminal states, distribute probability uniformly among possible next states
                    prob = 1.0 / len(next_states)  # Uniform distribution
                    for next_s in next_states:
                        next_state_idx = self.map[next_s]  #Get the index of the next state
                        if IS_POISON:
                            if next_s == 'Poison':
                                transition_probabilities[s, next_state_idx, a] = 1/30
                            else:
                                transition_probabilities[s, next_state_idx, a] = 29/30 * prob
                        else:
                            transition_probabilities[s, next_state_idx, a] = prob
                    

        return transition_probabilities



    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten' or self.states[s] == 'Poison': # The player has died
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards




    def simulate(self, start, policy, method):
        
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        
        if method == 'DynProg':
            horizon = policy.shape[1] # Deduce the horizon from the policy shape
            t = 0 # Initialize current time
            s = self.map[start] # Initialize current state 
            path.append(start) # Add the starting position in the maze to the path
            
            while t < horizon - 1:
                a = policy[s, t] # Move to next state given the policy and the current state
                next_states = self.__move(s, a) 
                next_s = random.choice(next_states) #Need to pick a random next state since the minotaur is random 
                path.append(next_s) # Add the next state to the path
                t +=1 # Update time and state for next iteration
                s = self.map[next_s]
                
        if method == 'ValIter': 
            t = 1 # Initialize current state, next state and time
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
            next_s = random.choice(next_states) #Need to pick a random next state since the minotaur is random 
            path.append(next_s) # Add the next state to the path
            ################CHANGE#########
            horizon = 30                               # Question e, we don't want to get stuck
            ###############
            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                s = self.map[next_s] # Update state
                next_states = self.__move(s, policy[s]) # Move to next state given the policy and the current state
                next_s = random.choice(next_states) #Need to pick a random next state since the minotaur is random 
                path.append(next_s) # Add the next state to the path
                t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI



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
    #TODO
    #Initialize V-function and the policy
    V = np.zeros((env.n_states,horizon)) #The terminal state, time wise, will always be 0
    policy = np.zeros((env.n_states,horizon))
    #DP to find the best policy in each state 
    for t in range(horizon-1,-1,-1): #Start the solution at the end and work your way backwards
        for s in range(env.n_states):
            v = np.zeros(env.n_actions)#stores the v-value for different actions
            ############Terminal state handling
            if env.states[s] == "Eaten" or env.states[s] == "Win":
                v[0] = env.rewards[s,0]#Irrelevant what action that is chosen
            elif t == horizon-1:
                for a in range(env.n_actions):
                    v[a] = env.rewards[s,a]#Don't need to include transition since the reward is same for all next states
            ###########
            else: 
                for a in range(env.n_actions):#How do we handle the actions (prob use policy(s))
                    v[a] += env.rewards[s,a]+np.dot(env.transition_probabilities[s,:,a],V[:,t+1])
            policy[s,t] = np.argmax(v)
            V[s,t] = np.max(v)

    return V, policy

def value_iteration(env, gamma=0.9, epsilon=0.01):#gamma being the discount 
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S
    """
    #TODO
    #Initialize 
    delta = 10
    policy = np.zeros(env.n_states)
    V = np.zeros(env.n_states) #The terminal state, time wise, will always be 0
    V_old = np.zeros(env.n_states)
    #DP to find the best policy in each state 
    while delta > epsilon*(1-gamma)/gamma:
        for s in range(env.n_states):
            V_old[s] = V[s]
            v = np.zeros(env.n_actions)#stores the v-value for different actions
            if env.states[s] == 'Eaten' or env.states[s] == 'Win' or env.states[s] == 'Poison':
                v[0] = env.rewards[s,a]
            else:
                for a in range(env.n_actions):#How do we handle the actions (prob use policy(s))
                    v[a] += env.rewards[s,a]+np.dot(env.transition_probabilities[s,:,a],gamma*V_old[:])
            V[s] = np.max(v)
            policy[s] = np.argmax(v)
        delta = np.linalg.norm(V-V_old)
        #print(delta)

    return V, policy



def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
        display.display(fig)
        time.sleep(0.1)
        display.clear_output(wait = True)



if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze
    
    start  = ((0,0), (6,5))
    
    env = Maze(maze) # Create an environment maze

    # Initialize variables for plotting win rate vs horizon
    wins = {}

    max_horizon = 30  # Assuming the maximum horizon is 30
    for h in range(max_horizon + 1):
        if h not in wins:
            wins[h] = 0

    # Solve the MDP problem with value iteration
    V, policy = value_iteration(env, gamma=0.9, epsilon=0.01)
    
    # Simulate and calculate win rates
    n_runs = 1000000
    for i in tqdm(range(n_runs), desc="Running simulations"):
        path = env.simulate(start, policy, 'ValIter')[0]
        if 'Win' in path:
            index = path.index('Win')
            if index not in wins:
                wins[index] = 0
            wins[index] += 1

    # Calculate regular and cumulative win rates
    horizons = sorted(list(wins.keys()))
    win_rates = [wins[i]/n_runs for i in horizons]
    cumulative_wins = [sum(wins[h] for h in horizons if h <= horizon) for horizon in horizons]
    cumulative_win_rates = [cw/n_runs for cw in cumulative_wins]

    print(wins)
    print(win_rates)
    print(cumulative_win_rates)
    
    threshold = 0.05
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Win Rate with Poison', fontsize=16, y=0.95)
    
    # Plot regular win rates
    ax1.plot(horizons, win_rates, marker='o', linestyle='-')
    ax1.set_xlabel('Horizon (T)')
    ax1.set_ylabel('Win Rate')
    ax1.set_ylim(0, 1)
    ax1.grid(True)
    
    prev_txt = 0
    for i, txt in enumerate(win_rates):
        if txt > 0 and abs(txt - prev_txt) > threshold:
            ax1.annotate(f'{txt*100:.2f}%', (horizons[i], win_rates[i]), textcoords="offset points", xytext=(0,10), ha='center')
            prev_txt = txt
    # for i, horizon in enumerate(horizons):
    #     if wins[horizon] > 0:
    #         ax1.annotate(f'{wins[horizon]}', (horizons[i], win_rates[i]), 
    #                     textcoords="offset points", xytext=(0,10), ha='center')

    # Plot cumulative win rates
    ax2.plot(horizons, cumulative_win_rates, marker='o', linestyle='-', color='green')
    ax2.set_xlabel('Horizon (T)')
    ax2.set_ylabel('Cumulative Win Rate')
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    # Annotate points with their values
    prev_txt = 0
    for i, txt in enumerate(cumulative_win_rates):
        if txt > 0 and abs(txt - prev_txt) > threshold or i == len(cumulative_win_rates) - 1: 
            if txt != 1:
                ax2.annotate(f'{txt*100:.1f}%', (horizons[i], cumulative_win_rates[i]), textcoords="offset points", xytext=(0,10), ha='center')
            else:
                ax2.annotate(f'{txt*100:.0f}%', (horizons[i], cumulative_win_rates[i]), textcoords="offset points", xytext=(0,10), ha='center')
            prev_txt = txt

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()
    

    #animate_solution(maze, path)