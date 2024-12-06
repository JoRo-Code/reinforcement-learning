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
    POISON_REWARD = 0         #TODO

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        #self.transition_probabilities = self.__transitions()
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
                        for bool in range(2):#Adds a boolean to the state to represent the key
                            if self.maze[i,j] != 1:#Only if the agents coordinates are in a wall we don't create a state
                                states[s] = ((i,j), (k,l), bool)
                                map[((i,j), (k,l), bool)] = s
                                s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1

        states[s] = 'Poison'
        map['Poison'] = s
        s += 1
        
        states[s] = 'Win'
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
        elif p <= 1/50:#Don't think it matters in what order minotaur or poison
            return ['Poison']
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position (int)
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position (int)
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            impossible_action_player = ((row_player,col_player),(6,5),0) in self.map  #TODO

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
                    if ((self.states[state][0][0], self.states[state][0][1]) == (0,7) and self.states[state][2] == 0) or self.states[state][2] == 1:#Pick up the key
                        key = 1
                    else:
                        key = 0

                    if  (self.states[state][0][0], self.states[state][0][1]) == (rows_minotaur[i], cols_minotaur[i]):# TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.states[state][0][0], self.states[state][0][1]) == (6,5) and self.states[state][2] == 1:# TODO: We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else:     # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i]),key))
            
                
                return states
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                    if ((row_player,col_player) == (0,7) and self.states[state][2] == 0) or self.states[state][2] == 1:#Pick up the key
                        key = 1    
                    else:
                        key = 0
                               
                    if  (row_player,col_player)==(rows_minotaur[i], cols_minotaur[i]):# TODO: We met the minotaur
                        states.append('Eaten')
                    
                    elif (row_player,col_player) == (6,5) and self.states[state][2] == 1:# TODO:We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i]),key))
                        
                return states
    

    def bestMino(self,next_states):
        """ Returns the next state that is best for the minotaur's goal to eat the agent 
        """ 
        if len(next_states) == 1:#One possible next state--> choose that action
            return next_states[0]
        else:
            distance = []#Keeps track of all distances to the agent
            for s_next in next_states:
                if s_next == 'Eaten':#If a next state is Eaten then that is the best action for the minotaur
                    return s_next
                elif s_next == 'Win':#If the next state is win with 0% chance of Eaten then it doesn't matter
                    if 'Eaten' not in next_states:
                        return s_next
                else:
                    distance.append(np.linalg.norm([s_next[0][0]-s_next[1][0],s_next[0][1]-s_next[1][1]]))
            return next_states[np.argmin(distance)]#If the next state isn't terminal we choose the next state with the shortest distance to the agent



    def __rewards(self):#Perhaps want to add a reward for picking the key up
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has died
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Poison':
                    rewards[s, a] = self.POISON_REWARD
                
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

    def SARSA(self, start, alpha, gamma, epsilon):
        """Trains the Q-network and returns the trained Q-table for SARSA
        """
        #Initialize Q-table
        Q = np.random.rand(self.n_states,self.n_actions)
        #Q = np.zeros((self.n_states,self.n_actions))
        nsa = np.zeros((self.n_states,self.n_actions))

        ######End conditions
        terminal = ['Poison','Win','Eaten']
        episodes_N = 50000
        initial = np.zeros(episodes_N)
        for state in terminal:
            s_in = self.map[state]
            Q[s_in,:] = 0#self.rewards[s_in,:]
        ######
        for i in tqdm(range(episodes_N), desc=f"Training SARSA - epsilon = {epsilon}"):
            s = self.map[start]
            a = epsilon_greedy(Q, s, epsilon)

            while self.states[s] not in terminal:

                next_states = self.__move(s, a)

                ####Minotaur action
                if random.random() <= 0.35:
                    next_s = self.bestMino(next_states)
                else:
                    next_s = random.choice(next_states)
                ####
                s_n = self.map[next_s]
                a_n = epsilon_greedy(Q, s_n, epsilon)

                # Q = q_update(Q,s,a,env.rewards[s,a],s_n,a_n,learning_rate(nsa[s,a],alpha),gamma)


                # SARSA update (uses Q[s_n, a_n] instead of max(Q[s_n,:]))
                Q[s,a] = Q[s,a] + learning_rate(nsa[s,a],alpha) * (
                    self.rewards[s,a] + gamma * Q[s_n,a_n] - Q[s,a]
                )
                nsa[s,a] += 1
                # nsa[s_n,a_n] += 1
                #Prepare for next iternation 
                s = s_n
                a = a_n
            initial[i] = (1-epsilon)*np.max(Q[self.map[start],:])+(epsilon/self.n_actions)*np.sum(Q[self.map[start],:])#Computed the value function based on Q
            for a in range(self.n_actions):   #Handles the Q-value for terminal 
                Q[s,a] = Q[s,a] +learning_rate(nsa[s,a],alpha)*(env.rewards[s,a]-Q[s,a]) #Since Q(s',a'_best)=0
                nsa[s,a] += 1
        return [Q,initial]


    def simulate(self, start, Q, method):

        path = list()
        terminal = ['Poison','Win','Eaten']
        reward = 0
        
        s = self.map[start]
        path.append(start) # Add the starting position in the maze to the path
        while self.states[s] not in terminal:
            a = np.argmax(Q[s,:])
            reward += self.rewards[s,a]
            next_states = self.__move(s, a)

            ####Minotaur action
            if random.random() <= 0.35:
                next_s = self.bestMino(next_states)
            else:
                next_s = random.choice(next_states)
            #Prepare for next iternation 
            path.append(next_s)
            s = self.map[next_s]
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




###For the Q-learning###
def epsilon_greedy(Q, s, epsilon):
    """Returns the action that the agent takes based on epsilon greedy policy
    """
    if random.random() <= epsilon:
        action = random.randint(0,4)#One for each action alternative
    else:
        action = np.argmax(Q[s,:])
    return action

def q_update(Q, s, a, r, s_next, a_next, learning_rate, gamma):
    """Updates the Q-value based on the latest transition
    """
    Q[s,a] = Q[s,a] + learning_rate*(r+gamma*Q[s_next,a_next]-Q[s,a]) 
    return Q

def learning_rate(times_visited, alpha):
    "Handles the learning rate"
    if times_visited == 0:
        lr = 1
    else: 
        lr = 1/(times_visited**(alpha))
    return lr

########################
    
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
    
    env = Maze(maze) # Create an environment maze
    start  = ((0,0), (6,5), 0)
    #Hyper parameters
    alpha = 2/3
    gamma = 1
    epsilon = 0.1

    [Q,initial] = env.SARSA(start,alpha,gamma,epsilon)
    # Simulate the shortest path starting from position A
    method = 'Q-learning'
    
    [Q2,initial2] = env.SARSA(start,alpha,gamma,epsilon*2)

    #path = env.simulate(start, Q, method)
    #print(path)

    sum = 0 
    poison = 0
    eaten = 0
    n_runs = 100000
    for i in tqdm(range(n_runs), desc="Running simulations"):
        path = env.simulate(start, Q, method)
        if 'Win' in path:
            sum += 1
        elif 'Poison' in path:
            poison += 1
        else:
            eaten += 1
        
    print("Poison:",poison/n_runs)   
    print("Eaten:",eaten/n_runs)  
    print("The probability of winning:",sum/n_runs)
    
    sum = 0 
    poison = 0
    eaten = 0
    for i in tqdm(range(n_runs), desc="Running simulations"):
        path = env.simulate(start, Q2, method)
        if 'Win' in path:
            sum += 1
        elif 'Poison' in path:
            poison += 1
        else:
            eaten += 1
    print("Poison:",poison/n_runs)   
    print("Eaten:",eaten/n_runs)  
    print("The probability of winning:",sum/n_runs)
    # Create the Plot
    #animate_solution(maze, path)
    x = np.arange(1, 50001)  # Episodes (1 to 50000)
    plt.figure(figsize=(10, 6))
    plt.plot(x, initial, label='epsilon=0.1', color='b', linewidth=1)
    plt.plot(x, initial2, label='epsilon=0.2', color='r', linewidth=1)  
    # Add Labels and Title
    plt.xlabel('Number of Episodes', fontsize=12)
    plt.ylabel('Value Function', fontsize=12)
    plt.title('Value Function over Episodes', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Show the Plot
    plt.tight_layout()
    plt.show()