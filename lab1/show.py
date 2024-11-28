BLACK        = '#000000'
WHITE        = '#FFFFFF'

import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display


def show_solution(maze, path):
    # Map a color to each cell in the maze

    col_map = {
        0: WHITE,          # Empty cell
        1: BLACK,          # Wall
        2: '#00FF00',      # Bright green for exit
        -1: '#FF0000',     # Bright red for minotaur
        -2: '#0000FF'      # Bright blue for player
    }
    
    
    rows, cols = maze.shape
    fig = plt.figure(1, figsize=(cols, rows))
    
    ax = plt.gca()
    ax.set_title('Path taken')
    ax.set_xticks([])
    ax.set_yticks([])
    
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]
    
    grid = plt.table(
        cellText=None,
        cellColours=colored_maze,
        cellLoc='center',
        loc=(0,0),
        edges='closed'
    )
    
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    
    # Show all positions in the path
    for i, pos in enumerate(path):
        if pos != 'Eaten' and pos != 'Win':
            # Use different shades to show progression
            alpha = 0.3 + 0.7 * (i / len(path))  # Gradually increase opacity
            grid.get_celld()[pos[0]].set_facecolor(col_map[-2])
            grid.get_celld()[pos[1]].set_facecolor(col_map[-1])
            # Set alpha for both cells
            grid.get_celld()[pos[0]].set_alpha(alpha)
            grid.get_celld()[pos[1]].set_alpha(alpha)
    
    plt.show()

def show_solution(maze, path):
    # Map a color to each cell in the maze with stronger colors
    col_map = {
        0: WHITE,          # Empty cell
        1: BLACK,          # Wall
        2: '#00FF00',      # Bright green for exit
        -1: '#FF0000',     # Bright red for minotaur
        -2: '#0000FF'      # Bright blue for player
    }
    
    rows, cols = maze.shape
    fig = plt.figure(1, figsize=(cols + 2, rows))  # Made figure slightly wider for legend
    
    ax = plt.gca()
    ax.set_title('Path taken')
    ax.set_xticks([])
    ax.set_yticks([])
    
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]
    
    grid = plt.table(
        cellText=None,
        cellColours=colored_maze,
        cellLoc='center',
        loc=(0,0),
        edges='closed'
    )
    
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    
    # Find and mark the goal position
    goal_pos = None
    for i in range(rows):
        for j in range(cols):
            if maze[i,j] == 2:  # Exit cell
                goal_pos = (i,j)
                # Add "GOAL" text to the exit cell
                grid.get_celld()[(i,j)].get_text().set_text('GOAL')
                grid.get_celld()[(i,j)].get_text().set_color('black')
                grid.get_celld()[(i,j)].get_text().set_fontweight('bold')
    
    # Show all positions in the path
    for i, pos in enumerate(path):
        if pos == 'Win':
            # Mark winning position differently
            grid.get_celld()[goal_pos].set_facecolor('#FFD700')  # Gold color
            grid.get_celld()[goal_pos].get_text().set_text('WIN!')
        elif pos == 'Eaten':
            # Could mark the last position where player was eaten
            last_pos = path[i-1]
            grid.get_celld()[last_pos[0]].set_facecolor('#880000')  # Dark red
            grid.get_celld()[last_pos[0]].get_text().set_text('EATEN!')
        elif pos != 'Win':
            # Use different shades to show progression
            alpha = 0.3 + 0.7 * (i / len(path))  # Gradually increase opacity
            grid.get_celld()[pos[0]].set_facecolor(col_map[-2])
            grid.get_celld()[pos[1]].set_facecolor(col_map[-1])
            # Set alpha for both cells
            grid.get_celld()[pos[0]].set_alpha(alpha)
            grid.get_celld()[pos[1]].set_alpha(alpha)
            
            # Add numbers to show sequence
            grid.get_celld()[pos[0]].get_text().set_text(f'{i+1}')
            grid.get_celld()[pos[0]].get_text().set_fontsize(8)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[-2], label='Player'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[-1], label='Minotaur'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[2], label='Goal'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[1], label='Wall'),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.show()
    
def animate_solution(maze, path):
    # Map a color to each cell in the maze with stronger colors
    col_map = {
        0: WHITE,          # Empty cell
        1: BLACK,          # Wall
        2: '#00FF00',      # Bright green for exit
        -1: '#FF0000',     # Bright red for minotaur
        -2: '#0000FF'      # Bright blue for player
    }
    
    rows, cols = maze.shape
    fig = plt.figure(1, figsize=(cols + 2, rows))
    
    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Find goal position and mark it
    for i in range(rows):
        for j in range(cols):
            if maze[i,j] == 2:
                goal_pos = (i,j)

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[-2], label='Player'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[-1], label='Minotaur'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[2], label='Goal'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[1], label='Wall'),
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    for step in range(len(path)):
        # Clear previous frame
        ax.clear()
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Give a color to each cell
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]
        
        # Create a table to color
        grid = plt.table(
            cellText=None,
            cellColours=colored_maze,
            cellLoc='center',
            loc=(0,0),
            edges='closed'
        )
        
        # Modify the height and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0/rows)
            cell.set_width(1.0/cols)

        # Mark goal
        grid.get_celld()[goal_pos].get_text().set_text('GOAL')
        grid.get_celld()[goal_pos].get_text().set_color('black')
        grid.get_celld()[goal_pos].get_text().set_fontweight('bold')

        # Update current positions
        current = path[step]
        if current == 'Win':
            grid.get_celld()[goal_pos].set_facecolor('#FFD700')  # Gold
            grid.get_celld()[goal_pos].get_text().set_text('WIN!')
            title = f'Step {step}: Player Won!'
        elif current == 'Eaten':
            last_pos = path[step-1]
            grid.get_celld()[last_pos[0]].set_facecolor('#880000')  # Dark red
            grid.get_celld()[last_pos[0]].get_text().set_text('EATEN!')
            title = f'Step {step}: Player Eaten!'
        else:
            grid.get_celld()[current[0]].set_facecolor(col_map[-2])  # Player
            grid.get_celld()[current[1]].set_facecolor(col_map[-1])  # Minotaur
            title = f'Step {step}'
            
        ax.set_title(title)
        
        # Add legend again (since we cleared the axis)
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Display and pause
        display.display(fig)
        time.sleep(0.5)  # Pause for half a second
        display.clear_output(wait=True)
    
    # Show final frame for a bit longer
    display.display(fig)
    time.sleep(1)

def animate_solution(maze, path):
    # Map a color to each cell in the maze with stronger colors
    col_map = {
        0: WHITE,          # Empty cell
        1: BLACK,          # Wall
        2: '#00FF00',      # Bright green for exit
        -1: '#FF0000',     # Bright red for minotaur
        -2: '#0000FF'      # Bright blue for player
    }
    
    rows, cols = maze.shape
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(1, figsize=(cols + 2, rows))
    
    # Find goal position
    for i in range(rows):
        for j in range(cols):
            if maze[i,j] == 2:
                goal_pos = (i,j)

    # Add legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[-2], label='Player'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[-1], label='Minotaur'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[2], label='Goal'),
        plt.Rectangle((0,0), 1, 1, facecolor=col_map[1], label='Wall'),
    ]

    for step in range(len(path)):
        plt.clf()  # Clear the current figure
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Give a color to each cell
        colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]
        
        # Create a table to color
        grid = plt.table(
            cellText=None,
            cellColours=colored_maze,
            cellLoc='center',
            loc=(0,0),
            edges='closed'
        )
        
        # Modify the height and width of the cells in the table
        tc = grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0/rows)
            cell.set_width(1.0/cols)

        # Mark goal
        grid.get_celld()[goal_pos].get_text().set_text('GOAL')
        grid.get_celld()[goal_pos].get_text().set_color('black')
        grid.get_celld()[goal_pos].get_text().set_fontweight('bold')

        # Update current positions
        current = path[step]
        if current == 'Win':
            grid.get_celld()[goal_pos].set_facecolor('#FFD700')  # Gold
            grid.get_celld()[goal_pos].get_text().set_text('WIN!')
            title = f'Step {step}: Player Won!'
        elif current == 'Eaten':
            last_pos = path[step-1]
            grid.get_celld()[last_pos[0]].set_facecolor('#880000')  # Dark red
            grid.get_celld()[last_pos[0]].get_text().set_text('EATEN!')
            title = f'Step {step}: Player Eaten!'
        else:
            grid.get_celld()[current[0]].set_facecolor(col_map[-2])  # Player
            grid.get_celld()[current[1]].set_facecolor(col_map[-1])  # Minotaur
            title = f'Step {step}'
            
        ax.set_title(title)
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.draw()
        plt.pause(0.5)  # Pause for half a second
    
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show final state

if __name__ == "__main__":
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    
    path = [((0, 0), (6, 5)), ((0, 1), (6, 4)), ((1, 1), (6, 3)), ((2, 1), (6, 2)), ((3, 1), (6, 1)), ((4, 1), (6, 0)), ((4, 2), (6, 1)), ((4, 3), (6, 0)), ((4, 4), (6, 1)), ((4, 5), (6, 0)), ((4, 6), (6, 1)), ((4, 7), (6, 0)), ((5, 7), (6, 1)), ((6, 7), (6, 0)), ((6, 6), (6, 1)), 'Win', 'Win', 'Win', 'Win', 'Win']

    
    #show_solution(maze, path)
    animate_solution(maze, path)