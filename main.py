#import argparse
#from inspect import isfunction

import numpy as np
import random
import math
import time
import GeneticAlgorithm
import tkinter as tk
from sympy.strategies.core import switch

# def _build_parser():
#     parser = argparse.ArgumentParser(
#         description='visualize maze generation algorithms'
#     )
#     parser.add_argument('--algorithm', 
#         default='kruskal',
#         choices=[f for f in dir(generators) 
#                          if isfunction(getattr(generators, f))],
#         help='algorithm to use'
#     )
#     parser.add_argument('--speed',
#         type=int,
#         default=100,
#         metavar='ms',
#         help='visualization speed (milliseconds per edge)'
#     )
#     parser.add_argument('--width', 
#         type=int, 
#         default=10, 
#         metavar='columns',
#         help='maze width'
#     )
#     parser.add_argument('--height',
#         type=int,
#         default=10,
#         metavar='rows',
#         help='maze height'
#     )
#     parser.add_argument('--cellsize', 
#         type=int, 
#         default=50, 
#         metavar='pixels',
#         help='cell size'
#     )
#     return parser
def getPathFromVisited(matrix_visited, vertex_visited, vertex_visited_path):
    matrix_visited_path=np.copy(matrix_visited)    
   
    next_vertex = vertex_visited.pop()
    path = vertex_visited_path.pop()
    matrix_visited_path[next_vertex] = 3;    
    while True:
        x, y = next_vertex
        if path is 'North':
            next_vertex =(x+1, y) # checked
            matrix_visited_path[next_vertex] = 3;
            i = vertex_visited.index(next_vertex)
            path = vertex_visited_path[i]
            continue
        elif path is 'South':
            next_vertex =(x-1, y) # checked
            matrix_visited_path[next_vertex] = 3;
            i = vertex_visited.index(next_vertex)
            path = vertex_visited_path[i]
            continue
        elif path is 'East':
            next_vertex =(x, y-1) #checked
            matrix_visited_path[next_vertex] = 3;
            i = vertex_visited.index(next_vertex)
            path = vertex_visited_path[i]
            continue
        elif path is 'West':
            next_vertex =(x, y+1) # checked
            matrix_visited_path[next_vertex] = 3;
            i = vertex_visited.index(next_vertex)
            path = vertex_visited_path[i]
            continue
        elif path is 'None':
            # We have arrived at origin            
            matrix_visited_path[next_vertex] = 3;            
            return matrix_visited_path
        
    return

def drawCanvas(canvas, matrix, size):
    
    for (x,y), value in np.ndenumerate(matrix):
                  
        if value==1:
            rect(canvas, size, (x, y), (x+1, y+1),'black')
        if value == 0:
            rect(canvas, size, (x, y), (x+1, y+1),'white') 
        if value == 2:
            rect(canvas, size, (x, y), (x+1, y+1),'grey')
        if value == 3:
            rect(canvas, size, (x, y), (x+1, y+1),'blue')  
    #tk.Button(self.root, text="Quit", command=quit).pack()   
    #self.root.protocol("WM_DELETE_WINDOW", self.on_closing)   
    #self.root.mainloop()
   

def rect(canvas, size, a, b, color='black'):
    (x1, y1) = a
    (x2, y2) = b
    x1 *= size
    y1 *= size
    x2 *= size
    y2 *= size
    canvas.create_rectangle((x1, y1, x2, y2), fill=color)
    canvas.update_idletasks()


def buildMazeDict (maze_matrix):
    mazeH, mazeW= maze_matrix.shape
    #print (mazeH,mazeW)
    
    maze_dict={}
    
    for i in range (mazeH):
        for j in range (mazeW):
            if maze_matrix[i][j]==0:
                maze_dict[(i,j)]=[]
                               
    #print('For loop version')
    #print (maze_dict)
    
    for i in range (mazeH):
        for j in range (mazeW):
            if maze_matrix[i][j]==0:
                edgeN = ((i,j), 'North')
                edgeS = ((i+1,j), 'South')
                edgeW = ((i,j), 'West')
                edgeE = ((i,j+1), 'East')
                if i < mazeH-1 and maze_matrix[i+1][j]==0:
                    maze_dict[(i, j)].append(edgeS)
                    maze_dict[(i+1,j)].append(edgeN)
                if j < mazeW-1 and maze_matrix[i][j+1]==0:
                    maze_dict[(i, j)].append(edgeE)
                    maze_dict[(i, j+1)].append(edgeW)
            else:
                maze_dict[(i, j)] = []
                    
    #print('------DICT----------')
    #print (maze_dict)
    
    return maze_dict

def DFS(maze_dict,N, maze_matrix):
    print('------DFS---------')
    #start = time.time()
    maze_matrix_visited_dfs = np.copy(maze_matrix)
    origin= (0,0)
    visited, stack= list(), [(origin, 'None')]
    visited_path = list()
    #mazeH, mazeW= (maze.shape)
    #goal = (mazeH-1, mazeW-1)
    goal = (N-1, N-1)
   
    while stack:
        vertex, path = stack.pop()
        #print(vertex)
        
        if vertex not in visited:
            visited.append(vertex)
            visited_path.append(path)
            maze_matrix_visited_dfs[vertex]=2;
        else:
            continue
        if vertex == goal:
            print('Found the goal')
            print('Number of vertices visited: ' + str(len(visited)))
            #print ('Time upon completion: ' + str(time.time() - start) + ' seconds')
            return getPathFromVisited(maze_matrix_visited_dfs, visited, visited_path)
        #print (maze_dict[vertex])
        for children,path in maze_dict[vertex]:
            if children not in visited:
                stack.append( (children, path))
            
    print('Not Found')
    print('Number of vertices visited: ' + str(len(visited)))
    #print ('Time upon completion: ' + str(time.time() - start) + ' seconds')
    return maze_matrix_visited_dfs

def BFS(maze_dict,N, maze_matrix):
    print('------BFS---------')
    #start = time.time()
    maze_matrix_visited_bfs = np.copy(maze_matrix)
    origin= (0,0)
    visited, queue= list(), [(origin, 'None')]
    visited_path = list()
    #mazeH, mazeW= (maze.shape)
    #goal = (mazeH-1, mazeW-1)
    goal = (N-1, N-1)
   
    while queue:
        vertex, path = queue.pop(0)
        #print(vertex)
        
        if vertex not in visited:
            visited.append(vertex)
            visited_path.append(path)
            maze_matrix_visited_bfs[vertex]=2;
        else:
            continue
        if vertex == goal:
            print('Found the goal')
            print('Number of vertices visited: ' + str(len(visited)))
            #print ('Time upon completion: ' + str(time.time() - start) + ' seconds')
            return getPathFromVisited(maze_matrix_visited_bfs, visited, visited_path)
        #print (maze_dict[vertex])
        for children, path in maze_dict[vertex]:
            if children not in visited:
                queue.append( (children, path))
            
    print('Not Found')      
    print('Number of vertices visited: ' + str(len(visited)))
    #print ('Time upon completion: ' + str(time.time() - start) + ' seconds')
    return maze_matrix_visited_bfs

def euclidean(a,b):
    a_x, a_y = a;
    b_x, b_y = b;

    return math.sqrt((a_x-b_x)*(a_x-b_x) + (a_y-b_y)*(a_y-b_y)) 

def manhattan(a,b):
    a_x, a_y = a;
    b_x, b_y = b;

    return math.fabs(a_x-b_x) + math.fabs(a_y-b_y)

def Astar(maze_dict,N, maze_matrix, method):
    print('------A* with ' + method + '---------')
    maze_matrix_visited_astar = np.copy(maze_matrix)
    visited = list()
    goal = (N-1, N-1)
    
    origin= (0,0)
    visited_path = list()
    
    if method is 'euclidean':
        queue =  [(origin,'None',0,euclidean(origin,goal))]
    elif method is 'manhattan':   
        queue =  [(origin,'None',0,manhattan(origin, goal))]
    else:
        return -1
    
    
    while queue:
        vertex, path, true_cost, total_cost = min(queue, key = lambda t: t[3])
        queue.remove((vertex, path, true_cost,total_cost))
        #print(vertex)
        
        if vertex not in visited:
            visited.append(vertex)
            visited_path.append(path)
            maze_matrix_visited_astar[vertex]=2;
        else:
            continue
        if vertex == goal:
            print('Found the goal')
            print('Number of vertices visited: ' + str(len(visited)))
            #print ('Time upon completion: ' + str(time.time() - start) + ' seconds')
            return getPathFromVisited(maze_matrix_visited_astar, visited, visited_path)
        # print (maze_dict[vertex])
        for children, path in maze_dict[vertex]:
            if children not in visited:
                if method is 'euclidean':
                    queue.append( (children, path, true_cost+1, true_cost+1+euclidean(children, goal)))
                else:   
                    queue.append( (children, path, true_cost+1, true_cost+1+manhattan(children, goal)))
                
            
    print('Not Found')
    print('Number of vertices visited: ' + str(len(visited)))
    #print ('Time upon completion of A* with ' + method + ': ' + str(time.time() - start) + ' seconds')        
    return maze_matrix_visited_astar

def RunGeneticAlgorithm(dimension=20,numberOfMazesInAPopulation=500,probabilityObstruction=0.4,dfs_percentileVisitedNodeLevel=80,bfs_percentileVisitedNodeLevel=20,astarE_percentileLevel=50,astarM_percentileLevel=50,retryGeneratingHardMazeInMinutes=1):
    N = dimension # dimension
    #listofprobability = np.arange(0.4, 1.0, 0.1) # probability from 0.2 to 0.9
    percentileLevel= dfs_percentileVisitedNodeLevel #80
    percentilelevelbfs= bfs_percentileVisitedNodeLevel#20 # has to be lower because for 75 it doesnt find anything
    percentilelevelAstarE= astarE_percentileLevel
    percentilelevelAstarM= astarM_percentileLevel
    mazePopulation = []
    bfsMazePopulation = []
    astarEMazePopulation=[]
    astarMMazePopulation=[]
    bfsmazesProbabiltyOfVisitedNodes = []
    mazesProbabiltyOfVisitedNodes = []
    astaremazesProbabiltyOfVisitedNodes=[]
    astarmmazesProbabiltyOfVisitedNodes =[]
    mazeNumberOfVisitedNodes = []
    bfsmazeNumberOfVisitedNodes = []
    astarEmazeNumberOfVisitedNodes  =[]
    astarMmazeNumberOfVisitedNodes =[]
    mazeNumberOfObstacleNodes=[]
    bfsmazeNumberOfObstacleNodes=[]
    astaremazeNumberOfObstacleNodes =[]
    astarMmazeNumberOfObstacleNodes =[]
    #for i in listofprobability:
    for j in range(0, numberOfMazesInAPopulation):
        print("Maze #" + str(j + 1) + ' generated for ' + str(probabilityObstruction) + ' probability')
        print("---------------------------------------------------------")
        maze_matrix = np.zeros((N, N))
        noOfOccupiedBlocks = int(np.floor(N * N * probabilityObstruction) - 2)
        if noOfOccupiedBlocks > 0:
            occ_Indices = random.sample(range(1, (N * N) - 1), noOfOccupiedBlocks)
            occ_Indices = np.unravel_index(occ_Indices, (N, N))
            maze_matrix[occ_Indices] = 1

        print(maze_matrix)
        maze_dict = buildMazeDict((maze_matrix))
        maze_matrix_visited_dfs = DFS(maze_dict, N, maze_matrix)
        maze_matrix_visited_bfs = BFS(maze_dict, N, maze_matrix)
        maze_matrix_visited_astarE = Astar(maze_dict, N, maze_matrix, 'euclidean')
        maze_matrix_visited_astarM = Astar(maze_dict, N, maze_matrix, 'manhattan')


        if (maze_matrix_visited_dfs[N - 1, N - 1] == 3):
            mazePopulation.append(maze_dict)
            unique, counts = np.unique(maze_matrix_visited_dfs, return_counts=True)
            mazestatus = dict(zip(unique, counts))
            numberofvisitednodes = mazestatus[3]
            mazeNumberOfObstacleNodes.append(mazestatus[1])
            mazeNumberOfVisitedNodes.append(numberofvisitednodes)

        if (maze_matrix_visited_bfs[N - 1, N - 1] == 3):
            bfsMazePopulation.append(maze_dict)
            unique, counts = np.unique(maze_matrix_visited_bfs, return_counts=True)
            mazestatus = dict(zip(unique, counts))
            numberofvisitednodes = mazestatus[3]
            bfsmazeNumberOfObstacleNodes.append(mazestatus[1])
            bfsmazeNumberOfVisitedNodes.append(numberofvisitednodes)

        if (maze_matrix_visited_astarE[N - 1, N - 1] == 3):
            astarEMazePopulation.append(maze_dict)
            unique, counts = np.unique(maze_matrix_visited_astarE, return_counts=True)
            mazestatus = dict(zip(unique, counts))
            numberofvisitednodes = mazestatus[3]
            astaremazeNumberOfObstacleNodes.append(mazestatus[1])
            astarEmazeNumberOfVisitedNodes.append(numberofvisitednodes)
        if (maze_matrix_visited_astarM[N - 1, N - 1] == 3):
            astarMMazePopulation.append(maze_dict)
            unique, counts = np.unique(maze_matrix_visited_astarM, return_counts=True)
            mazestatus = dict(zip(unique, counts))
            numberofvisitednodes = mazestatus[3]
            astarMmazeNumberOfObstacleNodes.append(mazestatus[1])
            astarMmazeNumberOfVisitedNodes.append(numberofvisitednodes)


    totalNumberVisitedNodes = sum(mazeNumberOfVisitedNodes)
    bfstotalNumberVisitedNodes = sum(bfsmazeNumberOfVisitedNodes)
    astaretotalNumberVisitedNodes = sum(astarEmazeNumberOfVisitedNodes)
    astarMtotalNumberVisitedNodes = sum(astarMmazeNumberOfVisitedNodes)
    totalProbabilty = 0.0
    for i in range(len(mazePopulation)):
        previousProbabiltiy = totalProbabilty
        #get the probability distribution for each maze
        totalProbabilty = totalProbabilty + (mazeNumberOfVisitedNodes[i] / totalNumberVisitedNodes)
        mazesProbabiltyOfVisitedNodes.append((previousProbabiltiy, totalProbabilty))
    totalProbabilty = 0.0
    for i in range(len(bfsMazePopulation)):
        previousProbabiltiy = totalProbabilty
        # get the probability distribution for each maze
        totalProbabilty = totalProbabilty + (bfsmazeNumberOfVisitedNodes[i] / bfstotalNumberVisitedNodes)
        bfsmazesProbabiltyOfVisitedNodes.append((previousProbabiltiy, totalProbabilty))
    totalProbabilty = 0.0
    for i in range(len(astarEMazePopulation)):
        previousProbabiltiy = totalProbabilty
        #get the probability distribution for each maze
        totalProbabilty = totalProbabilty + (astarEmazeNumberOfVisitedNodes[i] / astaretotalNumberVisitedNodes)
        astaremazesProbabiltyOfVisitedNodes.append((previousProbabiltiy, totalProbabilty))
    totalProbabilty = 0.0
    for i in range(len(astarMMazePopulation)):
        previousProbabiltiy = totalProbabilty
        #get the probability distribution for each maze
        totalProbabilty = totalProbabilty + (astarMmazeNumberOfVisitedNodes[i] / astarMtotalNumberVisitedNodes)
        astarmmazesProbabiltyOfVisitedNodes.append((previousProbabiltiy, totalProbabilty))


    if(len(mazePopulation)==0):
        print("No Solvable Maze for",probabilityObstruction,"probability")
        return False
    timeToRetry = 60 * (retryGeneratingHardMazeInMinutes)
    print(len(mazeNumberOfVisitedNodes) )
    percentiledfs = np.percentile(mazeNumberOfVisitedNodes, percentileLevel)
    percentilebfs = np.percentile(bfsmazeNumberOfVisitedNodes, percentilelevelbfs)
    percentileastare = np.percentile(bfsmazeNumberOfVisitedNodes, percentilelevelAstarE)
    percentileastarm = np.percentile(bfsmazeNumberOfVisitedNodes, percentilelevelAstarM)
    averageObstacle=sum(mazeNumberOfObstacleNodes)/float(len(mazeNumberOfObstacleNodes))
    bfsaverageObstacle=sum(bfsmazeNumberOfObstacleNodes)/float(len(bfsmazeNumberOfObstacleNodes))
    astareaverageObstacle=sum(astaremazeNumberOfObstacleNodes)/float(len(astaremazeNumberOfObstacleNodes))
    astarmsaverageObstacle=sum(astarMmazeNumberOfObstacleNodes)/float(len(astarMmazeNumberOfObstacleNodes))

    hardest_maze_so_far,dfs_hardest_maze_matrix_so_far,max_visited_nodes_so_far, latest_percentile, latest_averageObstacle= GeneticAlgorithm.GeneticAlgorithm(mazePopulation, timeToRetry, mazesProbabiltyOfVisitedNodes,percentiledfs, percentileLevel,averageObstacle,DFS, "",N)
    bfs_hardest_maze_so_far, bfs_hardest_maze_matrix_so_far ,bfs_max_visited_nodes_so_far, bfs_latest_percentile, bfs_latest_averageObstacle =GeneticAlgorithm.GeneticAlgorithm(bfsMazePopulation, timeToRetry, bfsmazesProbabiltyOfVisitedNodes,percentilebfs, percentilelevelbfs,bfsaverageObstacle,BFS,"", N)
    astare_hardest_maze_so_far, astare_hardest_maze_matrix_so_far ,astare_max_visited_nodes_so_far, astare_latest_percentile, astare_latest_averageObstacle =GeneticAlgorithm.GeneticAlgorithm(astarEMazePopulation, timeToRetry, astaremazesProbabiltyOfVisitedNodes,percentileastare, percentilelevelAstarE,astareaverageObstacle,Astar,"euclidean", N)
    astarm_hardest_maze_so_far, astarm_hardest_maze_matrix_so_far ,astarm_max_visited_nodes_so_far, astarm_latest_percentile, astarm_latest_averageObstacle =GeneticAlgorithm.GeneticAlgorithm(astarMMazePopulation, timeToRetry, astarmmazesProbabiltyOfVisitedNodes,percentileastarm, percentilelevelAstarM,astarmsaverageObstacle,Astar,"manhattan", N)

    #max visited node means longest path
    print("DSF Before Run GA Average Obstacle : ", averageObstacle)
    print("DFS Before Run GA Percentile Longest Path : ", percentiledfs)
    print("DFS Maximum Longest Path: ", max_visited_nodes_so_far)
    print("DSF Average Obstacle : ", latest_averageObstacle)
    print("DFS Latest Percentile Longest Path : ", latest_percentile)
    print("DFS Hardest Maze")
    print("-----------------")
    print(dfs_hardest_maze_matrix_so_far)
    print("BFS Before Run GA Average Obstacle : ", bfsaverageObstacle)
    print("BFS Before Run GA Percentile Longest Path : ", percentilebfs)
    print("BFS Maximum Longest Path : ", bfs_max_visited_nodes_so_far)
    print("BFS Average Obstacle : ", bfs_latest_averageObstacle)
    print("BFS Latest Percentile Longest Path: ", bfs_latest_percentile)
    print("BFS Hardest Maze")
    print("-----------------")
    print(bfs_hardest_maze_matrix_so_far)

    print("A Star Euclidean Before Run GA Average Obstacle : ", astareaverageObstacle)
    print("A Star Euclidean Before Run GA Percentile Longest Path : ", percentileastare)
    print("A Star Euclidean Maximum Longest Path : ", astare_max_visited_nodes_so_far)
    print("A Star Euclidean Average Obstacle : ", astare_latest_averageObstacle)
    print("A Star Euclidean Latest Percentile Longest Path: ", astare_latest_percentile)
    print("A Star Euclidean Hardest Maze")
    print("------------------------------")
    print(astare_hardest_maze_matrix_so_far)
    print("A Star Manhattan Before Run GA Average Obstacle : ", astarmsaverageObstacle)
    print("A Star Manhattan Before Run GA Percentile Longest Path : ", percentileastarm)
    print("A Star Manhattan Maximum Longest Path : ", astarm_max_visited_nodes_so_far)
    print("A Star Manhattan Average Obstacle : ", astarm_latest_averageObstacle)
    print("A Star Manhattan Latest Percentile Longest Path: ", astarm_latest_percentile)
    print("A Star Manhattan Hardest Maze")
    print("-----------------------------")
    print(astarm_hardest_maze_matrix_so_far)

    return dfs_hardest_maze_matrix_so_far, bfs_hardest_maze_matrix_so_far, astare_hardest_maze_matrix_so_far, astarm_hardest_maze_matrix_so_far


if __name__ == '__main__':
    #args = _build_parser().parse_args()
    dim = int(input ('Enter the size of Maze: ')); # dimension of maze, e.g 4x4
    cellsize = (int) (500/dim); # dimension of cells (for visualization)
    typeAlg = ''

    while (typeAlg != 'E' or typeAlg != 'D' or typeAlg != 'S'):
        typeAlg = input('Do you want to generate and solve an easy maze or a difficult maze (E or D), or do you want to calculate the easy solvability (S)? ')
        if typeAlg == 'E':
            prob_occ = float(input ('Enter the probability of an obstructed cell (ideally between 0.2 and 0.3): ')); # probability that a cell is occupied (i.e. a wall)
            start = time.time()
            maze_matrix = np.zeros((dim,dim));
            #print(maze_matrix)
            
            noOfOccupiedBlocks = int(np.floor(dim*dim*prob_occ)-2); # (-2) if we don't count source and goal
            if noOfOccupiedBlocks>0:
                occ_Indices = random.sample(range(1,(dim*dim)-1), noOfOccupiedBlocks)
                occ_Indices = np.unravel_index(occ_Indices, (dim, dim))
                maze_matrix[occ_Indices]=1
            
            
            print(maze_matrix)
            
            maze_dict = buildMazeDict((maze_matrix))
            
            maze_matrix_visited_dfs = DFS(maze_dict, dim, maze_matrix)            
            print(maze_matrix_visited_dfs)
            
            maze_matrix_visited_bfs = BFS(maze_dict, dim, maze_matrix)            
            print(maze_matrix_visited_bfs)
            
            maze_matrix_visited_astarE = Astar(maze_dict, dim, maze_matrix, 'euclidean')
            print(maze_matrix_visited_astarE)

        
            maze_matrix_visited_astarM = Astar(maze_dict, dim, maze_matrix, 'manhattan')
            print(maze_matrix_visited_astarM)
        
            print ('Time taken to complete: ' + str(time.time() - start) + ' seconds')
            visuals = input("Do you want a visualization (Y or N)? ");
            if visuals == 'Y':
                master = tk.Tk();               
                master.title( "DFS" )
                gif1 = tk.PhotoImage(file='/Users/ingridwijaya/Desktop/Intro to AI/CS520_Assignment1/Maze.gif')

                canvasDFS = tk.Canvas(master, width=dim*cellsize, height=dim*cellsize)
                canvasDFS.create_image(50, 10, image=gif1, anchor='nw')
                canvasDFS.grid(row=0, column=0)
                drawCanvas(canvasDFS, maze_matrix_visited_dfs, cellsize)
                
                windowBFS = tk.Toplevel()
                windowBFS.title("BFS")                    
                canvasBFS = tk.Canvas(windowBFS, width=dim*cellsize, height=dim*cellsize)
                canvasBFS.grid(row=0, column=0)
                drawCanvas(canvasBFS, maze_matrix_visited_bfs, cellsize)
                
                windowAstarE = tk.Toplevel()
                windowAstarE.title("A* Euclidean")                    
                canvasAstarE = tk.Canvas(windowAstarE, width=dim*cellsize, height=dim*cellsize)
                canvasAstarE.grid(row=0, column=0)
                drawCanvas(canvasAstarE, maze_matrix_visited_astarE, cellsize)
                
                windowAstarM = tk.Toplevel()
                windowAstarM.title("A* Manhattan")                    
                canvasAstarM = tk.Canvas(windowAstarM, width=dim*cellsize, height=dim*cellsize)
                canvasAstarM.grid(row=0, column=0)
                drawCanvas(canvasAstarM, maze_matrix_visited_astarM, cellsize)
                
                master.mainloop();            
            
            break;
        elif typeAlg == 'D':
            numberOfMazesInAPopulation = 0
            while(numberOfMazesInAPopulation < 100):
                numberOfMazesInAPopulation = int(input("Enter the number of mazes to generate for population (must be more than 100): "))
            probabilityObstruction = float(input ('Enter the probability of an obstructed cell (ideally between 0.2 and 0.3): ')); # probability that a cell is occupied (i.e. a wall)
            start = time.time()
            dfs_percentileVisitedNodeLevel = 80
            bfs_percentileVisitedNodeLevel = 20
            retryGeneratingHardMazeInMinutes = 1
            astarE_percentileLevel=50
            astarM_percentileLevel=50
            maze_matrix_visited_dfs, maze_matrix_visited_bfs, maze_matrix_visited_astare, maze_matrix_visited_astarm = RunGeneticAlgorithm(dim, numberOfMazesInAPopulation, probabilityObstruction, 80, 30,astarE_percentileLevel,astarM_percentileLevel, retryGeneratingHardMazeInMinutes)

            print ('Time taken to complete: ' + str(time.time() - start) + ' seconds')
            visuals = input("Do you want a visualization (Y or N)? ");
            if visuals == 'Y':
                master = tk.Tk();               
                master.title( "DFS Hardest Maze" )  
                canvasDFS = tk.Canvas(master, width=dim*cellsize, height=dim*cellsize)
                canvasDFS.grid(row=0, column=0)
                drawCanvas(canvasDFS, maze_matrix_visited_dfs, cellsize)
                windowBFSHardest = tk.Toplevel()
                windowBFSHardest.title("BFS Hardest maze")                    
                canvasBFSHardest = tk.Canvas(windowBFSHardest, width=dim*cellsize, height=dim*cellsize)
                canvasBFSHardest.grid(row=0, column=0)
                drawCanvas(canvasBFSHardest, maze_matrix_visited_bfs, cellsize)
                windowAstarEHardest = tk.Toplevel()
                windowAstarEHardest.title("A* Euclidean Hardest maze")                    
                canvasAstarEHardest = tk.Canvas(windowAstarEHardest, width=dim*cellsize, height=dim*cellsize)
                canvasAstarEHardest.grid(row=0, column=0)
                drawCanvas(canvasAstarEHardest, maze_matrix_visited_astare, cellsize)
                windowAstarMHardest = tk.Toplevel()
                windowAstarMHardest.title("A* Manhattan Hardest maze")                    
                canvasAstarMHardest = tk.Canvas(windowAstarMHardest, width=dim*cellsize, height=dim*cellsize)
                canvasAstarMHardest.grid(row=0, column=0)
                drawCanvas(canvasAstarMHardest, maze_matrix_visited_astarm, cellsize)
                master.mainloop(); 
            break;
        elif typeAlg == 'S':
            prob_occ = float(input ('Enter the probability of an obstructed cell (ideally between 0.2 and 0.3): ')); # probability that a cell is occupied (i.e. a wall)
            numRepeats = 100
            numSolvable = 0
            start = time.time()
            for i in range (0, numRepeats):
                print("----------------Maze #" + str(i + 1) + "----------------")
                maze_matrix = np.zeros((dim,dim));
                #print(maze_matrix)
                
                noOfOccupiedBlocks = int(np.floor(dim*dim*prob_occ)-2); # (-2) if we don't count source and goal
                if noOfOccupiedBlocks>0:
                    occ_Indices = random.sample(range(1,(dim*dim)-1), noOfOccupiedBlocks)
                    occ_Indices = np.unravel_index(occ_Indices, (dim, dim))
                    maze_matrix[occ_Indices]=1
                
                maze_dict = buildMazeDict((maze_matrix))
                
                maze_matrix_visited_dfs = DFS(maze_dict, dim, maze_matrix) #fastest method         
                if (maze_matrix_visited_dfs[dim - 1, dim - 1] == 3):
                    numSolvable += 1
            print("Probability of solvability for p = " + str(prob_occ) + ": " + str((numSolvable / numRepeats)*100) + "%")
            print ('Time taken to complete: ' + str(time.time() - start) + ' seconds')
            break