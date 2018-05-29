import random
import time
import numpy as np
import collections


def GeneticAlgorithm(mazePopulation,max_time,mazeProbability,percentile,percentileLevel,averageObstacle,searchFunction,astarType,dimension):
    #maze population is a set of mazes
    repeatTheLoop=True
    start_time = time.time()
    max_visited_nodes_so_far=0
    hardest_maze_so_far=None
    hardest_maze_matrix_so_far=None
    #python has issued with array assignmetn
    #recommit the whole thing
    currentMazePopulation=[]+mazePopulation
    while(repeatTheLoop):
        new_maze_population= []
        mazeNumberOfVisitedNodes = []
        mazeNumberOfObstacles=[]
        mazeMatrixList=[]
        for i in range(1,len(currentMazePopulation)):
            fathermaze = RandomMazeSelection(currentMazePopulation,mazeProbability)
            mothermaze = RandomMazeSelection(currentMazePopulation,mazeProbability)
            childmaze = ReproduceMaze(fathermaze,mothermaze)
            childmaze = MutateMaze(childmaze)
            #try to find the goal with obstacle higher than average and number of nodes is higher than percentile 
            numberofvisitednodes,numberofobstacle,maze_matrix_visited=SolveMaze(childmaze,searchFunction,dimension,percentile,averageObstacle,astarType)
            if(numberofvisitednodes>0):
                new_maze_population.append(childmaze)
                mazeNumberOfVisitedNodes.append(numberofvisitednodes)
                mazeNumberOfObstacles.append(numberofobstacle)
                mazeMatrixList.append(maze_matrix_visited)

        currentMazePopulation.clear()
        for newmaze in new_maze_population:
            currentMazePopulation.append(newmaze)

        if(len(currentMazePopulation)==0):
            #if there is no more maze to be merge and mutate
            #return the hardest maze
            return hardest_maze_so_far,hardest_maze_matrix_so_far,max_visited_nodes_so_far,percentile,averageObstacle

        #maximum visite nodes is longest path , the variable name is mistakenly name maximum visited nodes
        maximumVisitedNodes = max(mazeNumberOfVisitedNodes)
        mazeIndex=[i for i, j in enumerate(mazeNumberOfVisitedNodes) if j == maximumVisitedNodes]

        if(max_visited_nodes_so_far==0):
            max_visited_nodes_so_far=maximumVisitedNodes
            hardest_maze_so_far=currentMazePopulation[mazeIndex[0]]
            hardest_maze_matrix_so_far = mazeMatrixList[mazeIndex[0]]
        elif(max_visited_nodes_so_far<maximumVisitedNodes):
            print("Replace Hardest Maze")
            max_visited_nodes_so_far=maximumVisitedNodes
            hardest_maze_so_far=currentMazePopulation[mazeIndex[0]]
            hardest_maze_matrix_so_far=mazeMatrixList[mazeIndex[0]]

        mazeProbability= RecalculateMazeProbabiliy(mazeNumberOfVisitedNodes)
        averageObstacle=sum(mazeNumberOfObstacles)/float(len(mazeNumberOfObstacles))
        #print(len(mazeNumberOfVisitedNodes))
        percentile= np.percentile(mazeNumberOfVisitedNodes, percentileLevel)
        # print("Time to complete: ",time.time() - start_time)
        currentMazeLen=len(currentMazePopulation)

        repeatTheLoop =((time.time() - start_time) < max_time) and currentMazeLen >= 10

        mazeMatrixList.clear()
        # print("Repeat Loop: ",repeatTheLoop)
    # print("DONE")
    return hardest_maze_so_far,hardest_maze_matrix_so_far,max_visited_nodes_so_far,percentile,averageObstacle


def SolveMaze(childmaze,searchFunction,N,percentile,averageObstacle,astarType):
    maze_matrix = np.zeros((N, N))
    numberofvisitednodes=0
    numberofobstacle=0
    for cell in childmaze:
        if childmaze[cell] == []:
            maze_matrix[cell[0]][cell[1]]=1
            numberofobstacle=numberofobstacle+1
    if(astarType=="euclidean"):
        maze_matrix_visited_dfs = searchFunction(childmaze, N, maze_matrix,astarType)
    elif(astarType=="manhattan"):
        maze_matrix_visited_dfs = searchFunction(childmaze, N, maze_matrix,astarType)
    else:
        print(astarType)
        maze_matrix_visited_dfs = searchFunction(childmaze, N, maze_matrix)

    if(maze_matrix_visited_dfs[N-1,N-1]!=3):
        print("don't add unsolved maze")
    else:
        unique, counts = np.unique(maze_matrix_visited_dfs, return_counts=True)
        mazestatus= dict(zip(unique, counts))
        #what it means visited nodes is length of path
        #it's mistakenly name as visited nodes
        numberofvisitednodes= mazestatus[3]
        numberofvisitednodes_withlengthofpath = mazestatus[2]+mazestatus[3]
        #mazeH, mazeW = maze_matrix_visited_dfs.shape
        #numberofvisitednodes=0
        #for i in range(mazeH):
        #    for j in range(mazeW):
        #        if maze_matrix_visited_dfs[i][j] == 3:
        #            numberofvisitednodes=numberofvisitednodes+1
        #get higher number visited nodes and higher number obstacle
        if(percentile<=numberofvisitednodes and numberofobstacle>averageObstacle):
            numberofvisitednodes
        else:
            numberofvisitednodes=0
    return numberofvisitednodes,numberofobstacle,maze_matrix_visited_dfs


def RecalculateMazeProbabiliy(mazeNumberOfVisitedNodes):
    totalNumberVisitedNodes=sum(mazeNumberOfVisitedNodes)
    totalProbabilty=0.0
    mazesProbabiltyOfVisitedNodes=[]
    for i in range(len(mazeNumberOfVisitedNodes)):
        previousProbabiltiy=totalProbabilty
        totalProbabilty = totalProbabilty + ( mazeNumberOfVisitedNodes[i]/ totalNumberVisitedNodes)
        mazesProbabiltyOfVisitedNodes.append((previousProbabiltiy,totalProbabilty))
    return mazesProbabiltyOfVisitedNodes


def ReproduceMaze(fatherMaze,motherMaze):
    lenmaze=len(fatherMaze)
    randomcut=random.randint(1, lenmaze - 1)
    fatherstartcell, fatherendcell,cropfathermaze=cropmaze(fatherMaze,0,randomcut)
    motherstartcell, motherendcell,cropmothermaze=cropmaze(motherMaze,randomcut,lenmaze)
    childmaze = collections.OrderedDict(**cropfathermaze, **cropmothermaze)
    # has to be in order dict because normal dict will mess with the cell ordering which is the key
    ReconstructEdgeOnSplitArea(childmaze,fatherendcell)
    ReconstructEdgeOnSplitArea(childmaze, motherstartcell )
    return childmaze

def RandomMazeSelection(mazePopulation,mazeProbability):
    randomselection = random.uniform(0, 1)
    mazePopulationIndex=FindProbabilityInterval(mazeProbability,randomselection)
    maze=mazePopulation[mazePopulationIndex]
    return maze

def FindProbabilityInterval(mazeProbabilityOfObstructions ,randomProbability):
    for currentindex in range(len(mazeProbabilityOfObstructions)):
        if (mazeProbabilityOfObstructions[currentindex][0] < randomProbability and randomProbability <= mazeProbabilityOfObstructions[currentindex][1]):
            return currentindex


def cropmaze(mazedictionary, start,end):
    newdict=collections.OrderedDict()
    cells=sorted(list(mazedictionary.keys()))
    listofcells= cells[start:end]
    for cell in listofcells:
        newdict[cell] = mazedictionary[cell]

    startcell=listofcells[0]
    endcell=listofcells[len(listofcells)-1]
    return startcell,endcell,newdict

def MutateMaze(childmaze):
    #randomly remove and add obstruction
    randomaction=random.uniform(0, 1)
    listofkeyofnotavailable=[]
    listofkeyofavailable=[]
    for cell in childmaze:
        if childmaze[cell] == []:
            listofkeyofnotavailable.append(cell)
        else:
            listofkeyofavailable.append(cell)

    if(randomaction<=0.5):
    #add obstruction
        randomselectioncell = random.randint(0, len(listofkeyofavailable) - 1)
        key=listofkeyofavailable[randomselectioncell]
        listofnodes= childmaze[key]

        for node in listofnodes:
            childmaze[node[0]]=[item for item in childmaze[node[0]] if not (item[0] == key)]
        childmaze[key] = []
    else:
        if((len(listofkeyofnotavailable) - 1)>0):
            randomselectioncell = random.randint(0, len(listofkeyofnotavailable) - 1)
            key = listofkeyofnotavailable[randomselectioncell]
            row= key[0]
            col=key[1]
            size = len(childmaze) ** (1 / 2.0)

            if(row>=1  and len(childmaze[(row-1, col)])>0):
                #add north edge to the selected cell
                edgeN = ((row-1, col), 'North')
                childmaze[key].append(edgeN)
            if(row <size-1  and len(childmaze[(row + 1, col)])>0):
                #add south edge to the selected cell
                edgeS = ((row + 1, col), 'South')
                childmaze[key].append(edgeS)
            if (col >= 1  and len(childmaze[(row, col - 1)]) > 0):
                #add west edge to the selected cell
                edgeW = ((row, col - 1), 'West')
                childmaze[key].append(edgeW)
            if (col < size - 1 and len(childmaze[ (row, col+ 1) ]) > 0):
                #add east edge to the selected cell
                edgeE = ((row, col+ 1), 'East')
                childmaze[key].append(edgeE)

            listofnodes = childmaze[key]
            for node in listofnodes:
                if(node[1]=="North"):
                    childmaze[node[0]].append((key, 'South'))
                elif (node[1] == "South"):
                    childmaze[node[0]].append((key, 'North'))
                elif (node[1] == "East"):
                    childmaze[node[0]].append((key, 'West'))
                if (node[1] == "West"):
                    childmaze[node[0]].append((key, 'East'))

    return childmaze

def ReconstructEdgeOnSplitArea(childMaze,cell):
    edges = childMaze[cell]
    obstructionExistOnCell=(len(edges)==0)

    cellrow = cell[0]
    cellcol=cell[1]
    lenroworcol = len(childMaze) ** (1 / 2.0)

    if(obstructionExistOnCell==True and cellrow >= 1 ):
        northcellkey=(cellrow - 1, cellcol)
        edgeSouth = (cell, 'South')
        if (len(childMaze[northcellkey]) > 0 and edgeSouth in childMaze[northcellkey]):
            #remove  cell from North Cell
            childMaze[northcellkey].remove(edgeSouth)
    elif (obstructionExistOnCell==False and cellrow >= 1 ):
        #add new edge to father cell if north does not exist
        northcellkey=(cellrow - 1, cellcol)
        edgeN = (northcellkey, 'North')
        if ( edgeN in childMaze[cell]):
            childMaze[cell].remove(edgeN);
        if(len(childMaze[northcellkey]) > 0 and edgeN not in childMaze[cell]):
            childMaze[cell].append(edgeN)

    if (obstructionExistOnCell == True and cellrow < lenroworcol - 1 ):
        sourthcellkey = (cellrow + 1, cellcol)
        edgeNorth = (cell, 'North')
        if (len(childMaze[sourthcellkey]) > 0 and edgeNorth in childMaze[sourthcellkey]):
            # remove father cell from South Cell
            childMaze[sourthcellkey].remove(edgeNorth)
    elif (obstructionExistOnCell == False and cellrow < lenroworcol - 1 ):
        # add new edge to father cell if south does not exist
        sourthcellkey = (cellrow + 1, cellcol)
        edgeSouth = (sourthcellkey, 'South')
        if ( edgeSouth in childMaze[cell]):
            #remove south edge to be updated with latest edge
            childMaze[cell].remove(edgeSouth)
        if (len(childMaze[sourthcellkey]) > 0 and edgeSouth not in childMaze[cell]):
            childMaze[cell].append(edgeSouth)

    if (obstructionExistOnCell == True and cellcol >= 1 ):
        westcell = (cellrow, cellcol - 1)
        edgeEast = (cell, 'East')
        if (len(childMaze[westcell]) > 0 and edgeEast in childMaze[westcell]):
            # remove East cell from West Cell
            childMaze[westcell].remove(edgeEast)
    elif (obstructionExistOnCell == False and cellcol >= 1 ):
        # add new edge to  cell if south does not exist
        westcell = (cellrow, cellcol - 1)
        edgeWest = (westcell, 'West')
        if (edgeWest in childMaze[cell]):
            # remove south edge to be updated with latest edge
            childMaze[cell].remove(edgeWest)
        if (len(childMaze[westcell]) > 0 and edgeWest not in childMaze[cell]):
            childMaze[cell].append(edgeWest)

    if (obstructionExistOnCell == True and cellcol < lenroworcol - 1):
        eastcell = (cellrow, cellcol + 1)
        edgeWest = (cell, 'West')
        if (len(childMaze[eastcell]) > 0 and edgeWest in childMaze[eastcell]):
            # remove West cell from East Cell
            childMaze[eastcell].remove(edgeWest)
    elif (obstructionExistOnCell == False and cellcol < lenroworcol - 1):
        # add new edge to  cell if east does not exist
        eastcell = (cellrow, cellcol + 1)
        edgeEast = (eastcell, 'East')
        if (edgeEast in childMaze[cell]):
            # remove south edge to be updated with latest edge
            childMaze[cell].remove(edgeEast)
        if (len(childMaze[eastcell]) > 0 and edgeEast not in childMaze[cell]):
            childMaze[cell].append(edgeEast)
