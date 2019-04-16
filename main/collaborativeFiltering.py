'''
Created on Jan 21, 2018

@author: Beatrice Moissinac
@summary: Run collaborative filtering
'''
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import cross_validation as cv


# Read collections.csv 
def readUsersCollections(newGameIds, file):
    adjacencyList = []
    usersList = {} # Dictionary {userName:userIdx}
    idx = 0
    with open(file) as f:
        for line in f:
            line = line.rstrip().replace(' ','_').split(',')
            # Add user to userlist
            userId = line[0]
            usersList[userId] = idx
            
            for i in range(1, len(line)):
                newGameId= newGameIds[int(line[i])]
                newAdjacency = ((idx,newGameId),1)
                adjacencyList.append(newAdjacency)
            idx += 1
    return adjacencyList, usersList

# Re-index games
def reindexingGames(games):
    gamesList = games.gameId.tolist()
    newGameIds = {}
    idx = 0
    for game in gamesList:
        newGameIds[game] = idx
        idx += 1
    return newGameIds


# Create matrix from adjacency list
def transformAdjacencyList(ydim, xdim, adjacencyList):

    mat = np.zeros((xdim, ydim))
    
    for (x,y),w in adjacencyList:
        mat[x][y] = w

    return mat

if __name__ == '__main__':
    #mechanics = readFileCSV('../data/dico/mechanics.csv')
    #categories = readFileCSV('../data/dico/categories.csv')
    games = pd.read_csv('../data/games/gameData.csv', header=None, names=['gameId', 'gameName','year'])
   
    # Convert gameId indexes to dictionary for fast look up
    newGameIds = reindexingGames(games)
    adjacencyList, users = readUsersCollections(newGameIds, '../data/users/collections.csv')
    
    userGames = transformAdjacencyList(games.shape[0], len(users), adjacencyList)
    
    user_similarity = pairwise_distances(userGames, metric='cosine')
    
    

    pass