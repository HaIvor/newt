#
# Discrete-time Average Consensus
# Ivano Notarnicola, Lorenzo Pichierri
# Bologna, 09/03/2022
#

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# Some technical settings
import matplotlib as mpl
np.set_printoptions(precision=3)
# np.random.seed(0)

# Creates a random graph. The nodes have a "p_ER" chance of being connected. 
# Useful to test different kinds of systems

NN = 8
p_ER = 0.3

I_NN = np.identity(NN, dtype=int)

while 1:
    G = nx.binomial_graph(NN,p_ER)
    Adj = nx.adjacency_matrix(G)
    Adj = Adj.toarray() # Makes it "normal" array so we can work on it
    test = np.linalg.matrix_power((I_NN+Adj),NN) # A check to see if the random created graph is connected
    print("Adjacency matrix: \n", Adj)
    if np.all(test>0):
        
        break 
    else:
        print("the graph is NOT connected.. Let's try agian!")
        

nx.draw(G, with_labels = True)
ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))

threshold = 1e-10
WW = 0.7*Adj + 0.9*I_NN # Arbitrary, could scale the numbers.
WW = np.abs(WW)
