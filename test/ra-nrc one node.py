# Pseudo code for node i 

import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt

log = np.log # The natural logarithm
log2 = np.log2 # Log base 2

# x[2] = m, x[1] = M, x[0] = N, cost function definition from Emil paper
def cost_function(x):
    return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))
# Minus at the start for turning it into convex function instead of concave

"""
The values we are actually optimizing: 

N = optimizable - Subcarriers 
M = optimizable - Modulation order
m = optimizable - Symbols per packet
"""
# Values used in the cost function
R_c = 0.5 # Coding rate
B = 12 # Bandwidth
n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
p_c = 0.25 # Cyclic prefix fraction
t_oh = 0.1 # Overhead (preamble, propagation...)
R_n = 0.5 # 0<R_n<1 andel av bærebølger 

r = 200 # r = 100-4000m -emil
c1 = 1500 # Speed of sound, 343 m/s
t_d = r/c1 # Transmission delay t_d = r/c

n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1

out_neigh = 1 # For the time being this is not calculated, but set beforehand
c = 1E-8
epsilon = 0.01 # just to give value temporary

# ------Initialization------


# Initial estimate for the global optimization. Arbitrary values, recommended by Emil
xi = np.array([ 
    [152], # N_start
    [20], #M_start
    [9] # m_start
]) 

yi, gi, gi_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zi, hi, hi_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmai_y, sigmai_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3

# Each np.zeros is for node j1, j2. Will make more np.zeros if it has more neighbors...
rhoi_y = np.zeros((n,1))
# Each np.zeros is for node j1, j2. Will make bigger if it has more neighbors.
# The rhoi_z is as array that contains nxn matrices (the hessians of node j's). ex. rhoi_z[j] = [3x3,3x3,3x3....]
rhoi_z = np.zeros((n,n))

# ------------------------------
i = 1 # node ID of this node

flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
flag_update = 1


sigmai_y, sigmai_z = np.zeros((n, 1)), np.zeros((n, n)) # skal egentlig komme fra node j!
sigmaj_y, sigmaj_z = np.zeros((n, 1)), np.zeros((n, n)) # skal egentlig komme fra node j!
j = 2

# ------Data transmission-------

if flag_transmission == 1:
    transmitter_node_ID = i # transmitter_node_ID = i, just saying we are working on node 1 now ex.
    
    # Push sum consensus
    yi = (1/(out_neigh + 1))*yi
    zi = (1/(out_neigh + 1))*zi
    # The sigmas are broadcasted to the neighbors 
    sigmai_y = sigmai_y + yi
    sigmai_z = sigmai_z + zi
    #pseudo code: Broadcast: transmitter_node_ID, sigma_y[i], sigma_z[i]. <<--------
    # Have to set up with modem. Subnero stuff...
    flag_transmission = 0
    
# ------Data Reception------

if flag_reception == 1:
    # Should get values from node j: transmitter_node_ID, sigmaj_y and sigmaj_z
    transmitter_node_ID = j # Should come from node j, just setting to 2 for the time being...

    yi = yi + sigmaj_y - rhoi_y # sigmaj_y/sigma_z skal komme fra node j 
    zi = zi + sigmaj_z - rhoi_z
    rhoi_y = sigmaj_y 
    rhoi_z = sigmaj_z 
    flag_reception = 0
    flag_update = 1
    

# ------Estimate Update------ 

if flag_update == 1:

    if (np.abs(np.linalg.eigvals(zi)) < c).all():
        zi = c*np.identity(n)

    xi = (1-epsilon)*xi + epsilon*np.linalg.inv(zi)@yi # Newton-Raphson Consensus

    gi_old = gi
    hi_old = hi
    hi = nd.Hessian(cost_function)(xi.reshape((3,))) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi) # numtools calc gradient
    gi = hi@xi-gradient.reshape((3,1))

    yi = yi + gi - gi_old
    zi = zi + hi - hi_old

    
    # Just some "useful" prints for debugging
    #print(f"Sigmai_y: \n {sigmai_y} \n Sigmai_z: \n {sigmai_z} \n xi: \n {xi}")
    #print("--neste i---")
    #print(f"iteration: {i}, \n gi = {gi}")

    flag_update = 0
    flag_transmission = 1 # "optional"

    i += 1 # For plotting and simulating
    #print(rhoj_y-sigmai_y)