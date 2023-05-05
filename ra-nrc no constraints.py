# Pseudo code for node i 

import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt

log = np.log # The natural logarithm
log2 = np.log2 # Log base 2
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
# x[2] = m, x[1] = M, x[0] = N, cost function definition from Emil paper
def cost_function(x):
    #return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))
    return x[0]**2 - 6*x[0]+x[1]**2-8*x[1]+x[2]**2-3*x[2]
# Minus at the start for turning it into convex function instead of concave

"""
The values we are actually optimizing: 

N = optimizable - Subcarriers 
M = optimizable - Modulation order
m = optimizable - Symbols per packet
"""


# ------Initialization------


# Initial estimate for the global optimization. Arbitrary values, recommended by Emil
xi = np.array([ 
    [19], # N_start
    [1], #M_start
    [3] # m_start
]) 

yi, gi, gi_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zi, hi, hi_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmai_y, sigmai_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3
sigmaj_y, sigmaj_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3

# Each np.zeros is for node j1, j2. Will make more np.zeros if it has more neighbors...
rhoi_y = np.zeros((n,1))
# Each np.zeros is for node j1, j2. Will make bigger if it has more neighbors.
# The rhoi_z is as array that contains nxn matrices (the hessians of node j's). ex. rhoi_z[j] = [3x3,3x3,3x3....]
rhoi_z = np.zeros((n,n))
        

# ------Initialization 2------

xj = np.array([ 
    [14], # N_start
    [3], #M_start
    [1] # m_start
]) 

yj, gj, gj_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zj, hj, hj_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmaj_y, sigmaj_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3

# Each np.zeros is for node j1, j2. Will make more np.zeros if it has more neighbors...
rhoj_y = np.zeros((n,1))
# Each np.zeros is for node j1, j2. Will make bigger if it has more neighbors.
# The rhoi_z is as array that contains nxn matrices (the hessians of node j's). ex. rhoi_z[j] = [3x3,3x3,3x3....]
rhoj_z = np.zeros((n,n))

# ------------------------------


flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
flag_update = 1

# Used for when plotting N, M and m. Just to see how they are evolving
N_list = np.array(xi[0])
M_list = np.array(xi[1])
m_list = np.array(xi[2])
N_list2 = np.array(xj[0])
M_list2 = np.array(xj[1])
m_list2 = np.array(xj[2])

i = 1
iterations = 1000 # For plotting the evolution of xi (N, M, m)
while i < iterations: 
    
# ------Data transmission-------

    if flag_transmission == 1:
        transmitter_node_ID = 1 # transmitter_node_ID = i, just saying we are working on node 1 now ex.
        
        # Push sum consensus
        yi = (1/(out_neigh + 1))*yi
        zi = (1/(out_neigh + 1))*zi
        # The sigmas are broadcasted to the neighbors 
        sigmai_y = sigmai_y + yi
        sigmai_z = sigmai_z + zi
        #pseudo code: Broadcast: transmitter_node_ID, sigma_y[i], sigma_z[i].
        # Have to set up with modem. Subnero stuff...
        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_reception2 = 1
        
# ------Data Reception------

    if flag_reception == 1:
        # Should get values from node j: transmitter_node_ID, sigmaj_y and sigmaj_z
        transmitter_node_ID = 2 # Should come from node j, just setting to 2 for the time being...

        
        # Just pretending the node j sends sigmas. They are constant though, so not optimal for testing...
        # sigmaj_y = np.array([
        #     [4],
        #     [2],
        #     [10]
        # ])
        # sigmaj_z = np.array([
        #     [0.3,0.3,0.2],
        #     [0.1,0.8,0.6],
        #     [0.6,0.3,0.2]
        # ])

        yi = yi + sigmaj_y - rhoi_y
        zi = zi + sigmaj_z - rhoi_z
        rhoi_y = sigmaj_y 
        rhoi_z = sigmaj_z 
        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_update = 1
        

# ------Estimate Update------ 

    if flag_update == 1:

        if (np.abs(np.linalg.eigvals(zi)) < c).all():
            zi = c*np.identity(n)

        xi = (1-epsilon)*xi + epsilon*np.linalg.inv(zi)@yi # Newton-Raphson Consensus
        print(zi)
        print("----")
        # For plotting the values...
        N_list = np.append(N_list, xi[0][0])
        M_list = np.append(M_list, xi[1][0])
        m_list = np.append(m_list, xi[2][0])
        

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

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission = 1

        i += 1 # For plotting and simulating
        #print(rhoj_y-sigmai_y)
        
        



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------


    # ------Data transmission 2-------

    if flag_transmission2 == 1:
        transmitter_node_ID = 2 # transmitter_node_ID = i, just saying we are working on node 1 now ex.
        
        # Push sum consensus
        yj = (1/(out_neigh + 1))*yj
        zj = (1/(out_neigh + 1))*zj
        # The sigmas are broadcasted to the neighbors 
        sigmaj_y = sigmaj_y + yj
        sigmaj_z = sigmaj_z + zj
        #pseudo code: Broadcast: transmitter_node_ID, sigma_y[i], sigma_z[i].
        # Have to set up with modem. Subnero stuff...
        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_reception = 1

    # ------Data Reception 2------

    if flag_reception2 == 1:
        # Should get values from node j: transmitter_node_ID, sigmaj_y and sigmaj_z
       # Will come from neighbor node (node j)

        yj = yj + sigmai_y - rhoj_y #andre noden er nr "1"
        zj = zj + sigmai_z - rhoj_z
        rhoj_y = sigmai_y 
        rhoj_z = sigmai_z 
        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_update2 = 1

    # ------Estimate Update 2------ 

    if flag_update2 == 1:
        if (np.abs(np.linalg.eigvals(zj)) < c).all():
            zj = c*np.identity(n)
        #print(zj)
        #print("----")
        
        xj = (1-epsilon)*xj + epsilon*np.linalg.inv(zj)@yj # Newton-Raphson Consensus
        
        # For plotting the values...
        # N_list = np.append(N_list, xj[0][0])
        # M_list = np.append(M_list, xj[1][0])
        # m_list = np.append(m_list, xj[2][0])
        N_list2 = np.append(N_list2, xj[0][0])
        M_list2 = np.append(M_list2, xj[1][0])
        m_list2 = np.append(m_list2, xj[2][0])
        
        gj_old = gj
        hj_old = hj
        hj = nd.Hessian(cost_function)(xj.reshape((3,))) # numtools calc hessian
        gradient = nd.Gradient(cost_function)(xj) # numtools calc gradient
        gj = hj@xj-gradient.reshape((3,1))

        yj = yj + gj - gj_old
        zj = zj + hj - hj_old

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission2 = 1


    # ------Plotting------
if i == iterations:   

    figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 8))
    ax1.plot(np.arange(iterations), N_list, "b", label="x - node 1")
    ax1.plot(np.arange(iterations-1), N_list2, "c", label="x - node 2")
    
    ax2.plot(np.arange(iterations), M_list, "r", label="y - node 1")
    ax2.plot(np.arange(iterations-1), M_list2, "y", label="y - node 2")
    
    ax3.plot(np.arange(iterations), m_list, "g", label="z - node 1")
    ax3.plot(np.arange(iterations-1), m_list2, "k", label="z - node 2")
    
    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")

    plt.xlabel('iterations') 
    
    figure.suptitle('ra-NRC with two nodes - brute forced', fontsize=16)

    plt.xlabel('iterations') 
    #plt.xticks(np.arange(iterations))

    print(f"The last x values: \n x1 = {N_list[-1]} \n x2 = {N_list2[-1]}\n")
    print(f"The last y values: \n y1 = {M_list[-1]} \n y2 = {M_list2[-1]} \n")
    print(f"The last z values: \n z1 = {m_list[-1]} \n z2 = {m_list2[-1]}")
    
    plt.show()

