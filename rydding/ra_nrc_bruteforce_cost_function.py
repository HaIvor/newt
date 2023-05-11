# Pseudo code for node i 

import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt
import sympy as sym
import sys 


"""
The values we are actually optimizing: 

N = optimizable - Subcarriers 
M = optimizable - Modulation order
m = optimizable - Symbols per packet
"""
def calculateHessianAndGradient(xi):

    R_c = 0.5 # Coding rate
    B = 12000 # Bandwidth
    n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
    p_c = 0.25 # Cyclic prefix fraction
    t_oh = 0.1 # Overhead (preamble, propagation...)
    R_n = 0.5 # 0<R_n<1 andel av bærebølger 

    r = 1000 # r = 100-4000m -emil
    c1 = 1500 # Speed of sound, 343 m/s
    t_d = r/c1 # Transmission delay t_d = r/c

    p_lt = 0.001
    gamma = 31.62 # 20db
    eta = 1

    

    x1, x2, x3 = sym.symbols('x1 x2 x3')

    N = x1
    m = x2
    M = x3
    # Federico constraints
    constraint1 = sym.log(-(1 - m)) + sym.log(-(m - 40)) #  1 < m < 40
    constraint2 = sym.log(-(N - 2000)) + sym.log(-(400 - N)) # 400 < N < 2000
    constraint3 = sym.log(-(2 - M)) + sym.log(-(M - 64)) # 2 < M < 64
    constraint4 = sym.log(-((m * (N / eta) * (sym.log(M, 2)) * (0.2 * sym.exp(-((3 * 100) / (2 * (M - 1))))) ** (1 / R_c)) - 0.1))

    # Defining the objective function we want to minimize
    # This example, arbitrary function with three constraints. 
    function_without_constraint = -(sym.log(x3)+ sym.log(R_c) + sym.log(B) + sym.log(R_n) + sym.log(x1) + sym.log(sym.log(x2, 2)) - sym.log(x3*(1+p_c)*x1 + B*(t_oh + t_d)))
    function = bb*( function_without_constraint ) - (constraint1+constraint2+constraint3+constraint4) 


    # Function value
    function_value = function.evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]}) 

    # Derivating f(x) for x1, x2, x3 (algebraic answer, without values)
    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)
    der_x3 = function.diff(x3)

    # Putting values into the derivatives
    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3_values = function.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the gradient of the objective function
    gradient_values = np.array([
        [der_x1_values],
        [der_x2_values],
        [der_x3_values]
    ], dtype=np.float32)

    # Derivating the objective function further to get the hessian
    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x3_values = der_x1.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3x3_values = der_x3.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx2x3_values = der_x2.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the hessian of the objective function
    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)

    # Check if the answer is feasible. A too big t value can cause numerical difficulties 
    if sym.im(function_value) != 0:
        print('=========COMPLEX ANSWER, NOT FEASIBLE, MAYBE DECREASE t =============')
        print("f(x): ", function_value)
        sys.exit()

    return gradient_values, hessian_values

n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1

out_neigh = 1 # For the time being this is not calculated, but set beforehand
c = 1E-8
epsilon = 0.1 # just to give value temporary
bb=5

# ------Initialization------


# Initial estimate for the global optimization. Arbitrary values, recommended by Emil
xi = np.array([
    [990],
    [18],
    [25]
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
    [1000], # N_start
    [21], #M_start
    [25] # m_start
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
iterations = 200 # For plotting the evolution of xi (N, M, m)
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

        # if (np.abs(np.linalg.eigvals(zi)) < c).all():
        #     zi = c*np.identity(n)

        xi = (1-epsilon)*xi + epsilon*np.linalg.inv(zi)@yi # Newton-Raphson Consensus

        # For plotting the values...
        N_list = np.append(N_list, xi[0][0])
        M_list = np.append(M_list, xi[1][0])
        m_list = np.append(m_list, xi[2][0])
        

        gi_old = gi
        hi_old = hi
        gradient, hi = calculateHessianAndGradient(xi)
        gi = hi@xi-gradient

        print("yi ", yi)
        print("zi ", zi)
        print("xi: \n", xi)

        yi = yi + gi - gi_old
        zi = zi + hi - hi_old

        
        # Just some "useful" prints for debugging
        #print(f"Sigmai_y: \n {sigmai_y} \n Sigmai_z: \n {sigmai_z} \n xi: \n {xi}")
        #print("--neste i---")
        #print(f"iteration: {i}, \n gi = {gi}")

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission = 1

        i += 1 # For plotting and simulating
        print("iteration: ", i)
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
        gradient, hj = calculateHessianAndGradient(xj)
        gj = hj@xj-gradient

        yj = yj + gj - gj_old
        zj = zj + hj - hj_old

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission2 = 1


    # ------Plotting------
if i == iterations:   

    figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 8))
    ax1.plot(np.arange(iterations), N_list, "b", label="N - node 1")
    ax1.plot(np.arange(iterations-1), N_list2, "c", label="N - node 2")
    
    ax2.plot(np.arange(iterations), M_list, "r", label="M - node 1")
    ax2.plot(np.arange(iterations-1), M_list2, "y", label="M - node 2")
    
    ax3.plot(np.arange(iterations), m_list, "g", label="m - node 1")
    ax3.plot(np.arange(iterations-1), m_list2, "k", label="m - node 2")
    
    ax1.grid()
    ax2.grid()
    ax3.grid()

    ax1.legend(loc="lower right")
    ax2.legend(loc="lower right")
    ax3.legend(loc="lower right")

    plt.xlabel('iterations') 
    
    figure.suptitle('ra-NRC with two nodes - brute forced', fontsize=16)

    plt.xlabel('iterations') 
    #plt.xticks(np.arange(iterations))

    print(f"The last N values: \n N1 = {N_list[-1]} \n N2 = {N_list2[-1]}\n")
    print(f"The last M values: \n M1 = {M_list[-1]} \n M2 = {M_list2[-1]} \n")
    print(f"The last m values: \n m1 = {m_list[-1]} \n m2 = {m_list2[-1]}")
    
    plt.show()

