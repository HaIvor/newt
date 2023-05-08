# ra-NRC for two nodes, brute forced the order. 

import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt
import sympy as sym

def calculateHessianAndGradient(xi):

    x1, x2, x3 = sym.symbols('x1 x2 x3')

    function = 5*x1**2- 2*x1 + 3*x2**2 - 5*x2 + 8*x3**2 - 6*x3 

    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)
    der_x3 = function.diff(x3)

    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3_values = function.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    gradient_values = np.array([
        [der_x1_values], 
        [der_x2_values], 
        [der_x3_values]
    ], dtype=np.float32)

    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x3_values = der_x1.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3x3_values = der_x3.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx2x3_values = der_x2.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)
    
    return gradient_values, hessian_values

n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1

out_neigh = 1 # Just two nodes communicating, so 1 neighbor each
c = 1E-8 # If hessian is not behaving 
epsilon = 0.3 # just to give value temporary

# -----------Initialization-----------

# Initial estimate for the global optimization. Arbitrary values

# Initial values for node i 
xi = np.array([
    [14],
    [20],
    [9]
])

yi, gi, gi_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zi, hi, hi_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmai_y, sigmai_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3

# Initialize to a 3x1 matrix with zeros
rhoi_y = np.zeros((n,1))

# Initialize to a 3x3 matrix with zeros
rhoi_z = np.zeros((n,n))


# -----------Initialization 2-----------

# Initial values for node j
xj = np.array([ 
    [32], # N_start
    [13], #M_start
    [20] # m_start
]) 

yj, gj, gj_old = np.zeros((n, 1)), np.zeros((n, 1)), np.zeros((n, 1)) # if n=3 -> they are 3x1 zero matrices
zj, hj, hj_old = np.identity(n), np.identity(n), np.identity(n) # if n=3 -> they are 3x3 identity matrices
sigmaj_y, sigmaj_z = np.zeros((n, 1)), np.zeros((n, n)) # 3x1 and 3x3. Gradient is 3x1 and hessian is 3x3

# Initialize to a 3x1 matrix with zeros
rhoj_y = np.zeros((n,1))

# Initialize to a 3x3 matrix with zeros
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
iterations = 20 # Basically how many receptions/transmissions 
while i < iterations: 
    
# -----------Data transmission------------

    if flag_transmission == 1:

        # transmitter_node_ID = i, just saying we are transmitting from node i now.
        transmitter_node_ID = "i" 
        
        # Push sum consensus
        yi = (1/(out_neigh + 1))*yi
        zi = (1/(out_neigh + 1))*zi

        # See if things work
        print("yi ", yi)
        print("zi ", zi)

        # Update the amout of weight it has broadcast / tried to broadcast
        sigmai_y = sigmai_y + yi
        sigmai_z = sigmai_z + zi

        # The sigmas are broadcasted + nodeID to the neighbors... 
        # Subnero code stuff...

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_reception2 = 1
        
# -----------Data Reception-----------

    if flag_reception == 1:

        # Should get the two sigmas and nodeID from node j 
        transmitter_node_ID = "j" 

        # Here it updates its own value according to what it got from node j
        yi = yi + sigmaj_y - rhoi_y
        zi = zi + sigmaj_z - rhoi_z

        # Rhos accumulates for lost packets, if everything ideal => it does nothing
        rhoi_y = sigmaj_y 
        rhoi_z = sigmaj_z 

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_update = 1
        

# ----------Estimate Update-----------

    if flag_update == 1:

        if (np.abs(np.linalg.eigvals(zi)) < c).all():
            zi = c*np.identity(n)

        xi = (1-epsilon)*xi + epsilon*np.linalg.inv(zi)@yi # Newton-Raphson Consensus

        # For plotting the values...
        N_list = np.append(N_list, xi[0][0])
        M_list = np.append(M_list, xi[1][0])
        m_list = np.append(m_list, xi[2][0])

        gi_old = gi
        hi_old = hi
        gradient, hi = calculateHessianAndGradient(xi)
        gi = hi@xi - gradient

        yi = yi + gi - gi_old
        zi = zi + hi - hi_old

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_transmission = 1

        i += 1 # For plotting and simulating
        print("iteration: ", i)
        
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------


    # -----------Data transmission 2------------

    if flag_transmission2 == 1:
        
        # transmitter_node_ID = j, just saying we are transmitting from node j now.
        transmitter_node_ID = "j"  
        
        # Push sum consensus
        yj = (1/(out_neigh + 1))*yj
        zj = (1/(out_neigh + 1))*zj
        
        # Update the amout of weight it has broadcast / tried to broadcast 
        sigmaj_y = sigmaj_y + yj
        sigmaj_z = sigmaj_z + zj
        
        # The sigmas are broadcasted + nodeID to the neighbors... 
        # Subnero code stuff...

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_reception = 1

    # -----------Data Reception 2-----------

    if flag_reception2 == 1:

        # Should get the two sigmas and nodeID from node i
        transmitter_node_ID = "i" 

        # Here it updates its own value according to what it got from node i 
        yj = yj + sigmai_y - rhoj_y 
        zj = zj + sigmai_z - rhoj_z

        # Rhos accumulates for lost packets, if everything ideal => it does nothing
        rhoj_y = sigmai_y 
        rhoj_z = sigmai_z 

        flag_reception, flag_update, flag_transmission, flag_update2, flag_reception2, flag_transmission2 = 0, 0, 0, 0, 0, 0
        flag_update2 = 1

    # ------------Estimate Update 2-----------

    if flag_update2 == 1:

        if (np.abs(np.linalg.eigvals(zj)) < c).all():
            zj = c*np.identity(n)

        xj = (1-epsilon)*xj + epsilon*np.linalg.inv(zj)@yj # Newton-Raphson Consensus
        
        # For plotting the values...
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


# ------------Plotting------------
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