import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt

epsilon = 0.001




"""
The values we are actually optimizing: 

N = optimizable - Subcarriers 
M = optimizable - Modulation order
m = optimizable - Symbols per packet
"""



n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1

xi = np.array([ 
    [20], # N_start
    [50], #M_start
    [40] # m_start
]) 

N_list = np.array(xi[0])
M_list = np.array(xi[1])
m_list = np.array(xi[2])

hi = np.identity(n)# if n=3 -> they are 3x3 identity matrices
gradient = np.zeros((n,1))

i = 1
iterations = 11

# Values used in the cost function
R_c = 0.5 # Coding rate
B = 12 # Bandwidth
n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
p_c = 0.25 # Cyclic prefix fraction
t_oh = 0.1 # Overhead (preamble, propagation...)
R_n = 0.5 # 0<R_n<1 andel av bærebølger 

r = 1000 # r = 100-4000m
c = 1500 # Speed of sound, 1500 m/s
t_d = r/c # Transmission delay t_d = r/c
log = np.log # The natural logarithm
log2 = np.log2 # Log base 2
# x[2] = m, x[1] = M, x[0] = N
def cost_function(x):
    return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))

while i < iterations:

    if (np.abs(np.linalg.eigvals(hi)) < c).all():
            hi = c*np.identity(n)

    #xi = (1-epsilon)*xi + epsilon*np.linalg.inv(zi)@yi # Newton-Raphson Consensus
    # print("det: \n", np.linalg.det(hi))
    # print(i)
    xi = xi - epsilon*(np.linalg.inv(hi)@gradient)
    #print("second x \n", xi)
    #print(np.linalg.inv(hi)@gradient)
    # For plotting the values...
   

    hi = nd.Hessian(cost_function)(xi.reshape((3,))) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi).reshape((3,1)) # numtools calc gradient
    #print(hi)
    N_list = np.append(N_list, xi[0][0])
    M_list = np.append(M_list, xi[1][0])
    m_list = np.append(m_list, xi[2][0])
    print(xi)
    i = i+1


figure, (ax1, ax2, ax3) = plt.subplots(1,3)
ax1.plot(np.arange(iterations), m_list, "g")

ax2.plot(np.arange(iterations), M_list, "r")

ax3.plot(np.arange(iterations), N_list, "b")

#plt.legend(["M"])
ax1.set_title('m')
ax2.set_title('M')
ax3.set_title('N')
ax1.grid()
ax2.grid()
ax3.grid()
plt.xlabel('iterations') 
figure.suptitle('Only using newton raphson', fontsize=16)
#plt.xticks(np.arange(iterations))
plt.show()
#print(M_list)