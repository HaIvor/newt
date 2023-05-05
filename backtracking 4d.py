import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt




n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = np.array([
    [423],
    [5],
    [2]
])

N_list = np.array(xi[0])
M_list = np.array(xi[1])
m_list = np.array(xi[2])


i = 1

def cost_function(x):
    return x[0]**2 - 6*x[0]+x[1]**2-8*x[1]+x[2]**2-3*x[2]
c = 1 #??

iterations = 3
alfa = 0.01
beta = 0.6
cost_list = np.array(cost_function(xi))
while i < iterations:

    

    hi = nd.Hessian(cost_function)(xi.reshape((n,))) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi).reshape((n,1))
    #xi = xi - 1/(hi[0][0])*gradient # Newton-Raphson Consensus
    if (np.abs(np.linalg.eigvals(hi)) < c).all():
            hi = c*np.identity(n)
    t = 1
    deltax = -1.0*(np.linalg.inv(hi)@gradient) #husk minus du


    while cost_function(xi+t*deltax) > (cost_function(xi)+alfa*t*np.transpose(gradient)@deltax):
         t = t*beta
         print("t: \n ", t)
         

    
    print("t:", t)
    print("---")

    xi = xi + t*deltax
    #xi = xi - epsilon*(np.linalg.inv(hi)@gradient)
    #print("second x \n", xi)
    #print(np.linalg.inv(hi)@gradient)
    # For plotting the values...

    N_list = np.append(N_list, xi[0][0])
    M_list = np.append(M_list, xi[1][0])
    m_list = np.append(m_list, xi[2][0])
    cost_list = np.append(cost_list, cost_function(xi))

    

    
    i = i+1



figure, (ax1, ax2, ax3, ax4) = plt.subplots(1,n+1, figsize=(14, 8))
ax1.plot(np.arange(iterations), N_list, "g")

ax2.plot(np.arange(iterations), M_list, "r")

ax3.plot(np.arange(iterations), m_list, "b")
ax4.plot(np.arange(iterations), cost_list, "k")

#plt.legend(["M"])
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()

ax1.set_xlabel('Iteration')
ax2.set_xlabel('Iteration')
ax3.set_xlabel('Iteration')
ax4.set_xlabel('Iteration')
figure.suptitle("Newton's Method for function, backtracking$", fontsize=16)

ax1.set_title('$x$')
ax2.set_title('$y$')
ax3.set_title('$z$')

#plt.xticks(np.arange(iterations))
plt.show()
#print(M_list)
