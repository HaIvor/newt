import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt




n = 2 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = np.array([
    [3],
    [5]
])

N_list = np.array(xi[0])
M_list = np.array(xi[1])


i = 1

def cost_function(x):
    return np.log(np.exp(x[0])+np.exp(x[1]))
c = 1 #??

iterations = 140
alfa = 0.01
beta = 0.6
cost_list = np.array(cost_function(xi))
while i < iterations:

    

    hi = nd.Hessian(cost_function)(xi.reshape((n,))) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi).reshape((n,1))

    #xi = xi - 1/(hi[0][0])*gradient # Newton-Raphson Consensus

    t = 1
    deltax = -1.0*(np.linalg.inv(hi)@gradient) #husk minus du


    while cost_function(xi+t*deltax) > (cost_function(xi)+alfa*t*np.transpose(gradient)@deltax):
         t = t*beta
         print("t: \n ", t)
         if t < 1e-1:
             break
         

    if (np.abs(np.linalg.eigvals(hi)) < c).all():
            hi = c*np.identity(n)
    print(xi)
    print("---")

    xi = xi + t*deltax
    #xi = xi - epsilon*(np.linalg.inv(hi)@gradient)
    #print("second x \n", xi)
    #print(np.linalg.inv(hi)@gradient)
    # For plotting the values...

    N_list = np.append(N_list, xi[0][0])
    M_list = np.append(M_list, xi[1][0])
    cost_list = np.append(cost_list, cost_function(xi))

    

    
    i = i+1



figure, (ax1, ax2, ax3) = plt.subplots(1,n+1, figsize=(14, 8))
ax1.plot(np.arange(iterations), N_list, "g")

ax2.plot(np.arange(iterations), M_list, "r")

ax3.plot(np.arange(iterations), cost_list, "k")

#plt.legend(["M"])
ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('Iteration')
ax2.set_xlabel('Iteration')
ax3.set_xlabel('Iteration')

figure.suptitle("Newton's Method for function, backtracking$", fontsize=16)

ax1.set_title('$x$')
ax2.set_title('$y$')
ax3.set_title('$z$')

#plt.xticks(np.arange(iterations))
plt.show()
#print(M_list)
