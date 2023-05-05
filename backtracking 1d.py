import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt




n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = -1.0
N_list = np.array(xi)


i = 1

def cost_function(x):
    #return (1/20)*x**4-(2/5)*x+1
    return x**5 - 2*x**4 + 5
c = 1 #??

iterations = 300
alfa = 0.01
beta = 0.6

while i < iterations:

    

    hi = nd.Hessian(cost_function)(xi) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi)

    #xi = xi - 1/(hi[0][0])*gradient # Newton-Raphson Consensus

    t = 1
    deltax = -1.0*(gradient/hi[0][0]) #husk minus du
    m=0

    while cost_function(xi+t*deltax) > (cost_function(xi)+alfa*t*gradient*deltax):
         t = t*beta
         print("t: \n ", t)
         if t < 1e-1:
             break
         
    xi = xi + t*deltax
    #xi = xi - epsilon*(np.linalg.inv(hi)@gradient)
    #print("second x \n", xi)
    #print(np.linalg.inv(hi)@gradient)
    # For plotting the values...

    N_list = np.append(N_list, xi)

    print(xi)


    
    i = i+1



plt.plot(np.arange(iterations), N_list, "b")


plt.grid()
plt.xlabel('iterations') 
plt.ylabel("x")
plt.title("Using newton's method on function with backtracking")
#plt.xticks(np.arange(iterations))
plt.show()
