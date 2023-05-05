import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt




n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = 0.03
N_list = np.array(xi)
t = 1e-9

i = 1

def cost_function(x):
    return x*np.log(x)
c = 1 #??
def const_function_faktisk(x):
    return x*np.log(x)

iterations = 50
alfa = 0.01
beta = 0.6

while i < iterations:


    hi = nd.Hessian(cost_function)(xi) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi)

    #xi = xi - 1/(hi[0][0])*gradient # Newton-Raphson Consensus

    p = 1
    deltax = -1.0*(gradient/hi[0][0]) #husk minus du
    m=0
    

    xi = xi + p*deltax
    print("p:", p)

    #xi = xi - epsilon*(np.linalg.inv(hi)@gradient)
    #print("second x \n", xi)
    #print(np.linalg.inv(hi)@gradient)
    # For plotting the values...

    N_list = np.append(N_list, xi)

    print("xi: ", xi)
    print("i: ", i)
    print("deltax", deltax)


    
    i = i+1



plt.plot(np.arange(iterations), N_list, "b")


plt.grid()
plt.xlabel('iterations') 
plt.ylabel("x")
plt.title("Using newton's method on function with backtracking")
#plt.xticks(np.arange(iterations))
print("cost function is: ", const_function_faktisk(xi))
plt.show()
