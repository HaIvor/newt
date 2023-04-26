import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt

epsilon = 0.1



n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = 1.72
N_list = np.array(xi)


i = 1

def cost_function(x):
    return x**3-9*x+4
c = 1 #??

iterations = 60

while i < iterations:

    

    hi = nd.Hessian(cost_function)(xi) # numtools calc hessian
    gradient = nd.Gradient(cost_function)(xi)

    #xi = xi - 1/(hi[0][0])*gradient # Newton-Raphson Consensus
    xi = (1-epsilon)*xi+epsilon*(1/hi[0][0])*(hi[0][0]*xi-gradient)
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
plt.title("Using newton's method on function: $x^3$ - 9x + 4. \n Root is really sqrt(3)=1.73205080757..")
#plt.xticks(np.arange(iterations))
plt.show()
