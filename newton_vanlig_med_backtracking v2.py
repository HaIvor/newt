import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt


R_c = 0.5 # Coding rate
B = 12 # Bandwidth
n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
p_c = 0.25 # Cyclic prefix fraction
t_oh = 0.1 # Overhead (preamble, propagation...)
R_n = 0.5 # 0<R_n<1 andel av bærebølger 

r = 1000 # r = 100-4000m, range
c1 = 1500 # Speed of sound, 1500 m/s
t_d = r/c1 # Transmission delay t_d = r/c
log = np.log # The natural logarithm
log2 = np.log2 # Log base 2



n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = np.array([ 
    [0.4], # N_start
    [-0.3], #M_start
]) 
N_list = np.array(xi[0])
M_list = np.array(xi[1])

hi = np.identity(n)# if n=3 -> they are 3x3 identity matrices

i = 1
# x[2] = m, x[1] = M, x[0] = N
def cost_function(x):
    #return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))
    #return x[0]**2-500*x[0]+x[1]**2-30*x[1]+5*x[2]**2-60*x[2]
    return (x[0]**2 - 30*x[0] + x[1]**2 - x[1])
c = 1E-6 #??

cost_list = np.array(cost_function(xi))


iterations = 4
alfa = 0.2
beta = 0.6

print(cost_function(xi))

while i < iterations:

    

    hi = nd.Hessian(cost_function)(xi.reshape((2,))) # numtools calc hessian
    
    gradient = nd.Gradient(cost_function)(xi).reshape((2,1)) # numtools calc gradient

    if (np.abs(np.linalg.eigvals(hi)) < c).all():
            hi = c*np.identity(n)

    
    t = 1
    deltax = -1.0*np.linalg.inv(hi)@gradient #husk minus du
    

    while cost_function(xi+t*deltax) > (cost_function(xi)+alfa*np.transpose(gradient)@deltax):
         t = t*beta
         print("t: \n ", t)
    
    #xi = xi - 1/(hi[0][0])*gradient # Newton-Raphson Consensus
    xi = xi + t*deltax
    print("i: \n ", i)
    print("-------")
    #xi = xi - epsilon*(np.linalg.inv(hi)@gradient)
    #print("second x \n", xi)
    #print(np.linalg.inv(hi)@gradient)
    # For plotting the values...

    N_list = np.append(N_list, xi[0][0])
    M_list = np.append(M_list, xi[1][0])
    cost_list = np.append(cost_list, cost_function(xi))

    

    if i == iterations-1:
        print("xi er da på slutten: \n xi=[N,M,m] \n", xi)
        print("cost functionen er da: \n", cost_function(xi))
    i = i+1



figure, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14, 8))
ax1.plot(np.arange(iterations), N_list, "g")

ax2.plot(np.arange(iterations), M_list, "r")


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

figure.suptitle("Newton's Method for: \n $x^2 - 500x + y^2 - 30y + 5z^2 - 60z$", fontsize=16)

ax1.set_title('$x$')
ax2.set_title('$y$')
ax3.set_title('$z$')
ax4.set_title("Function value")

#plt.xticks(np.arange(iterations))
plt.show()
#print(M_list)

