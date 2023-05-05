import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt

epsilon = 0.01

R_c = 0.5 # Coding rate
B = 12000 # Bandwidth
n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
p_c = 0.25 # Cyclic prefix fraction
t_oh = 0.1 # Overhead (preamble, propagation...)
R_n = 0.5 # 0<R_n<1 andel av bærebølger 

r = 1000 # r = 100-4000m, range
c1 = 1500 # Speed of sound, 1500 m/s
t_d = r/c1 # Transmission delay t_d = r/c
log = np.log # The natural logarithm
log2 = np.log2 # Log base 2

kv = 3 # no idea
p_lt = 0.001
gamma = 31.62

n = 3 # dimension of hessian / gradient. ex. 3x3 and 3x1
xi = [512, 4, 5]
N_list = np.array(xi[0])
M_list = np.array(xi[1])
m_list = np.array(xi[2])

hi = np.identity(n)# if n=3 -> they are 3x3 identity matrices
i = 1
t = 1
# x[2] = m, x[1] = M, x[0] = N
def cost_function(x):
    #return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d))) - t*log(-(x[0]-600))
    #return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d))) + max(0, x[2]-(4*B)/((1+p_c)*x[0]))**2 + max(0,x[1]-64)**2 + max(0, log(x[2]) + log(R_n) + log(x[0]) + log(log2(x[1])) + (1/R_c) * ( log(0.2) - (3*gamma)/(2*(x[1]-1)))-log(p_lt))**2
    const_N = 500
    const_M = 64
    const_m = (4*B)/((1+p_c)*x[0])
    
    # if (x[0]-const_N) > 0:
    #      x[0] = const_N - 1E-9
    # if (x[2]-2) > 0:
    #      x[2] = 2 - 1E-9
    # if (x[1]-const_m) > 0:
    #      x[1] = const_m - 1E-9

    
    #return t*(x[0]**2-500*x[0]+x[1]**2-30*x[1]+5*x[2]**2-40*x[2]) - (  log(-(x[0]-const_N))  )
    return t*(-(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))) - (  log(-(2-x[1]))+log(-(1-x[2]))+log(-(1+2-x[0]))+log(-(x[0]-700)) +log(-(   log(x[1])+log(x[0])+log(R_n)+log(log2(x[1])) + (1/R_c)*(log(0.2) - (3*gamma)/(2*(x[1]-1))) - log(p_lt)   ) ) )

def faktisk_cost(x):
     return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))
c = 1E-6 #??

cost_list = np.array(faktisk_cost(xi))

alfa = 0.01
beta = 0.6
iteration = 1
nu = 0.01
m = 5
# a, b, c loop 
while (m/t) > epsilon:
    print("main loop i: ", i)

    
    #a - calculate with newton
    j = 1
    k = 1
    d = np.array([
        [1],
        [1],
        [1]
    ])
    while np.linalg.norm(d) > epsilon and j < 1000:
        
        xi_old = xi
        print("\n newton loop j:", j)
        hessian = nd.Hessian(cost_function)(xi)
        if np.isnan(hessian[0][0]) == 1:
            print("====ERROR====")
            t=9999999
            break
        gradient = nd.Gradient(cost_function)(xi)
        d = -np.linalg.inv(hessian)@gradient
        xi = xi + d
        N_list = np.append(N_list, xi[0])
        M_list = np.append(M_list, xi[1])
        m_list = np.append(m_list, xi[2])
        iteration += 1
        j = j + 1
        print("h:\n", hessian)
        print("g: \n", gradient)
        print("xi: \n", xi)
        print("---")

    print("=====DONE NR=====")

    #b - update x 
    #xi = xi # ? 

    #c - update t 
    t = (22/13)*t


    
    print("t :", t)
    i+=1
print(xi_old)
print("====DONE MAIN!====")   

figure, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14, 8))
ax1.plot(np.arange(iteration), N_list, "g")

ax2.plot(np.arange(iteration), M_list, "r")

ax3.plot(np.arange(iteration), m_list, "b")

#plt.legend(["M"])
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plt.xlabel('iterations') 
#figure.suptitle("Newton's Method for: \n $x^2 - 500x + y^2 - 30y + 5z^2 - 60z$", fontsize=16)
figure.suptitle("Newton's Method for: \n cost function. Using penalty functions on all", fontsize=16)
ax1.set_title('$N$')
ax2.set_title('$M$')
ax3.set_title('$m$')
ax4.set_title("Function value")

#plt.xticks(np.arange(iterations))
plt.show()
#print(M_list)
