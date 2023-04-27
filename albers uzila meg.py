import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt

log = np.log

def cost_function(x):
    return t*(x[0]+x[1]) - (log(-(x[0]**2+x[1]**2-1)))

def g(x, t):
    x1, x2 = x
    grad = np.zeros(2)
    grad[0] = (2*x1)/(t*(-x1**2-x2**2+1))+1
    grad[1] = (2*x2)/(t*(-x1**2-x2**2+1))+1
    return grad

def H(x, t):
    x1, x2 = x
    hess = np.zeros((2,2))
    hess[0][0] = (2*(x1**2-x2**2+1))/(t*(x1**2+x2**2-1)**2)
    hess[1][1] = (2*(x2**2-x1**2+1))/(t*(x1**2+x2**2-1)**2)
    hess[0][1] = (4*x1*x2)/(t*(x2**2+x1**2 - 1)**2)
    hess[1][0] = hess[0][1]
    return hess
xi = [-0.3, 0.7]
t = 1
epsilon = 1e-5
m = 1 # number of constraints 
n = 2 #dimensions 
i = 1
nu = 0.01
x1_list = np.array(xi[0])
x2_list = np.array(xi[1])
iteration = 1
# a, b, c loop 
while (m/t) > epsilon:
    print("main loop i: ", i)

    
    #a - calculate with newton
    j = 1
    k = 1
    d = np.array([
        [1],
        [1]
    ])
    while np.linalg.norm(d) > epsilon and j < 1000:
        
        xi_old = xi
        print("\n newton loop j:", j)
        hessian = H(xi, t) 
        print(hessian)
        gradient = g(xi, t)
        d = -np.linalg.inv(hessian)@gradient
        xi = xi + d
        x1_list = np.append(x1_list, xi[0])
        x2_list = np.append(x2_list, xi[1])
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
    

figure, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 8))

ax1.plot(np.arange(iteration), x1_list, "g")

ax2.plot(np.arange(iteration), x2_list, "r")

ax1.grid()
ax2.grid()

ax1.set_xlabel('x1')
ax2.set_xlabel('x2')

figure.suptitle("Interior method - no cheating, had to calculate hessian and gradient by hand 😏😎", fontsize=16)

plt.show()