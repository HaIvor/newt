import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt
import sympy as sym
log = np.log

xi = [0, 0]
def calculateHessianAndGradient(xi):
    x1, x2 = sym.symbols('x1 x2')
    function = t*(x1-x2)-sym.log(-(x1**2+x2**2-1))

    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)

    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0], x2: xi[1]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0], x2: xi[1]})

    gradient_values = np.array([
        [der_x1_values, der_x2_values]
    ], dtype=np.float32)

    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0], x2: xi[1]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0], x2: xi[1]})
    der_cross_values = der_x1.diff(x2).evalf(subs={x1: xi[0], x2: xi[1]})

    hessian_values = np.array([
        [der_x1x1_values, der_cross_values],
        [der_cross_values, der_x2x2_values]
    ], dtype=np.float32)
    
    return gradient_values, hessian_values


t = 1
epsilon = 1e-5
m = 1 # number of constraints 
n = 2 #dimensions 
i = 1
nu = 0.01
x1_list = np.array(xi[0])
x2_list = np.array(xi[1])
iteration = 1
alpha = 0.01
beta = 0.5
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
        gradient = calculateHessianAndGradient(xi)[0]
        hessian = calculateHessianAndGradient(xi)[1]
        print("lol")

        print(np.linalg.inv(hessian))
        print(gradient)
        d = -np.linalg.inv(hessian)@gradient.reshape(2,)
        print("d:", d)
        p=1
        #backtracking 
        # while cost_function(xi + p*d) > cost_function(xi)+alpha*p*np.transpose(gradient)@d:
        #     p = beta*p
        #     print("new p: ", p)
        
        xi = xi + p*d
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

ax1.set_xlabel('iterations')
ax2.set_xlabel('iterations')

ax1.set_title('$x_1$')
ax2.set_title('$x_2$')

figure.suptitle("Interior method + backtracking: $f(x_1,x_2) = x_1 - x_2$ \n $s.t. x_1^2 + x_2^2 -1 < 0$ ", fontsize=16)

plt.show()