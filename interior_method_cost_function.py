import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt
import sympy as sym
log = np.log


m = 3 # number of constraints 
xi = [500, 4, 5]
def calculateHessianAndGradient(xi):
    R_c = 0.5 # Coding rate
    B = 12 # Bandwidth
    n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
    p_c = 0.25 # Cyclic prefix fraction
    t_oh = 0.1 # Overhead (preamble, propagation...)
    R_n = 0.5 # 0<R_n<1 andel av bærebølger 

    r = 200 # r = 100-4000m -emil
    c1 = 1500 # Speed of sound, 343 m/s
    t_d = r/c1 # Transmission delay t_d = r/c

    p_lt = 0.001
    gamma = 31.62

    x1, x2, x3 = sym.symbols('x1 x2 x3')
    function = t*( -(sym.log(x3)+ sym.log(R_c) + sym.log(B) + sym.log(R_n) + sym.log(x1) + sym.log(sym.log(x2, 2)) - sym.log(x3*(1+p_c)*x1 + B*(t_oh + t_d))    )  )- (  sym.log(-(x2-64)) + sym.log(-(x3-4*B/(x1*(1+p_c)))) + sym.log(  sym.log(x3)+sym.log(x1)+sym.log(R_n)+sym.log(sym.log(x2, 2)) + (1/R_c)*(sym.log(0.2)-(3*gamma)/(2*(x2-1))) - sym.log(p_lt)    ) + sym.log(x1-600)    )

    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)
    der_x3 = function.diff(x3)

    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_x3_values = function.diff(x3).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})

    gradient_values = np.array([
        [der_x1_values, der_x2_values, der_x3_values]
    ], dtype=np.float32)

    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_crossx1x3_values = der_x1.diff(x3).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_x3x3_values = der_x3.diff(x3).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})
    der_crossx2x3_values = der_x2.diff(x3).evalf(subs={x1: xi[0], x2: xi[1], x3: xi[2]})

    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)
    
    return gradient_values, hessian_values


t = 0.1
epsilon = 1e-5

n = 3 #dimensions 
i = 1
nu = 0.01
x1_list = np.array(xi[0])
x2_list = np.array(xi[1])
x3_list = np.array(xi[2])
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
        d = -np.linalg.inv(hessian)@gradient.reshape(3,)
        print("d:", d)
        p=1
        #backtracking 
        # while cost_function(xi + p*d) > cost_function(xi)+alpha*p*np.transpose(gradient)@d:
        #     p = beta*p
        #     print("new p: ", p)
        
        xi = xi + p*d
        x1_list = np.append(x1_list, xi[0])
        x2_list = np.append(x2_list, xi[1])
        x3_list = np.append(x3_list, xi[2])
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
    

figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14, 8))

ax1.plot(np.arange(iteration), x1_list, "g")

ax2.plot(np.arange(iteration), x2_list, "r")

ax3.plot(np.arange(iteration), x3_list, "k")

ax1.grid()
ax2.grid()
ax3.grid()

ax1.set_xlabel('iterations')
ax2.set_xlabel('iterations')
ax3.set_xlabel('iterations')

ax1.set_title('$x_1$')
ax2.set_title('$x_2$')
ax3.set_title('$x_3$')

figure.suptitle("Interior method f(x1, x2, x3)", fontsize=16)

plt.show()