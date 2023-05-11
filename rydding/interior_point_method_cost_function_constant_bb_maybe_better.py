import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt
import sympy as sym
import time
import sys 
start_time = time.time()


xi = np.array([
    [700],
    [4],
    [5]
])

def calculateHessianAndGradient(xi):

    R_c = 0.5 # Coding rate
    B = 12000 # Bandwidth
    n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
    p_c = 0.25 # Cyclic prefix fraction
    t_oh = 0.1 # Overhead (preamble, propagation...)
    R_n = 0.5 # 0<R_n<1 andel av bærebølger 

    r = 1000 # r = 100-4000m -emil
    c1 = 1500 # Speed of sound, 343 m/s
    t_d = r/c1 # Transmission delay t_d = r/c

    p_lt = 0.001
    gamma = 31.62 # 20db
    eta = 1

    

    x1, x2, x3 = sym.symbols('x1 x2 x3')

    N = x1
    m = x2
    M = x3
    # Federico constraints
    constraint1 = sym.log(-(1 - m)) + sym.log(-(m - 40)) #  1 < m < 40
    constraint2 = sym.log(-(N - 2000)) + sym.log(-(400 - N)) # 400 < N < 2000
    constraint3 = sym.log(-(2 - M)) + sym.log(-(M - 64)) # 2 < M < 64
    constraint4 = sym.log(-((m * (N / eta) * (sym.log(M, 2)) * (0.2 * sym.exp(-((3 * 100) / (2 * (M - 1))))) ** (1 / R_c)) - 0.1))

    # Defining the objective function we want to minimize
    # This example, arbitrary function with three constraints. 
    function_without_constraint = -(sym.log(x3)+ sym.log(R_c) + sym.log(B) + sym.log(R_n) + sym.log(x1) + sym.log(sym.log(x2, 2)) - sym.log(x3*(1+p_c)*x1 + B*(t_oh + t_d)))
    function = bb*( function_without_constraint ) - (constraint1+constraint2+constraint3+constraint4) 


    # Function value
    function_value = function.evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]}) 

    # Derivating f(x) for x1, x2, x3 (algebraic answer, without values)
    der_x1 = function.diff(x1)
    der_x2 = function.diff(x2)
    der_x3 = function.diff(x3)

    # Putting values into the derivatives
    der_x1_values = function.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2_values = function.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3_values = function.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the gradient of the objective function
    gradient_values = np.array([
        [der_x1_values],
        [der_x2_values],
        [der_x3_values]
    ], dtype=np.float32)

    # Derivating the objective function further to get the hessian
    der_x1x1_values = der_x1.diff(x1).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x2_values = der_x1.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x2x2_values = der_x2.diff(x2).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx1x3_values = der_x1.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_x3x3_values = der_x3.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})
    der_crossx2x3_values = der_x2.diff(x3).evalf(subs={x1: xi[0][0], x2: xi[1][0], x3: xi[2][0]})

    # Putting the derivatives together in a matrix so we get the hessian of the objective function
    hessian_values = np.array([
        [der_x1x1_values, der_crossx1x2_values, der_crossx1x3_values],
        [der_crossx1x2_values, der_x2x2_values, der_crossx2x3_values],
        [der_crossx1x3_values, der_crossx2x3_values, der_x3x3_values]
    ], dtype=np.float32)

    # Check if the answer is feasible. A too big t value can cause numerical difficulties 
    if sym.im(function_value) != 0:
        print('=========COMPLEX ANSWER, NOT FEASIBLE, MAYBE DECREASE t =============')
        print("f(x): ", function_value)
        sys.exit()

    return gradient_values, hessian_values


accuracy = 1e-5 # How exact one wants the end result to be.
m = 1 # number of constraints 
i = 1 # main loop iteration start
x1_list = np.array(xi[0]) # Used for plotting
x2_list = np.array(xi[1]) # Used for plotting
x3_list = np.array(xi[2]) # Used for plotting

NR_iteration = 1 # All the Newton's method interations, used for plotting x-axis

# Backtracking values
alpha = 0.3
beta = 0.7

bb = 2.9# starting value barrier steepness, too large starting point will result in errors.

epsilon = 0.3

gi = np.zeros((3,1))
hessian = np.identity(3)

# So the code goes in the inner loop after t is changed 
d = np.array([
    [1],
    [1],
    [1]
])

j = 1 # Reset temporary inner loop number 

# Inner loop (Newton's method) 
while  j < 200:
    
    print("\n newton loop j:", j)
   
    print("epsilon: ", epsilon)
    # Newton's method + backtracking 

    
    xi = (1-epsilon)*xi + epsilon*np.linalg.inv(hessian)@gi
    gradient, hessian = calculateHessianAndGradient(xi)
    gi = (hessian@xi - gradient)
    x1_list = np.append(x1_list, xi[0])
    x2_list = np.append(x2_list, xi[1])
    x3_list = np.append(x3_list, xi[2])
    
    NR_iteration += 1
    j = j + 1
    print("h:\n", hessian)
    print("g: \n", gradient)
    print("xi: \n", xi)
    print("---")
print("=====DONE NR=====")



# Plotting stuff 
figure, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(14, 8))

ax1.plot(np.arange(NR_iteration), x1_list, "g")

ax2.plot(np.arange(NR_iteration), x2_list, "r")

ax3.plot(np.arange(NR_iteration), x3_list, "k")

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
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()