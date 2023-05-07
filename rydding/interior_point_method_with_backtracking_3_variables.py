import numpy as np 
import numdifftools as nd 
import matplotlib.pyplot as plt
import sympy as sym
import time
start_time = time.time()


xi = np.array([
    [2],
    [2],
    [2]
])

def calculateHessianAndGradient(xi):

    x1, x2, x3 = sym.symbols('x1 x2 x3')

    # Defining the objective function we want to minimize
    # This example, arbitrary function with three constraints. 
    function = (x1*sym.log(x1)+x2*sym.log(x2)+x3*sym.log(x3))-(1/t)*(sym.log(-(-x1)) + sym.log(-(-x2)) + sym.log(-(-x3)) )


    #function value
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

    # Backtracking f(x + t delta x) > f(x) alpha t delta^T delta x 
    epsilon = 1
    deltax = -1.0*np.linalg.inv(hessian_values)@gradient_values
    xEpsilondeltax = xi + epsilon*deltax
    f_xEpsilondeltax = function.evalf(subs={x1: xEpsilondeltax[0][0], x2: xEpsilondeltax[1][0], x3: xEpsilondeltax[2][0]})

    while f_xEpsilondeltax > function_value+alpha*epsilon*np.transpose(gradient_values)@deltax:
        xEpsilondeltax = xi + epsilon*deltax
        f_xEpsilondeltax = function.evalf(subs={x1: xEpsilondeltax[0][0], x2: xEpsilondeltax[1][0], x3: xEpsilondeltax[2][0]})
        epsilon = beta*epsilon

    return gradient_values, hessian_values, epsilon

t = 1 # starting value t
accuracy = 1e-5 # How exact one wants the end result to be (lower epsilon => more exact)
m = 1 # number of constraints 
i = 1 # main loop iteration start
x1_list = np.array(xi[0]) # Used for plotting
x2_list = np.array(xi[1]) # Used for plotting
x3_list = np.array(xi[2]) # Used for plotting

NR_iteration = 1 # All the Newton's method interations, used for plotting x-axis

# Backtracking values
alpha = 0.3
beta = 0.7

# Outer loop
while (m/t) > accuracy:
    print("main loop i: ", i)

    # So the code goes in the inner loop after t is changed 
    d = np.array([
        [1],
        [1],
        [1]
    ])

    j = 1 # Reset temporary inner loop number 

    # Inner loop (Newton's method) 
    while np.linalg.norm(d) > accuracy and j < 1000:
        
        print("\n newton loop j:", j)
        gradient, hessian, epsilon = calculateHessianAndGradient(xi)

        d = -np.linalg.inv(hessian)@gradient
        print("epsilon: ", epsilon)

        # Newton's method + backtracking 
        
        xi = xi + epsilon*d

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

    # Increasing t 
    t = (22/13)*t

    print("t :", t)
    i+=1
print("====DONE MAIN!====")   
    
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