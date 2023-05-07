import random 
import numpy as np 
import numdifftools as nd
import matplotlib.pyplot as plt
import sympy as sym

epsilon = 0.1

def calculateHessianAndGradient(xi):

    x1, x2, x3 = sym.symbols('x1 x2 x3')

    # Defining the objective function we want to minimize
    # This example, arbitrary function with one constraint x1^2+x2^2+x3^2 < 2. 
    function = x1**2-500*x1+x2**2-30*x2+5*x3**2-60*x3

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
    
    return gradient_values, hessian_values, function_value


xi = np.array([ 
    [10], # N_start
    [20], #M_start
    [70] # m_start
]) 
N_list = np.array(xi[0])
M_list = np.array(xi[1])
m_list = np.array(xi[2])

i = 1
c = 1E-6

cost_list = np.array([])
accuracy = 1e-5

iterations = 1
lamba_2 = 1

# Stopping criterion -> "The Newton decrement"^2 / 2 > accuracy value
while (lamba_2/2) > accuracy:

    gradient, hi, function_value = calculateHessianAndGradient(xi)

    lamba_2 = np.transpose(gradient)@np.linalg.inv(hi)@gradient

    if (np.abs(np.linalg.eigvals(hi)) < c).all():
            hi = c*np.identity(3) # If hessian too small -> estimate hessian as identity matrix with variable c in.

    print("hi: \n", hi)
    print("gradient: \n", gradient)

    # Newton's method 
    xi = (1-epsilon)*xi+epsilon*(np.linalg.inv(hi))@(hi@xi-gradient)

    # For plotting
    N_list = np.append(N_list, xi[0][0])
    M_list = np.append(M_list, xi[1][0])
    m_list = np.append(m_list, xi[2][0])
    cost_list = np.append(cost_list, function_value)
    iterations = iterations + 1

print("xi is then in the end: \n xi=[N,M,m] \n", xi)
print("cost functionen is: \n", function_value)

figure, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(14, 8))
ax1.plot(np.arange(iterations), N_list, "g")

ax2.plot(np.arange(iterations), M_list, "r")

ax3.plot(np.arange(iterations), m_list, "b")

ax4.plot(np.arange(iterations-1), cost_list, "k")

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