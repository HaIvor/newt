import numpy as np
from scipy.optimize import minimize

# Values for the cost function

R_c = 0.5 # Coding rate
B = 12000 # Bandwidth
n_x = 0.1 # Non-data subcarriers, Blir ikke brukt?
p_c = 0.25 # Cyclic prefix fraction
t_oh = 0.1 # Overhead (preamble, propagation...)
R_n = 0.5 # 0<R_n<1 andel av bærebølger t
r = 1000 # r = 100-4000m, range
c1 = 1500 # Speed of sound, 1500 m/s
t_d = r/c1 # Transmission delay t_d = r/c
f_c = 24000 # 24 kHz
p_lt = 0.001
gamma = 31.62
eta = 1

log = np.log # The natural logarithm
log2 = np.log2 # Log base 2

# Starting values 
start_N = 512
start_M = 4
start_m = 5



# x[0] = N, x[1] = M, x[2] = m 
def f(x):
    return -(log(x[2])+log(R_c)+log(B)+log(R_n)+log(x[0])+log(log2(x[1]))-log(x[2]*(1+p_c)*x[0]+B*(t_oh+t_d)))

# Defined as g_i(x) > 0 (opposite of barrier method)
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 400},
        {'type': 'ineq', 'fun': lambda x: 2000 - x[0]},
        {'type': 'ineq', 'fun': lambda x: x[1] - 2},
        {'type': 'ineq', 'fun': lambda x: 64 - x[1]},
        {'type': 'ineq', 'fun': lambda x: x[2] - 1},
        {'type': 'ineq', 'fun': lambda x: 40 - x[2]},
        {'type': 'ineq', 'fun': lambda x: -((x[2] * (x[0]/eta)*(log2(x[1])) * (0.2*np.exp(-((3*100)/(2* (x[1] - 1))))) ** (1 / R_c)) - 0.1)},
        {'type': 'ineq', 'fun': lambda x: -(log(x[1])+log(R_n) + log(x[0]) + log(log2(x[1])) + (1/R_c)*(log(0.2)-(3*gamma)/(2*(x[1]-1))) - log(p_lt))})


# minimize f with initial guess x0 with constraints
res = minimize(f, (start_N,start_M, start_m), constraints=cons) 


print("optimal verdi: \n")
print(f" N = {res.x[0]}, M = {res.x[1]}, m = {res.x[2]} \n\n with starting values = [{start_N}, {start_M}, {start_m}]")
print(f"\n cost function is then equal to: {res.fun}")

# Values coming from newton's method, ignore 
# con123, 6
print(f([1343.32638092,5.84360854,25.99181974]))
# con123
print(f([1382.96208666,28.34026602,23.12493068]))
print("done")
