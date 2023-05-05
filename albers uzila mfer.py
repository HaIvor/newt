import numpy as np
import matplotlib.pyplot as plt

def f(x):
    '''Function to be minimized'''
    return (x[0]-5)**2 + (x[1]-6)**2

def g(x, t):
    '''Gradient of f(x) + 1/t * ϕ(x)'''

    x1, x2 = x
    
    dcdx1 = []
    dcdx1.append(2*x1 / (x1**2 - 4))
    dcdx1.append(-np.exp(-x1) / (np.exp(-x1) - x2))
    dcdx1.append(1 / (x1 + 2*x2 - 4))
    dcdx1.append(-1 / (-x1))
    dcdx1.append(0 / (-x2))
    
    dcdx2 = []
    dcdx2.append(0 / (x1**2 - 4))
    dcdx2.append(-1 / (np.exp(-x1) - x2))
    dcdx2.append(2 / (x1 + 2*x2 - 4))
    dcdx2.append(0 / (-x1))
    dcdx2.append(-1 / (-x2))
    
    dthetadx1 = 0
    dthetadx2 = 0
    for a, b in zip(dcdx1, dcdx2):
        dthetadx1 -= a
        dthetadx2 -= b
    
    grad = np.zeros(2)
    grad[0] = 2 * (x1-5) + 1/t * dthetadx1
    grad[1] = 2 * (x2-6) + 1/t * dthetadx2
    
    return grad

def H(x, t):
    '''Hessian of f(x) + 1/t * ϕ(x)'''
    
    x1, x2 = x
    
    terms = []
    terms.append(x1**2 - 4)
    terms.append(np.exp(-x1) - x2)
    terms.append(x1 + 2*x2 - 4)
    
    hess = np.zeros((2, 2))
    hess[0][0] = 2 - 1/t * ((-2 * x1**2 - 8)/(terms[0]**2) - \
                            (x2 * np.exp(-x1))/(terms[1]**2) - \
                            1/(terms[2]**2))
    
    hess[0][1] = -1/t * (-np.exp(-x1)/(terms[1]**2) - \
                         2/(terms[2]**2))
    
    hess[1][0] = hess[0][1]
    hess[1][1] = 2 - 1/t * (-1/(terms[1]**2) - 4/(terms[2]**2))
    
    return hess
    


def BarrierMethod(x_init, m, t, nu=0.01, tol_barrier=1e-5, tol_newton = 1e-5, max_iter=1000):
    '''Main algorithm for Barrier Method'''
    
    x = x_init               # store initial value
    xs = [x]                 # initialize tabulation of x for each iteration
    fs = [f(x)]              # initialize tabulation of function value at x
    duality_gap = [m/t]      # initialize tabulation of duality gap
    k = 0                    # number of iterations
    print(f'Initial condition: x = {x}, f(x) = {fs[k]:.4f}\n')

    # loop until stopping criterion is met
    while m / t > tol_barrier:
        # centering step: Newton Algorithm
        i = 0
        d = np.array([[1], [1]])
        while np.linalg.norm(d) > tol_newton and i < max_iter:
            gx = g(x, t)
            Hx = H(x, t)
            d = -np.dot(np.linalg.inv(Hx), gx)
            x = x + d
            xs.append(x)
            i += 1
        
        # update parameter t
        t = (1 + 1/(13 * np.sqrt(nu))) * t
        
        # update tabulations
        duality_gap.append(m/t)
        fs.append(f(x))
        k += 1
        
        # print result
        print(f'Iteration: {k} \t x = {x}, f(x) = {fs[k]:.4f}, gap = {duality_gap[k]:.4f}')
        
    xs = np.array(xs)
    return xs, fs, duality_gap

def plot_feasible_set(x, y):
    '''Plot feasible set to be color coded'''
    c1 = lambda a, b : a**2 - 4
    c2 = lambda a, b : np.exp(-a) - b
    c3 = lambda a, b : a + 2*b - 4
    c4 = lambda a, b : -a
    c5 = lambda a, b : -b
    plt.imshow(
        ((c1(x,y)<=0) & (c2(x,y)<=0) & (c3(x,y)<=0) & (c4(x,y)<=0) & (c5(x,y)<=0)).astype(int),
        extent=(x.min(),x.max(),y.min(),y.max()),
        origin='lower',
        cmap='inferno'
    )
    
def plot_contour(x_min, x_max, y_min, y_max):
    '''Plot contour of the objective function'''
    delta = 0.025
    x = np.arange(x_min, x_max, delta)
    y = np.arange(y_min, y_max, delta)
    x, y = np.meshgrid(x, y)
    z = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_coor = x[i][j]
            y_coor = y[i][j]
            z[i][j] = f(np.array([[x_coor], [y_coor]]))
    CS = plt.contour(x, y, z)
    plt.clabel(CS, fmt='%1.2f')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Contour of $f(x)$ along with the Feasible Set and Iteration Path')
    plt.show()
    
def plot_learning_curve(fx, duality_gap):
    '''Plot learning curve of the algorithm'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.plot(fs, c='m')
    ax1.scatter(len(fs)-1, fs[-1], c='c')
    ax1.set(
        title='$f(x)$ for each Iteration',
        xlabel='Number of Iterations',
        ylabel='$f(x)$',
        xlim=0
    )
    ax2.plot(duality_gap, c='m')
    ax2.scatter(len(duality_gap)-1, duality_gap[-1], c='c')
    ax2.set(
        title='Duality Gap for each Iteration',
        xlabel='Number of Iterations',
        ylabel='Duality Gap',
        xlim=0
    )
    plt.show()
    
def plot_all(xs, fs, duality_gap):
    '''Plot all results'''
    
    plt.figure(figsize=(10, 10))
    plt.plot(xs[:,0], xs[:,1], 'm')
    plt.scatter(xs[-1][0], xs[-1][1], color='c')
    
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    d = np.linspace(min(x_min, y_min, -0.25), max(x_max, y_max, 2.25), 2000)
    x, y = np.meshgrid(d, d)
    
    plot_feasible_set(x, y)
    plot_contour(x.min(), x.max(), y.min(), y.max())
    plot_learning_curve(fs, duality_gap)
x_init = np.array([0.50, 0.75])
xs, fs, duality_gap = BarrierMethod(x_init, 5, 0.1)
plot_all(xs, fs, duality_gap)