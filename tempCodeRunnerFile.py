while function_xi_t_deltax > (function_xi + alpha*p*np.transpose(gradient_values)@deltax):
    #     xi_t_deltax = xi + p*deltax 
    #     function_xi_t_deltax = function.evalf(subs={x1: xi_t_deltax[0][0], x2: xi_t_deltax[1][0], x3: xi_t_deltax[2][0]})
    #     p = p*beta