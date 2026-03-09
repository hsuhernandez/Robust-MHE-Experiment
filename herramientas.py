import casadi as cs
import numpy as np
import casadi.tools as ctools
import matplotlib.pyplot as plt
import time
from scipy import linalg

def variables_y_parametros(N):
    """Función que define las estructuras de variables y parámetros del problema de optimización, donde:
        N es un diccionario con las dimensiones del sistema, donde:
            N['x'] es la cantidad de estados
            N['u'] es la cantidad de entradas
            N['y'] es la cantidad de mediciones
            N['w'] es la cantidad de ruido de proceso (si no se especifica, es igual a N['x'])
            N['v'] es la cantidad de ruido de medición (siempre es igual a N['y'])
            N['t'] es el tamaño de ventana
       Y devuelve:
        opt_var las variables del problema de optimización 
        opt_par los parámetros del problema de optimización
    """
    
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    N['v'] = N['y']

    # Variables (desconozco estados y ruidos)
    opt_var = ctools.struct_symSX([ctools.entry('x', shape=(N['x'], 1), repeat=N['t']),
                                ctools.entry('v', shape=(N['v'], 1), repeat=N['t']),
                                ctools.entry('w', shape=(N['w'], 1), repeat=N['t'] - 1)
                                ])
    
    # Parámetros (conozco mediciones, predefino x0 y P0)
    opt_par = ctools.struct_symSX([ctools.entry('y', shape = (N['y'], 1), repeat = N['t']),
                                ctools.entry('u', shape = (N['u'], 1), repeat = N['t']-1),
                                ctools.entry('x0bar', shape = (N['x'], 1)),
                                ctools.entry('P0_X', shape=(N['x'], N['x']))])
    
    
  
    return opt_var, opt_par

def costo_y_restricciones(N, opt_var, opt_par, f, h, Q, R = None, rho_huber = None, 
                          bounds = {}, w_pos = False):
    """Función que define la función de costo y las restricciones del problema de optimización, donde:
        N es un diccionario con las dimensiones del sistema, donde:
            N['x'] es la cantidad de estados
            N['y'] es la cantidad de mediciones
            N['w'] es la cantidad de ruido de proceso (si no se especifica, es igual a N['x'])
            N['v'] es la cantidad de ruido de medición (siempre es igual a N['y'])
            N['t'] es el tamaño de ventana
        opt_var son las variables del problema de optimización
        opt_par son los parámetros del problema de optimización
        f es la función de proceso
        h es la función de medición
        Q es la matriz de covarianza del ruido de proceso
        R es la matriz de covarianza del ruido de medición
        rho_huber es el parámetro de Huber (si no se especifica, se considera None)
        bounds es un diccionario con las cotas de cada variable, donde:
            bounds['lbx'] y bounds['ubx'] son las cotas inferior y superior de las variable x
            bounds['lbv'] y bounds['ubv'] son las cotas inferior y superior de las variable v
            bounds['lbw'] y bounds['ubw'] son las cotas inferior y superior de las variable w
        (si no se especifican, se consideran desde -∞ hasta ∞)
        w_pos es un booleano que indica si el ruido de proceso es positivo (True) o no (False)
       Y devuelve:
        J la función de costo
        con las restricciones del problema de optimización
        lbg la cota inferior de las restricciones
        ubg la cota superior de las restricciones
        lbx la cota inferior de las variables
        ubx la cota superior de las variables
    """

    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    N['v'] = N['y']

    # Defino las cotas para cada variable (si no están definidas se consideran desde -∞ hasta ∞)
    if 'lbx' not in bounds:
        bounds['lbx'] = -cs.DM.inf(N['x']) # cota inferior para x
    if 'ubx' not in bounds:
        bounds['ubx'] = cs.DM.inf(N['x']) # cota superior para x
    if 'lbv' not in bounds:
        bounds['lbv'] = -cs.DM.inf(N['v']) # cota inferior para v
    if 'ubv' not in bounds:
        bounds['ubv'] = cs.DM.inf(N['v']) # cota superior para v
    if 'lbw' not in bounds:
        bounds['lbw'] = -cs.DM.inf(N['w']) # cota inferior para w
    if 'ubw' not in bounds:
        bounds['ubw'] = cs.DM.inf(N['w']) # cota superior para w

    # Costo de arribo
    J = cs.mtimes([(opt_var['x', 0] - opt_par['x0bar']).T,
                    opt_par['P0_X'],
                    (opt_var['x', 0] - opt_par['x0bar'])])

    state_constraints = [] # restricciones de estados
    sc_lb, sc_ub = [], [] # cota inferior y superior para restricción de estado 
    measurement_constraints = [] # restricciones de medición
    mc_lb, mc_ub = [], [] # cota inferior y superior para restricción de medición 

    lbx = [] # cota inferior para variables
    ubx = [] # cota superior para variables

    # Invierto matrices de covarianza para no invertirlas en cada iteración
    Q_inv = linalg.inv(Q) # inversa de Q
    R_inv = linalg.inv(R) # inversa de R

    if w_pos:
        mu_w = w_pos*np.sqrt(2 / np.pi)

    for i in range(N['t'] - 1):
        # costo para el ruido de proceso
        w_i = opt_var['w', i]
        if w_pos:
            w_i -= mu_w*cs.DM.ones(N['w'])
        J += cs.mtimes([w_i.T, Q_inv, w_i])
        # costo ruido de medición
        if rho_huber is not None:
            J += huber(opt_var['v', i], rho_huber)
        else:
            J += cs.mtimes([opt_var['v', i].T, R_inv, opt_var['v', i]])

        # restricciones
        # x_i+1 = f(x_i, u_i, w_i)
        state_constraints.append(opt_var['x', i+1] - f(opt_var['x',i], opt_par['u', i], opt_var['w',i]))
        sc_lb.append(cs.DM.zeros(N['x']))
        sc_ub.append(cs.DM.zeros(N['x']))
        #y_i = h(x_i) + v_i
        measurement_constraints.append(opt_par['y', i] - h(opt_var['x', i])- opt_var['v', i])
        mc_lb.append(cs.DM.zeros(N['y']))
        mc_ub.append(cs.DM.zeros(N['y']))

        # cotas para variables
        # lbx.extend([bounds['lbx'],bounds['lbv'], bounds['lbw']])
        # ubx.extend([bounds['ubx'], bounds['ubv'], bounds['ubw']])

    # Término extra  de medición
    if rho_huber is not None:
        J += huber(opt_var['v', -1], rho_huber)
    else:
        J += cs.mtimes([opt_var['v', -1].T, R_inv, opt_var['v', -1]])

    measurement_constraints.append(opt_par['y', -1] - h(opt_var['x', -1])- opt_var['v', -1])
    mc_lb.append(cs.DM.zeros(N['y']))
    mc_ub.append(cs.DM.zeros(N['y']))
    # lbx.extend([bounds['lbx'], bounds['lbv']])
    # ubx.extend([bounds['ubx'], bounds['ubv']])

    # Todas las restricciones
    con = state_constraints + measurement_constraints
    con = cs.vertcat(*con)
    # Todas las cotas de restricciones
    lbg = sc_lb + mc_lb
    lbg = cs.vertcat(*lbg)
    ubg = sc_ub + mc_ub
    ubg = cs.vertcat(*ubg)
    # Cotas para las variables x,v, y w
    lbx = N['t'] * [bounds['lbx']] + N['t'] * [bounds['lbv']] + (N['t'] - 1) * [bounds['lbw']]
    ubx = N['t'] * [bounds['ubx']] + N['t'] * [bounds['ubv']] + (N['t'] - 1) * [bounds['ubw']]
    # Concatenar las cotas en un único vector
    lbx = cs.vertcat(*lbx)
    ubx = cs.vertcat(*ubx)

    return J, con, lbg, ubg, lbx, ubx

def huber(a, rho):
    """Función que define la función de costo de Huber, donde:
        a es el vector de errores
        rho es el parámetro de Huber
       Y devuelve:
        la función de costo de Huber
    """
    norm_a = cs.norm_2(a)
    return cs.if_else(norm_a <= rho, norm_a**2, 2*rho*norm_a - rho**2)


def mhe(N, f, h, x_0, u, y, P, Q, R, opt_par, opt_var,
                 J, con, lbg=0, ubg=0, lbx =-cs.inf, ubx=cs.inf,
                 f_jacx=None,f_jacw=None,h_jacx=None, mhe_method = None, w_pos = False, **kwargs):
    """Función que resuelve el problema de optimización, donde:
        N es un diccionario con las dimensiones del sistema, donde:
            N['x'] es la cantidad de estados
            N['y'] es la cantidad de mediciones
            N['w'] es la cantidad de ruido de proceso (si no se especifica, es igual a N['x'])
            N['v'] es la cantidad de ruido de medición (siempre es igual a N['y'])
            N['t'] es el tamaño de ventana
        x_0 es el estado inicial
        f es la función de proceso
        h es la función de medición
        u es la entrada al sistema
        y es la salida del sistema
        P es la matriz de covarianza del estado
        Q es la matriz de covarianza del ruido de proceso
        R es la matriz de covarianza del ruido de medición
        opt_par son los parámetros del problema de optimización
        opt_var son las variables del problema de optimización
        J es la función de costo
        con son las restricciones del problema de optimización
        lbg es la cota inferior de las restricciones (por defecto 0)
        ubg es la cota superior de las restricciones (por defecto 0)
        lbx es la cota inferior de las variables (por defecto -∞)
        ubx es la cota superior de las variables (por defecto ∞)
        f_jacx es la función jacobiana de la función de proceso (por defecto None)
        f_jacw es la función jacobiana de la función de proceso (por defecto None)
        h_jacx es la función jacobiana de la función de medición (por defecto None)
        mhe_method es el método de actualización de matriz P, donde:
            'None': ningún método
            'KF': filtro de Kalman
            'AD-CF': ganancia adaptativa con factor de olvido constante
            'AD-CT': ganancia adaptativa con traza constante
            'AD-VF': ganancia adaptativa con factor de olvido variable
        w_pos es la desviación estándar del ruido de proceso (si es positivo)
        **kwargs son argumentos adicionales para el método de actualización de matriz P
       Y devuelve:
        x_mhe los estados estimados
        w_mhe los ruidos de proceso estimados
        v_mhe los ruidos de medición estimados
        t_mhe el tiempo de ejecución por ventana
        trace_Pinv la traza de la inversa de la matriz P
    """
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    N['v'] = N['y']

    Nsim = y.shape[1]  # cantidad de mediciones

    # Inicializo parámetros que son usados por el método de actualización de matriz P
    alpha = kwargs.get('alpha', 0.95)
    beta  = kwargs.get('beta', 1)
    Xi    = kwargs.get('Xi', 20)
    eta   = kwargs.get('eta', 3)
    sigma = kwargs.get('sigma', 100)
    c     = kwargs.get('c', 10e6)

    # Crear problema de optimización
    nlp = {'x':opt_var, # variables
        'p':opt_par, # parámetros
        'f':J, # función de costo
        'g':con # restricciones
        }
    # Configuración para el solver IPOPT
    opts = {"ipopt.print_level": 0, "print_time": False, 'ipopt.max_iter': 100}
    # Creación del solver de optimización no lineal (nlpsol) utilizando el método IPOPT
    solver = cs.nlpsol("solver", "ipopt", nlp, opts)

    # Inicializo variables de salida
    x_mhe = cs.DM.zeros(N['x'], Nsim)
    w_mhe = cs.DM.zeros(N['w'], Nsim - 1)
    v_mhe = cs.DM.zeros(N['v'], Nsim)
    t_mhe = []
    trace_Pinv = []
    # Inicialización primera ventana
    iteration_time = -time.time() # inicio del tiempo de ejecución de la primer ventana
    current_parameters = opt_par(0)
    current_parameters['y', lambda x: cs.horzcat(*x)] = y[:, :N['t']]
    current_parameters['u', lambda x: cs.horzcat(*x)] = u[:, :N['t'] - 1]
    current_parameters['P0_X'] = linalg.inv(P)
    current_parameters['x0bar'] = x_0
    # Inicialización de variables
    initialisation_state = opt_var(0)
    # Aplico solver a la primera ventana
    res = solver(p=current_parameters, x0=initialisation_state,
                 lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)


    solution = opt_var(res['x'])
    x_mhe[:, :N['t']] = solution['x', lambda x: cs.horzcat(*x)]
    w_mhe[:, :N['t'] - 1] = solution['w', lambda x: cs.horzcat(*x)]
    v_mhe[:, :N['t']] = solution['v', lambda x: cs.horzcat(*x)]
    trace_Pinv.append(np.trace(np.linalg.inv(P)))

    iteration_time += time.time() # tiempo de ejecución de la primer ventana
    t_mhe.append(iteration_time)

    x_0 = x_mhe[:, 1]  # Actualizo x_0 con el estado estimado de la primer ventana

    for i in range(1, Nsim-N['t']+1):
        iteration_time = - time.time()

        # Actualizo parámetros
        current_parameters['y',lambda x: cs.horzcat(*x)] = y[:,i:i+N['t']]
        current_parameters['u',lambda x: cs.horzcat(*x)] = u[:,i:i+N['t']-1]
        try:
            current_parameters['P0_X'] = linalg.inv(P)
        except np.linalg.LinAlgError:
            P_reg = P + 1e-12 * cs.DM.eye(N['x'])
            print("P no invertible,", mhe_method)
            current_parameters['P0_X'] = linalg.inv(P_reg)
        current_parameters['x0bar']= x_0 
        
        # Initialize the system with the shifted solution
        # initialisation_state["w",lambda x: cs.horzcat(*x),0:N['t']-2] = w_mhe[:,i:i+N['t']-2] # The shifted solution for the disturbances
        # initialisation_state["w",N['t']-2] = cs.DM.zeros(N['w'],1) # The last node for the disturbances is initialized with zeros
        # initialisation_state["x",lambda x: cs.horzcat(*x),0:N['t']-1] = x_mhe[:,i:i+N['t']-1] # The shifted solution for the state estimates
        # # The last node for the state is initialized with a forward simulation
        # phi0 = f(initialisation_state["x",N['t']-1], current_parameters["u",-1], initialisation_state["w",-1])
        # initialisation_state["x",N['t']-1] = phi0

        # Solución para la ventana actual
        res = solver(p=current_parameters, x0=initialisation_state,
                    lbg=lbg, ubg=ubg, ubx = ubx, lbx = lbx)
        solution = opt_var(res['x'])
        # Guardamos el estimado (solo nos interesa el último nodo del horizonte)
        x_mhe[:,N['t']-1+i] = solution['x',N['t']-1]
        w_mhe[:,N['t']-2+i] = solution['w',N['t']-2]
        v_mhe[:,N['t']-1+i] = solution['v',N['t']-1]

        # Actualizo x_0 y P según el método de MHE
        if mhe_method == None:
            x_0 = solution['x', 1]

        if mhe_method == 'KF':
            w = solution['w', 0]
            # if w_pos:
            #     mu_w = w_pos*np.sqrt(2 / np.pi)
            #     w += mu_w*cs.DM.ones(N['w'], 1)
            x_0, P = ekf_2(f,h,x_0, solution['x', 0], current_parameters['u', 0], w, 
                                   current_parameters['y',0], P, Q, R,
                                   f_jacx=f_jacx, f_jacw=f_jacw, h_jacx=h_jacx)

        if mhe_method == 'AD-CF': # Adaptative gain, constant forgetting factor
            xk = solution['x', 0]
            xPx = cs.mtimes([xk.T, P, xk])
            PxxP = cs.mtimes([P,xk, xk.T, P])
            P = 1/alpha * (P - (PxxP)/(alpha/beta + xPx))
            x_0 = solution['x', 1]

        if mhe_method == 'AD-CT': # Adaptative gain, constant trace
            xk = solution['x', 0]
            xPx = cs.mtimes([xk.T, P, xk])
            PxxP = cs.mtimes([P,xk, xk.T, P])
            alpha = 1/Xi * np.trace(P-xPx/(eta+xPx))
            beta = eta**-1 * alpha
            den = alpha/beta + xPx
            if np.abs(den) < 1e-8 or alpha < 1e-8:
                P = P + 1e-6 * cs.DM.eye(P.shape[0])  # fallback suave
            else:
                P = 1/alpha * (P - PxxP / den)
            x_0 = solution['x', 1]
        
        if mhe_method == 'AD-VF':
            xk = solution['x', 0]
            xPx = cs.mtimes([xk.T, P, xk])
            PxxP = cs.mtimes([P,xk, xk.T, P])
            # eps = 1e-6  # para evitar divisiones por cero
            meas_err = np.linalg.norm(current_parameters['y', 0] - h(xk), ord=2)**2 # + eps
            Nag = (1.0 + xPx) * (sigma / meas_err)
            alpha = 1.0 - 1.0/Nag
            # alpha = np.clip(alpha, 0.5, 0.99)
            W = P - PxxP/(1.0 + xPx)
            if 1/alpha*np.trace(W) <= c:
                P = 1/alpha * W
            else:
                P = W
            x_0 = solution['x', 1]
            
        trace_Pinv.append(np.trace(np.linalg.inv(P)))
        iteration_time += time.time()
        t_mhe.append(iteration_time)
    return x_mhe, w_mhe, v_mhe, t_mhe, trace_Pinv

def ekf_2(f, h, x_0, x, u, w, y, P, Q, R, f_jacx=None, f_jacw=None, h_jacx=None):
    # Defino jacobianos
    if f_jacx is None:
        f_jacx = jacobiano(f, 0)
    if f_jacw is None:
        f_jacw = jacobiano(f, 2)
    if h_jacx is None:
        h_jacx = jacobiano(h, 0)

    H0 = h_jacx(x)
    S = cs.mtimes([H0,P,H0.T])+R # covarianza de innovación (o residual)
    
    K = cs.mtimes([P,H0.T,linalg.inv(S)])
    P = cs.mtimes((cs.DM.eye(P.shape[0])-cs.mtimes(K,H0)),P)

    h0 = h(x)
    y_tilde = y - h0
    x_0 = x_0 + cs.mtimes(K, y_tilde-cs.mtimes(H0,x_0-x))

    x_0 = f(x_0, u, w)
    F = f_jacx(x, u, w)
    G = f_jacw(x, u, w)
    P = cs.mtimes([F,P,F.T]) + cs.mtimes([G,Q,G.T])

    return x_0, P

def resolver_ekf(N, f, h, x_0, u, y, P, Q, R, f_jacx=None,f_jacw=None,h_jacx=None, w_pos=False, rho_huber = None):
    """"""
    # Defino dimensiones del sistema
    if 'w' not in N:
            N['w'] = N['x']
    
    # Defino jacobianos
    if f_jacx is None:
        f_jacx = jacobiano(f, 0)
    if f_jacw is None:
        f_jacw = jacobiano(f, 2)
    if h_jacx is None:
        h_jacx = jacobiano(h, 0)

    Nsim = len(np.array(y)[0,:]) # Cantidad de mediciones
    x_ekf = cs.DM.zeros(N['x'], Nsim) # estimados de x
    x_ekf[:,0] = x_0 # estado inicial
    P_0 = P
    w_0 = cs.DM.zeros(N['w'], 1) # ruido de proceso inicial
    t_ekf = [] # tiempo de ejecución por ventana
    
    if w_pos:
        mu_w = w_pos*np.sqrt(2/np.pi)
        w_0 += mu_w*cs.DM.ones(N['w'],1) # ruido de proceso inicial

    for i in range(Nsim-1):
        iteration_time = -time.time()
        x, P_0 = ekf(f,h, x_ekf[:, i], u[:, i], w_0,  y[:,i],
                     P_0, Q, R, f_jacx=f_jacx, f_jacw=f_jacw, h_jacx=h_jacx, rho_huber=rho_huber)
        x_ekf[:,i+1] = x
        iteration_time += time.time()
        t_ekf.append(iteration_time)
    
    return x_ekf, t_ekf

def ekf(f, h, x, u, w, y, P, Q, R, f_jacx=None, f_jacw=None, h_jacx=None, rho_huber=None):
    """Función que implementa el filtro de Kalman extendido, donde:
        f es la función de proceso
        h es la función de medición
        x es el estado, xhat(k-1 | k-1)
        u es la entrada, u(k-1)
        w es el ruido de proceso, w(k-1)
        y es la medición, y(k)
        P es la matriz de covarianza del estado P(k-1 | k-1)
        Q es la matriz de covarianza del ruido de proceso
        R es la matriz de covarianza del ruido de medición
        f_jacx es la jacobiana de f respecto a x
        f_jacw es la jacobiana de f respecto a w
        h_jacx es la jacobiana de h respecto a x
        (si no se especifican, se calculan automáticamente)
       Y devuelve:
        x_upd el estado actualizado, xhat(k | k)
        P_upd la matriz de covarianza del estado actualizado, P(k | k)
    """
    # Defino jacobianos
    if f_jacx is None:
        f_jacx = jacobiano(f, 0)
    if f_jacw is None:
        f_jacw = jacobiano(f, 2)
    if h_jacx is None:
        h_jacx = jacobiano(h, 0)

    # con Huber
    def huber(residuo, delta):
        norm_residuo = np.linalg.norm(residuo)
        if norm_residuo <= delta:
            return residuo
        else:
            return delta*residuo/norm_residuo
    
    # Predicción
    x_pred = f(x, u, w) # estado predicho, xhat(k | k-1) 
    F = f_jacx(x, u, w) # jacobiana de f respecto a xkat(k-1 | k-1)
    G = np.array(f_jacw(x, u, w)) # jacobiana de f respecto a w (k-1 | k-1)
    P_pred = cs.mtimes([F,P,F.T]) + cs.mtimes([G,Q,G.T]) # Covarianza de estimación predicha, P(k | k-1)
    
    # Actualización
    y_tilde = y - h(x_pred) # residuo de la medición
    H = h_jacx(x_pred) # jacobiana de h respecto a xhat(k | k-1)
    S = cs.mtimes([H,P_pred,H.T])+R # covarianza de innovación (o residual)
    K = cs.mtimes([P_pred,H.T,linalg.inv(S)]) # ganancia de Kalman
    
    # con humbral
    # sigma_t = 2*np.sqrt(np.trace(S)) # traza de la covarianza de innovación
    # if np.linalg.norm(y_tilde) <= sigma_t:
    # # Normal: actualizar como siempre
    #     x_upd = x_pred + cs.mtimes(K,y_tilde) # estado actualizado, xhat(k | k)
    #     P_upd = cs.mtimes((cs.DM.eye(P_pred.shape[0])-cs.mtimes(K,H)),P_pred) # Covarianza de estimación actualizada, P(k | k)
    # else:
    #     # Outlier detectado: o ignorás la medición, o modificás la actualización
    #     x_upd = x_pred
    #     P_upd = P_pred

    if rho_huber:
        y_tilde = huber(y_tilde, rho_huber)

    # forma tradicional
    x_upd = x_pred + cs.mtimes(K, y_tilde) # estado actualizado, xhat(k | k)
    P_upd = cs.mtimes((cs.DM.eye(P_pred.shape[0])-cs.mtimes(K,H)), P_pred) # Covarianza de estimación actualizada, P(k | k)
    
    return x_upd, P_upd

def jacobiano(func, indep, dep=0, name=None):
    """Función que devuelve la función jacobiana de una función de Casadi, donde:
        func es la función de Casadi
        indep es el índice de la variable independiente
        dep es el índice de la variable dependiente (por defecto es 0)
        name es el nombre de la función jacobiana (por defecto es None)
       Y devuelve:
        la función jacobiana de la función de Casadi
    
    func should be a casadi.Function object. indep and dep should be the index
    of the independent and dependent variables respectively. They can be
    (zero-based) integer indices, or names of variables as strings.
    """
    if name is None:
        name = "jac_" + func.name()
    jacname = ["jac"]
    for (i, arglist) in [(dep, func.name_out), (indep, func.name_in)]:
        if isinstance(i, int):
            i = arglist()[i]
        jacname.append(i)
    jacname = ":".join(jacname)
    return func.factory(name, func.name_in(), [jacname])

def resolver_mhe(N, f, h, x0, u, y, P, Q, R, f_jacx=None,f_jacw=None,h_jacx=None, 
        mhe_method = None, huber_rho = None, bounds = {}, w_pos = False, **kwargs):
    """ Función que implementa el estimador de máxima verosimilitud (MHE), donde:
         N es un diccionario con las dimensiones del sistema, donde:
            N['x'] es la cantidad de estados
            N['y'] es la cantidad de mediciones
            N['w'] es la cantidad de ruido de proceso (si no se especifica, es igual a N['x'])
            N['v'] es la cantidad de ruido de medición (siempre es igual a N['y'])
            N['t'] es el tamaño de ventana
        f es la función de proceso
        h es la función de medición
        x0 es el estado inicial
        u es la entrada al sistema
        y son las mediciones
        P es la matriz de covarianza del estado inicial
        Q es la matriz de covarianza del ruido de proceso
        R es la matriz de covarianza del ruido de medición
        f_jacx es la jacobiana de f respecto a x
        f_jacw es la jacobiana de f respecto a w
        h_jacx es la jacobiana de h respecto a x
        (si no se especifican, se calculan automáticamente)
        mhe_method es el método de actualización de matriz P, donde:
            'None': ningún método
            'KF': filtro de Kalman
            'AD-CF': ganancia adaptativa con factor de olvido constante
            'AD-CT': ganancia adaptativa con traza constante
            'AD-VF': ganancia adaptativa con factor de olvido variable
        huber_rho es el parámetro de Huber (si no se especifica, se considera None)
        bounds es un diccionario con las cotas de cada variable, donde:
            bounds['lbx'] y bounds['ubx'] son las cotas inferior y superior de las variable x
            bounds['lbv'] y bounds['ubv'] son las cotas inferior y superior de las variable v
            bounds['lbw'] y bounds['ubw'] son las cotas inferior y superior de las variable w
            (si no se especifican, se consideran desde -∞ hasta ∞)
        w_pos es la desviación estándar del ruido de proceso (si es positivo)
        **kwargs son argumentos adicionales para la función mhe, para actualizar la matriz P
       Y devuelve:
        x_mhe los estados estimados
        w_mhe los ruidos de proceso estimados
        v_mhe los ruidos de medición estimados
        t_mhe el tiempo de ejecución por ventana
        trace_Pinv la traza de la inversa de la matriz P
    """
    # Obtengo estructuras de variables y parámetros
    opt_var, opt_par = variables_y_parametros(N)
    # Obtengo función de costo y restricciones
    J, con, lbg, ubg, lbx, ubx =costo_y_restricciones(N, opt_var, opt_par, f, h, Q, R, 
                                                      huber_rho, bounds, w_pos)
    # Obtengo variables estimadas
    x_mhe, w_mhe, v_mhe, t_mhe, trace_Pinv = mhe(N, f, h, x0, u, y, P, Q, R, opt_par, opt_var,
                                                 J, con, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx, 
                                                 f_jacx=f_jacx, f_jacw=f_jacw, h_jacx=h_jacx, 
                                                 mhe_method=mhe_method, w_pos=w_pos, **kwargs)
    return x_mhe, w_mhe, v_mhe, t_mhe, trace_Pinv