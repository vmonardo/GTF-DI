from ctypes import DllCanUnloadNow
import numpy as np
import networkx as nx

from GroupProximalOperator import L1ProximalOperator, SCADProximalOperator, MCPProximalOperator
from Penalty import L1Penalty, SCADPenalty, MCPPenalty
from scipy import sparse
from scipy.linalg import inv
from scipy.linalg import solve_sylvester

def admm(gamma, tau, penalty_param, Y, X, Dk, penalty_f, tol_abs=10**(-5), tol_rel=10**(-4), max_iter=1000, B_init = None):
    """
    solves min_{B} 1/2||Y-X*B||_F^2 + h(D^(k+1)*B'; gamma, penalty_param) 
    augmented Lagrangian problem:
        L_rho(B, Z, U)  = 1/2||Y-X*B||_F^2 + h(Z; gamma, penalty_param) 
                        + tau/2* ||D^(k+1)*B^T- Z + U||_F^2 - tau/2* ||U||_F^2
    Y : observed signal on graph
    X : feature matrix used to observe B
    gamma : parameter for first penalty term 
    rho : parameter for second penalty term 
    tau: Lagrangian multiplier 
    Dk : kth order graph difference operator, Dk \in R^{r x d}
    penalty_f : L1, SCAD, MCP
    penalty_param : extra parameter needed for calculating SCAD and MCP proximal operators

    Z  = D^(k+1)*B^T

    Forward model:
    Y = XB + noise
    B \in R^{d x n} : signal
    X \in R^{p x d} : feature matrix
    Y \in R^{p x n} : measurements

    n : number of nodes
    d : signal dimensions
    p : number of measurements

    Optimization parameters:
    Z \in R^{r x d} : graph signal pairwise difference
    U \in R^{r x d} : dual variable

    """
    if penalty_f == "L1":
        prox = L1ProximalOperator()
        pen = L1Penalty()
        pen_func = pen.calculate
    elif "SCAD" in penalty_f:
        prox = SCADProximalOperator(penalty_param)
        pen = SCADPenalty(penalty_param)
        pen_func = pen.calculate
    elif "MCP" in penalty_f:
        prox = MCPProximalOperator(penalty_param)
        pen = MCPPenalty(penalty_param)
        pen_func = pen.calculate
    else:
        print ("This penalty is not supported yet.")
        raise Exception

    iter_num = 0
    conv = 0

    # # Initialize B, eta_tilde and u_tilde
    if B_init is None:
        B_init = X.T.dot(Y)
    B = B_init.copy()
    # print('B init: ', B)
 
    # problem dimensions
    d = B.shape[0]  # dimension of unknown signal at each node
    n = B.shape[1]  # number of nodes
    # p = X.shape[0]  # number of observed features at each node (unnecessary for computation)
    m = Dk.shape[0] # number of edges

    # first constraint
    DBT = Dk.dot(B.T)
    Z = DBT.copy()
    U = DBT - Z

    # Calculate the initial objective function value
    obj = 0.5 * np.linalg.norm(Y - X.dot(B), 'fro') ** 2
    db_norms = np.linalg.norm(DBT, axis=1)
    vfunc = np.vectorize(lambda x: pen_func(x, gamma))
    f_Z = vfunc(db_norms)
    obj += gamma * sum(f_Z)

    # This will contain obj, r_norm{1,2}, eps_pri{1,2}, s_norm{1,2}, eps_dual{1,2}
    err_path = [[],[],[],[],[]]
    err_path[0].append(obj)

    ## pre-calculations performed out of loop for speed up
    XTX = X.T.dot(X)
    DTD = Dk.T.dot(Dk).todense()
    XTY = X.T.dot(Y)

    while not conv:
        ########################################
        ## Update B (ground truth signal) with Sylvester eq
        # Solve Sylvester's equation AX + XB= Q, where 
        #     X = B 
        #     A = X^TX 
        #     B = D^TD 
        #     Q = X^T Y + \tau Z^\top D  - \tau U^T D
        ########################################

        Qtilde = XTY + tau * (Z.T.dot(Dk.todense()) - U.T.dot(Dk.todense()))
        B = solve_sylvester(XTX, tau*DTD, Qtilde) 
        B = prox.threshold(B, 5*gamma / tau)
        DBT = Dk.dot(B.T)

        ########################################
        ## Update Z
        ## z = prox([Dk*B]_l + u_l), param = gamma/tau
        ########################################

        Z_prev = Z.copy()
        Z = prox.threshold(DBT + U, gamma / tau)

        ########################################
        ## Update U (scaled lagrangian variable)
        ## U = U + B - C'
        ########################################

        U += DBT - Z

        ## Check the stopping criteria
        eps_pri = np.sqrt(m) * tol_abs + tol_rel * max(np.linalg.norm(DBT, 'fro'), np.linalg.norm(Z, 'fro'))
        eps_dual = np.sqrt(n) * tol_abs + tol_rel * np.linalg.norm(tau * Dk.T.dot(U), 'fro')
        r = np.linalg.norm(DBT - Z, 'fro')
        s = np.linalg.norm(tau * Dk.T.dot(Z - Z_prev), 'fro')

        if r < eps_pri and s < eps_dual:
            conv = 1

        err_path[1].append(r)
        err_path[2].append(eps_pri)
        err_path[3].append(s)
        err_path[4].append(eps_dual)


        ## Calculate the objective function 1/2||y-Psi*beta||_2^2 + gamma* \sum f(D^(k+1)*beta))
        obj = 0.5 * np.linalg.norm(Y - X.dot(B), 'fro') ** 2
        db_norms = np.linalg.norm(DBT, axis=1)
        vfunc = np.vectorize(lambda x: pen_func(x, gamma))
        f_Z = vfunc(db_norms)
        obj += gamma * sum(f_Z)
        
        err_path[0].append(obj)

        iter_num += 1
        if iter_num > max_iter:
            break

    return {
        'B': B,
        'obj': obj,
        'err_path': err_path
    }

def admm_proxy(args, Y, X, Dk, penalty_f, tol_abs=10**(-5), tol_rel=10**(-4), max_iter=1000, B_init = None):
    from hyperopt import STATUS_OK
    gamma, tau, penalty_param = args
    result = admm(  gamma=gamma,
                    tau=tau, 
                    penalty_param=penalty_param, 
                    Y=Y,
                    X=X,
                    Dk=Dk, 
                    penalty_f=penalty_f, 
                    tol_abs=tol_abs, 
                    tol_rel=tol_rel,
                    max_iter=max_iter, 
                    B_init=B_init)
    return {
        'loss': result['obj'],
        'status': STATUS_OK,
        'B': result['B'],
        'obj': result['obj'],
        'err_path': result['err_path']
    }

if __name__ == "__main__":
    from Utilities import create2DSignal, penalty_matrix, create2DPath
    import matplotlib.pyplot as plt
    ""
    INPUT_SNR = 0                         
    k = 0
    ""

    # Forward model:
    # Y = XB + noise
    # B \in R^{d x n} : signal
    # X \in R^{p x d} : feature matrix
    # Y \in R^{p x n} : measurements

    # n : number of nodes
    # d : signal dimensions
    # p : number of measurements

    ""
    n1 = 10
    d = 50
    p = 25
    Y_HIGH = 10
    Y_LOW = -5

    Gnx, signal_2d, b_true, xs, ys = create2DSignal(k, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
    # Gnx, signal_2d, b_true, xs = create2DPath(k, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
    
    n = nx.number_of_nodes(Gnx)
    B = np.zeros((d, n))

    class1 = Y_HIGH*np.random.choice([0, 1], size=(d,1), p=[.9, .1])
    class2 = Y_LOW*np.random.choice([0, 1], size=(d,1), p=[.9, .1])
    class3 = np.random.choice([0, 1], size=(d,1), p=[.9, .1])

    for i in range(n):
        if b_true[i] == Y_HIGH:
            B[:, i] = class1.ravel()
        if b_true[i] == Y_LOW:
            B[:, i] = class2.ravel()
        if b_true[i] == 1:
            B[:, i] = class3.ravel()

    print(np.count_nonzero(B) / (d*n))
    sigma_sq = 0.0

    Dk = penalty_matrix(Gnx, k)
    DTD = Dk.T.dot(Dk).toarray()
    [S, V] = np.linalg.eigh(DTD)

    # X = np.eye(d)
    X = np.random.normal(scale=1/np.sqrt(d), size=(p, d))
    # print(X)

    # Observed vector-valued graph signal Y
    y_true = X.dot(B)
    Y = y_true + np.random.normal(scale=np.sqrt(sigma_sq), size=(p, n))
    # Y = B

    gamma = 0.01
    tau = 0.01
    penalty_param = 3
    penalty_f = 'SCAD'

    # print(Y) 
    print('True B: ', np.around(B, decimals=3))
    output = admm(Y=Y, X=X, gamma=gamma, tau=tau, Dk=Dk, penalty_f=penalty_f, penalty_param=penalty_param, max_iter=200)
    B_hat, obj_val, err_path = output['B'], output['obj'], output['err']
    Bls = np.linalg.pinv(X).dot(Y)

    print('Estimate B: ', np.around(B_hat, decimals=1))
    print('NNMSE: ', np.linalg.norm(B_hat - B)**2 / np.linalg.norm(B)**2)

    plt.figure()
    plt.subplot(311)
    plt.imshow(B)
    plt.subplot(312)
    plt.imshow(B_hat)
    plt.subplot(313)
    plt.imshow(Bls)

    plt.figure()
    plt.plot(err_path[0])
    plt.xlabel('iterations')
    plt.ylabel(r'$\frac{1}{2}||y-\beta||_2^2 + \gamma\sum (W_{ij}f(\beta_i-\beta_j)$')
    plt.title(penalty_f+'. rho: '+str(tau)+' gamma: '+str(gamma))

    plt.figure()
    plt.subplot(211)
    plt.plot(err_path[1], 'k', label='r norm')
    plt.plot(err_path[2], 'k--', label='eps_pri')
    plt.title(penalty_f+'. rho: '+str(tau)+' gamma: '+str(gamma))
    plt.ylabel(r'$||r||_2$')
    plt.legend()

    plt.subplot(212)
    plt.plot(err_path[3], 'k', label='s norm')
    plt.plot(err_path[4], 'k--', label='eps_dual')
    plt.ylabel(r'$||s||_2$')
    plt.xlabel('iterations')
    plt.legend()
    plt.show()