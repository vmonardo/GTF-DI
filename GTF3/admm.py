import numpy as np
import networkx as nx

from .GroupProximalOperator import L1ProximalOperator, SCADProximalOperator, MCPProximalOperator
from .Penalty import L1Penalty, SCADPenalty, MCPPenalty
from scipy import sparse
from scipy.linalg import inv
from scipy.linalg import solve_sylvester

def admm(Y, X, gamma, rho, Dk, penalty_f, penalty_param, tol_abs=10**(-5), tol_rel=10**(-4), max_iter=1000, B_init = None):
    """
    solves min_{B} 1/2||Y-X*B||_F^2 + h(D^(k+1)*B'; gamma, penalty_param)
    augmented Lagrangian problem:
        L_rho(B, Z, U)    = 1/2||Y-X*B||_F^2 + h(Z)
                                + rho/2* ||D^(k+1)*B^T- Z + U||_F^2 - rho/2* ||U||_F^2
    Y : observed signal on graph
    X : feature matrix used to observe B
    gamma : parameter for penalty term (lambda in paper)
    rho_1: Lagrangian multiplier
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
    err_path = [[],[],[],[],[],[],[],[],[]]
    err_path[0].append(obj)

    ## pre-calculations performed out of loop for speed up
    XTX = X.T.dot(X)
    DTD = Dk.T.dot(Dk)

    # pre-calculate 
    XTY = X.T.dot(Y)

    while not conv:
        ########################################
        ## Update B (ground truth signal)
        ## B = (rho1*I+ X'*X)^(-1)* (X'*Y + rho1(C' - U))
        ########################################

        Qtilde = XTY + rho * (Z.T.dot(Dk) - U.T.dot(Dk))
        B = solve_sylvester(XTX, DTD, Qtilde) 

        ########################################
        ## Update Z
        ## z = prox([Dk*B]_l + u_l), param = gamma/rho
        ########################################

        Z_prev = Z.copy()
        DBT = Dk.dot(B.T)
        Z = prox.threshold(DBT + U, gamma / rho)

        ########################################
        ## Update U (scaled lagrangian variable)
        ## U = U + B - C'
        ########################################

        U += DBT - Z

        ## Check the stopping criteria
        eps_pri = np.sqrt(m) * tol_abs + tol_rel * max(np.linalg.norm(DBT, 'fro'), np.linalg.norm(Z, 'fro'))
        eps_dual = np.sqrt(n) * tol_abs + tol_rel * np.linalg.norm(rho * Dk.T.dot(U), 'fro')
        r = np.linalg.norm(DBT - Z, 'fro')
        s = np.linalg.norm(rho * Dk.T.dot(Z - Z_prev), 'fro')

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

    return B, obj, err_path


if __name__ == "__main__":
    from Utilities import create2DSignal, penalty_matrix
    ""
    PENALTIES = ['L1'] 
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
    name = '2d-grid'
    n1 = 2
    d = 100
    p = 80
    Y_HIGH = 10
    Y_LOW = -10

    Gnx, signal_2d, b_true, xs, ys = create2DSignal(k, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
    
    n = nx.number_of_nodes(Gnx)
    B = np.zeros((d, n))

    class1 = Y_HIGH*np.random.normal(size=(d,1)) * np.random.choice([0, 1], size=(d,1), p=[.75, .25])
    class2 = Y_LOW*np.random.normal(size=(d,1)) * np.random.choice([0, 1], size=(d,1), p=[.75, .25])
    class3 = np.random.normal(size=(d,1)) * np.random.choice([0, 1], size=(d,1), p=[.75, .25])

    for i in range(n):
        if b_true[i] == Y_HIGH:
            B[:, i] = class1.ravel()
        if b_true[i] == Y_LOW:
            B[:, i] = class2.ravel()
        if b_true[i] == 1:
            B[:, i] = class3.ravel()

    print(np.count_nonzero(B) / (d*n))
    sigma_sq = 0

    Dk = penalty_matrix(Gnx, k)
    DTD = Dk.T.dot(Dk).toarray()
    [S, V] = np.linalg.eigh(DTD)

    # X = np.eye(p)
    X = np.random.normal(scale=np.sqrt(1/p), size=(p, d))
    # print(X)

    # Observed vector-valued graph signal Y
    y_true = X.dot(B)
    Y = y_true + np.random.normal(scale=np.sqrt(sigma_sq), size=(p, n))

    # print(Y)
    print('True B: ', np.around(B, decimals=3))
    B_hat, obj_val, err = admm(Y=Y, X=X, gamma=.1, c1=.00, c2=.01, rho1=0.01, rho2=0.01, Dk=Dk, penalty_f='L1', penalty_param=3)
    print('Estimate B: ', np.around(B_hat, decimals=3))
    print('NNMSE: ', np.linalg.norm(B_hat - B)**2 / np.linalg.norm(B)**2)