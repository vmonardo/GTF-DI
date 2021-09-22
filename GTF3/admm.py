import numpy as np
import networkx as nx

from GroupProximalOperator import L1ProximalOperator, SCADProximalOperator, MCPProximalOperator
from Penalty import L1Penalty, SCADPenalty, MCPPenalty
from scipy import sparse
from scipy.linalg import inv

def admm(Y, X, gamma, rho1, rho2, Dk,  penalty_f, penalty_param, tol_abs=10**(-5), tol_rel=10**(-4), max_iter=1000, B_init = None, invF=None, invG = None):
    """
    solves min_{B} 1/2||Y-X*B||_F^2 + h(D^(k+1)*B'; gamma, penalty_param)
    augmented Lagrangian problem:
        L_rho(B, C, Z, U, V)    = 1/2||Y-X*B||_F^2 + h(Z)
                                + rho_1/2* ||D^(k+1)*C- Z + U||_F^2 - rho_1/2* ||U||_F^2
                                + rho_2/2* ||B - C' + V||_F^2 - rho_2/2 ||V||_F^2
    Y : observed signal on graph
    X : feature matrix used to observe B
    gamma : parameter for penalty term (lambda in paper)
    rho_1, rho_2 : Lagrangian multipliers
    Dk : kth order graph difference operator, Dk \in R^{r x d}
    penalty_f : L1, SCAD, MCP
    penalty_param : extra parameter needed for calculating SCAD and MCP proximal operators

    Z  = D^(k+1)*C
    C = B' 

    Forward model:
    Y = XB + noise
    B \in R^{d x n} : signal
    X \in R^{p x d} : feature matrix
    Y \in R^{p x n} : measurements

    n : number of nodes
    d : signal dimensions
    p : number of measurements

    Optimization parameters:
    C \in R^{n x d} : transpose of signal
    Z \in R^{r x d} : graph signal pairwise difference
    U \in R^{r x d} : first dual variable
    V \in R^{d x n} : second dual variable

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
    
    # first constraint
    C = B.copy()
    C = C.T
    U = B - C.T

    # second constraint
    DC = Dk.dot(C)
    Z = DC.copy()
    V = DC - Z

    # problem dimensions
    d = B.shape[0]  # dimension of unknown signal at each node
    n = B.shape[1]  # number of nodes
    # p = X.shape[0]  # number of observed features at each node (unnecessary for computation)
    m = Dk.shape[0] # number of edges

    # Calculate the initial objective function value
    obj = 0.5 * np.linalg.norm(Y - X.dot(B), 'fro') ** 2
    db_norms = np.linalg.norm(DC, axis=1)
    vfunc = np.vectorize(lambda x: pen_func(x, gamma))
    f_Z = vfunc(db_norms)
    obj += gamma * sum(f_Z)

    # This will contain obj, r_norm{1,2}, eps_pri{1,2}, s_norm{1,2}, eps_dual{1,2}
    err_path = [[],[],[],[],[],[],[],[],[]]
    err_path[0].append(obj)

    ## pre-calculations performed out of loop for speed up
    if invF is None:
        # For updating B
        XTX = X.T.dot(X)
        invF = inv(rho1 * np.eye(d) + XTX)

    if invG is None:
        # For updating C
        DTD = Dk.T.dot(Dk)
        invG = inv(rho1 * np.eye(n) + rho2 * DTD.toarray())

    # pre-calculate 
    W = X.T.dot(Y)

    while not conv:
        ########################################
        ## Update B (ground truth signal)
        ## B = (rho1*I+ X'*X)^(-1)* (X'*Y + rho1(C' - U))
        ########################################

        CT = C.T
        B = np.matmul(invF, W + rho1*(CT - U))

        ########################################
        ## Update C (ground truth signal, transposed)
        ## C = (rho1*I+ Dk'*Dk)^(-1)* (rho1*(B + U)' + rho2*D'*(V - Z))
        ########################################

        BpUT = (B + U).T
        DkTVZ = Dk.T.dot(V - Z)
        
        C_prev = C.copy()
        C = np.matmul(invG, rho1*BpUT + rho2*DkTVZ)

        ########################################
        ## Update Z
        ## z = prox([Dk*B]_l + u_l), param = gamma/rho
        ########################################

        Z_prev = Z.copy()
        Z = prox.threshold(DC + V, gamma / rho2)

        ########################################
        ## Update U (scaled lagrangian variable)
        ## U = U + B - C'
        ########################################

        U += B - C.T

        ########################################
        ## Update V (scaled lagrangian variable)
        ## V = V + Dk*C - Z
        ########################################
        DC = Dk.dot(C)
        V += DC - Z

        ## Check the stopping criteria
        eps_pri1 = np.sqrt(m) * tol_abs + tol_rel * max(np.linalg.norm(B, 'fro'), np.linalg.norm(C, 'fro'))
        eps_dual1 = np.sqrt(n) * tol_abs + tol_rel * np.linalg.norm(rho1 * U, 'fro')
        r1 = np.linalg.norm(B - C.T, 'fro')
        s1 = np.linalg.norm(rho1 * C - C_prev, 'fro')

        eps_pri2 = np.sqrt(m) * tol_abs + tol_rel * max(np.linalg.norm(DC, 'fro'), np.linalg.norm(Z, 'fro'))
        eps_dual2 = np.sqrt(n) * tol_abs + tol_rel * np.linalg.norm(rho2 * Dk.T.dot(V), 'fro')
        r2 = np.linalg.norm(DC - Z, 'fro')
        s2 = np.linalg.norm(rho2 * Dk.T.dot(Z - Z_prev), 'fro')

        if r1 < eps_pri1 and s1 < eps_dual1 and r2 < eps_pri2 and s2 < eps_dual2:
            conv = 1

        err_path[1].append(r1)
        err_path[2].append(eps_pri1)
        err_path[3].append(s1)
        err_path[4].append(eps_dual1)

        err_path[5].append(r2)
        err_path[6].append(eps_pri2)
        err_path[7].append(s2)
        err_path[8].append(eps_dual2)

        ## Calculate the objective function 1/2||y-Psi*beta||_2^2 + gamma* \sum f(D^(k+1)*beta))
        obj = 0.5 * np.linalg.norm(Y - X.dot(B), 'fro') ** 2
        db_norms = np.linalg.norm(DC, axis=1)
        vfunc = np.vectorize(lambda x: pen_func(x, gamma))
        f_Z = vfunc(db_norms)
        obj += gamma * sum(f_Z)
        
        err_path[0].append(obj)

        iter_num += 1
        if iter_num > max_iter:
            break

    return B, obj, err_path







def showPlots(rho, gamma, penalty_f, err_path):
    # Just to declutter the test functions
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(err_path[0])
    plt.xlabel('iterations')
    plt.ylabel(r'$\frac{1}{2}||y-\beta||_2^2 + \gamma\sum (W_{ij}f(\beta_i-\beta_j)$')
    plt.title(penalty_f+'. rho: '+str(rho)+' gamma: '+str(gamma))
    #plt.show()

    plt.figure()
    plt.subplot(211)
    plt.plot(err_path[1], 'k', label='r norm')
    plt.plot(err_path[2], 'k--', label='eps_pri')
    plt.title(penalty_f+'. rho: '+str(rho)+' gamma: '+str(gamma))
    plt.ylabel(r'$||r||_2$')
    plt.legend()

    plt.subplot(212)
    plt.plot(err_path[3], 'k', label='s norm')
    plt.plot(err_path[4], 'k--', label='eps_dual')
    plt.ylabel(r'$||s||_2$')
    plt.xlabel('iterations')
    plt.legend()
    #plt.show()

if __name__ == "__main__":
    from Utilities import create2DSignal, penalty_matrix
    ""
    PENALTIES = ['L1'] 
    INPUT_SNR = 0                         
    k = 0
    ""

    ""
    name = '2d-grid'
    n1 = 4
    d = 3
    p = 10
    Y_HIGH = 10
    Y_LOW = -5

    Gnx, signal_2d, b_true, xs, ys = create2DSignal(k, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
    B = np.tile(b_true, (d, 1))
    n = nx.number_of_nodes(Gnx)

    sigma_sq = 0.0

    print ('INPUT_SNR:', INPUT_SNR)
    print ('SIGMA_SQ:', sigma_sq)
    
    Dk = penalty_matrix(Gnx, k)
    DTD = Dk.T.dot(Dk).toarray()
    [S, V] = np.linalg.eigh(DTD)

    X = np.random.normal(scale=np.sqrt(1/p), size=(p, d))
    print(X)

    # Observed vector-valued graph signal Y
    y_true = X.dot(B)
    Y = y_true + np.random.normal(scale=np.sqrt(sigma_sq), size=(p, n))

    print(Y)
    print(b_true)
    B_hat, obj_val, err = admm(Y=Y, X=X, gamma=.0001, rho1=0.01, rho2=0.00001, Dk=Dk, penalty_f='L1', penalty_param=.000000001)
    print(B_hat)