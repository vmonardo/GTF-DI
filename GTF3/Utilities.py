from functools import *
import networkx as nx
import numpy as np
import random
import scipy as sp
import matplotlib.pyplot as plt


# generate piecewise-constant signal based on shortest path distances
def gen_pwc_sig(Gnx,num_seeds,path_lens):
    seeds = random.sample(range(0,Gnx.number_of_nodes()),num_seeds)
    seeds_sig_dict = dict(zip(seeds,range(num_seeds)))
    #path_lens = dict(nx.shortest_path_length(Gnx))
    get_nearest = lambda source: seeds[np.argmin(np.array([path_lens[source][target]
                                       for target in seeds]))]
    closest_seeds = np.array(list(map(get_nearest, range(Gnx.number_of_nodes()))))
    sig = [seeds_sig_dict[x] for x in closest_seeds]
    return np.array(sig)


# solve generalized Poisson eqn \Delta x = b ,sparsity of b = df
def poisson_sig(Gnx, k, df):
    if k % 2 == 0:
        r = Gnx.number_of_edges()
    elif k % 2 == 1:
        r = Gnx.number_of_nodes()
    eta = sp.sparse.random(r,1,density = float(df)/r)
    return np.squeeze(np.asarray(np.linalg.pinv(penalty_matrix(Gnx,k).todense()) * eta))


# get (weighted) incidence matrix
def incidence_matrix(Gnx):
    Delta = nx.incidence_matrix(Gnx, oriented=True, weight='weight').transpose()
    Delta_sign = sp.sparse.csr_matrix(np.sign(Delta.todense()))
    Delta_sqrt = np.sqrt(np.abs(Delta))
    return Delta_sign.multiply(Delta_sqrt)


# get generalized penalty matrix
def penalty_matrix(Gnx, k):
    if k < 0:
        raise ValueError("k must be non-negative")
    elif k == 0:
        return incidence_matrix(Gnx)
    elif k % 2 == 0:
        return incidence_matrix(Gnx) * penalty_matrix(Gnx,k-1)
    elif k % 2 == 1:
        return incidence_matrix(Gnx).transpose() * penalty_matrix(Gnx,k-1)


""
def create2DGraph(n=10, plot_flag=0):
    # Create lattice graph. Thanks networkx.
    G = nx.grid_2d_graph(n, n, periodic=False)
    if plot_flag:
        nx.draw_kamada_kawai(G)
        plt.title('Lattice Graph Visualization')
        plt.show()
    return G

def createLineGraph(n=10, plot_flag=0):
    G = nx.path_graph(n)
    if plot_flag:
        nx.draw_kamada_kawai(G)
        plt.title('Lattice Graph Visualization')
        plt.show()
    return G

def create2DPath(k=0, n=10, Y_HIGH=10, Y_LOW=-5):
    if k==0:
        signal_2d = np.ones((n,1))
        signal_2d[:n//4+1] = Y_HIGH
        signal_2d[3*n//4-1:] = Y_LOW
    else:
        print ("we dont have that yet!")
        return

    Gnx = createLineGraph(n, plot_flag=0)
    xs = []
    ys = []
    y_true = []

    for node in Gnx.nodes():
        x = node
        xs.append(x)

        y_true.append(signal_2d[x])

    y_true = np.array(y_true)

    return Gnx, signal_2d, y_true, xs


def create2DSignal(k=0, n=10, Y_HIGH=10, Y_LOW=-5):
    if k == 0:
        signal_2d = np.ones((n, n))
        signal_2d[:n//4+1, :n//4+1] = Y_HIGH
        signal_2d[3*n//4-1:, 3*n//4-1:] = Y_LOW

    else:
        print ("we dont have that yet!")
        return
    Gnx = create2DGraph(n, plot_flag=0)
    xs = []
    ys = []
    y_true = []

    for node in Gnx.nodes():
        x, y = node
        xs.append(x)
        ys.append(y)

        y_true.append(signal_2d[y, x])

    y_true = np.array(y_true)

    return Gnx, signal_2d, y_true, xs, ys



