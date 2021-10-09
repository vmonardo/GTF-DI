import networkx as nx
from matplotlib import pyplot as plt
import numpy as np

from GTF3.Utilities import create2DSignal, create2DPath, penalty_matrix
from GTF3.admm import admm

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
n1 = 10
d = 20
p = 20
Y_HIGH = 10
Y_LOW = -10

# Gnx, signal_2d, b_true, xs, ys = create2DSignal(k, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)
Gnx, signal_2d, b_true, xs = create2DPath(k, n1, Y_HIGH=Y_HIGH, Y_LOW=Y_LOW)

n = nx.number_of_nodes(Gnx)
print(n)
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