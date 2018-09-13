import numpy as np

from numpy.linalg import inv
from numpy.linalg import matrix_rank
from numpy import hstack as horz_cat
from numpy import vstack as vert_cat
from numpy import argmax as max_idx
from numpy import argmin as min_idx
from numpy import amin as amin
from numpy import zeros

def pad_zeros(l):
    return np.matrix(zeros(l))

def first_zero(v):
    return (v == 0).argmax()

def print_tableau(c, z, A, x):
    m, n = A.shape
    print(c.T, z)
    for row in range(m):
        print(A[row, 0:n], x[row])

c     = np.matrix([ 0, 0, 0,  -1,  2]).T
A_bar = np.matrix([[1, 0, 0,  -2,  1],
                   [0, 1, 0,   1, -3],
                   [0, 0, 1,   1, -1]])
b_bar = np.matrix([ 2, 1, 2]).T

m, n = A_bar.shape

if m != matrix_rank(A_bar):
    print('not full rank matrix')
    quit()

for iteration in range(999):
    print('iteration', iteration)
    # test BFS
    x_bar = horz_cat([b_bar.T, pad_zeros(n - m)]).T
    bfs = (x_bar >= 0).all() # basic feasible solution
    if not bfs:
        print('not BFS, change basis and try again!')
        quit()
    # update z_0 and xi
    c_B = c[0:m]
    z_0 = c.T * x_bar
    xi = c_B.T * A_bar - c.T
    print_tableau(xi.T, z_0, A_bar, b_bar)
    # test optimality
    optmal = (xi <= 0).all()
    if optmal:
        print('optmal!')
        quit()
    # calculate k and A_bar_k, test boundness
    k = max_idx(xi)
    A_bar_k = A_bar[:, k]
    lower_bound = (A_bar_k > 0).any()
    if not lower_bound:
        print('no lower bound!')
        quit()
    # calculate Delta and theta (step vector)
    e_k = zeros((n, 1))
    e_k[k] = 1
    Delta = vert_cat([-A_bar_k, pad_zeros(n - m).T]) + e_k
    select = (A_bar_k > 0)
    theta = amin(b_bar[select] / A_bar_k[select])
    # move to the new solution space
    x_hat = x_bar + theta * Delta
    r = first_zero(x_hat)
    print('pivot k, r = ', k, r)
    # swap k, r rows in x_hat
    x_hat[[k,r]] = x_hat[[r,k]]
    # swap k, r columns in A_bar
    A_bar[:, [k,r]] = A_bar[:, [r,k]]
    # swap k, r rows in C_T
    c[[k,r]] = c[[r,k]]
    # update x_bar and A_bar
    b_bar = x_hat[0:m]
    B = A_bar[:, 0:m]
    A_bar = inv(B) * A_bar
