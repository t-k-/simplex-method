import numpy as np

from numpy.linalg import inv
from numpy.linalg import matrix_rank
from numpy import hstack as horz_cat
from numpy import vstack as vert_cat
from numpy import argmax as max_idx
from numpy import argmin as min_idx
from numpy import amin as amin
from numpy import zeros

# set numpy print function to print fraction instead of decimal
import fractions
np.set_printoptions(formatter={'all':
    lambda x: str(fractions.Fraction(x).limit_denominator()).ljust(6)
})

# example input #1
c     = np.matrix([ 0, 0, 0,  -1,  2]).T
A_bar = np.matrix([[1, 0, 0,  -2,  1],
                   [0, 1, 0,   1, -3],
                   [0, 0, 1,   1, -1]])
b_bar = np.matrix([ 2, 1, 2]).T
#b_bar = np.matrix([ 0, 0, 0]).T # degeneracy

# example input #2
# c     = np.matrix([ 0, 0,-2, 0,-8,  1,  1]).T
# A_bar = np.matrix([[1, 0, 1,-1, 6, -1,  0],
#                    [0, 1, 1, 1, 2,  0, -1]])
# b_bar = np.matrix([ 2, 1]).T

m, n = A_bar.shape

# zeros of length l
def pad_zeros(l):
    return np.matrix(zeros(l))

# get the index of first zero element
def first_zero(v):
    return (v == 0).argmax()

# print Simplex Tableau
def print_tableau(c, z, A, x):
    m, n = A.shape
    print(c.T, z)
    for row in range(m):
        print(A[row, 0:n], x[row])

# check if it has redundant rows
if m != matrix_rank(A_bar):
    print('not full rank matrix')
    quit()

# begin iterations
for iteration in range(999):
    print('iteration', iteration)
    # test BFS
    x_bar = horz_cat([b_bar.T, pad_zeros(n - m)]).T
    bfs = (x_bar >= 0).all() # basic feasible solution
    if not bfs:
        print('not BFS initially, try artificial vars!')
        quit()
    # update z_0 and xi
    c_B = c[0:m]
    z_0 = c.T * x_bar
    xi = (c_B.T * A_bar - c.T).T
    xi[0:m] = 0
    print_tableau(xi, z_0, A_bar, b_bar)
    # test optimality
    optmal = (xi <= 0).all()
    if optmal:
        print('optmal!')
        break
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
    if theta == 0:
        print('degeneracy, choose a random r...')
        r = np.random.randint(m)
    else:
        # move to the new solution
        x_bar = x_bar + theta * Delta
        r = first_zero(x_bar)
    print('pivot k = %u, r = %u.' % (k, r))
    # swap k, r rows in x_bar
    x_bar[[k,r]] = x_bar[[r,k]]
    # swap k, r columns in A_bar
    A_bar[:, [k,r]] = A_bar[:, [r,k]]
    # swap k, r rows in C_T
    c[[k,r]] = c[[r,k]]
    # update x_bar and A_bar
    b_bar = x_bar[0:m]
    B = A_bar[:, 0:m]
    A_bar = inv(B) * A_bar
