import numpy as np

from numpy.linalg import inv
from numpy.linalg import matrix_rank
from numpy import hstack as horz_cat
from numpy import vstack as vert_cat
from numpy import argmax as max_idx
from numpy import asscalar as scalar
from numpy import zeros

def pad_zeros(l):
    return np.matrix(zeros(l))

# c = np.matrix( [6.8, 2.7,  0,  0]).T
# A = np.matrix([[668, 883, -1,  0],
#                [1.5, 1.2,  0, -1]])

c = np.matrix([0,   6.8, 2.7, 0 ]).T
A = np.matrix([[-1, 668, 883, 0 ],
               [0,  1.5, 1.2, -1]])

b = np.matrix([16800, 525]).T
print('c = ', c)
print('A = ', A)
print('b = ', b)
print()

m = matrix_rank(A)
m, n = A.shape
B = A[:, 0:m]
N = A[:, m:n]

A_bar = inv(B) * A # first half is eyes matrix
b_bar = inv(B) * b

bfs = (b_bar >= 0).all() # basic feasible solution
if not bfs:
    print('not BFS, change basis and try again!')
    quit()

for iteration in range(2):
    x_bar = horz_cat([b_bar.T, pad_zeros(n - m)]).T
    # print(A * x_bar) # should be equal to b

    c_B = c[0:m]
    c_N = c[m:n]

    z_0 = c.T * x_bar
    print(z_0)

    # print(z_0 == c_B.T * b_bar)

    xi = c_B.T * A_bar - c.T

    # print(A_bar, end="\n\n")
    # print(A, end="\n\n")

    optmal = (xi <= 0).all()
    if optmal:
        print('optmal!')
        quit()
    else:
        k = max_idx(xi)
        A_bar_k = A_bar[:, k]
        lower_bound = (A_bar_k > 0).any()
        if not lower_bound:
            print('no lower bound!')
            quit()
        e_k = zeros((n, 1))
        e_k[k] = 1
        d = vert_cat([-A_bar_k, pad_zeros(n - m).T]) + e_k
        select = (A_bar_k > 0)
        theta = scalar(min(b_bar[select] / A_bar_k[select]))
        x_hat = x_bar + theta * d
    x_bar = x_hat
