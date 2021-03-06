import numpy as np

from numpy.linalg import inv
from numpy.linalg import matrix_rank
from numpy import hstack as horz_cat
from numpy import vstack as vert_cat
from numpy import argmax as max_idx
from numpy import argmin as min_idx
from numpy import amin as amin
from numpy import zeros

np.seterr(divide='ignore') # ignore divided-by-zero error

def print_float(x):
    return str("%8.3f" % x)

def print_frac(x):
    return str(fractions.Fraction(x).limit_denominator()).rjust(8)

# set numpy print function to print fraction instead of decimal
import fractions
np.set_printoptions(formatter={})
np.set_printoptions(formatter ={
    'float': print_float,
    #'float': print_frac,
    'str_kind': lambda x: str("%8s" % x)
})

# zeros of length l
def pad_zeros(l):
    return np.matrix(zeros(l))

# get the index of first zero element
def first_zero(v):
    return (v == 0).argmax()

# print Simplex Tableau
def print_tableau(names, xi, z, A, x):
    m, n = A.shape
    print(names)
    print(xi.T, z)
    for row in range(m):
        print(A[row, 0:n], x[row])

def simplex_tableau(names, c, A_bar, b_bar):
    # get dimensions
    m, n = A_bar.shape
    # check if it has redundant rows
    if m != matrix_rank(A_bar):
        print('not full rank matrix')
        quit()
    # check if initial b_bar is BFS
    if (b_bar < 0).any():
        print('initial b_bar not all >= 0')
        quit()
    # initialize xi and z_0
    xi = -c
    z_0 = np.matrix([0.0])
    print_tableau(names, xi, z_0, A_bar, b_bar)
    # start iterations
    for iteration in range(99):
        # update objective row (zero those on eye matrix)
        A_bar_N = A_bar[:,m:n]
        xi_B, xi_N = xi[0:m], xi[m:n]
        xi_N_update = xi_N.T - xi_B.T * A_bar_N
        xi = horz_cat([pad_zeros(m), xi_N_update]).T
        # update objective row (update z_0)
        z_0 -= xi_B.T * b_bar
        # new iteration ready ...
        print('\n', '## Iteration', iteration)
        print_tableau(names, xi, z_0, A_bar, b_bar)
        if (xi <= 0).all():
            print('optimal!')
            break
        # get entering position
        k = (xi > 0).argmax() # Bland's rule
        print('k = %u (entering)' % k, end=", ")
        A_bar_k = A_bar[:, k]
        # lower bounded?
        if (A_bar_k <= 0).all():
            print('no lower bound!')
            quit()
        # get exiting position
        tmp = b_bar / A_bar_k
        tmp[A_bar_k <= 0] = float("inf")
        r = min_idx(tmp) # Bland's rule
        print('r = %u (exiting), theta = %f' % (r, amin(tmp)))
        # swap k, r columns in A_bar, xi and names 
        A_bar[:, [k,r]] = A_bar[:, [r,k]]
        xi[[k,r]] = xi[[r,k]]
        names[[k,r]] = names[[r,k]]
        # update A_bar and b_bar using row operations
        B = A_bar[:, 0:m] # must be linear independent, i.e. basis
        A_bar, b_bar = inv(B) * A_bar, inv(B) * b_bar
        print_tableau(names, xi, z_0, A_bar, b_bar)

# theory-style calculation
def simplex_theory(names, c, A_bar, b_bar):
    # get dimensions
    m, n = A_bar.shape
    # check if it has redundant rows
    if m != matrix_rank(A_bar):
        print('not full rank matrix')
        quit()
    # check if initial b_bar is BFS
    if (b_bar < 0).any():
        print('initial b_bar not all >= 0')
        quit()
    # initialize xi and z_0
    xi = -c
    z_0 = np.matrix([0.0])
    print_tableau(names, xi, z_0, A_bar, b_bar)
    # start iterations
    for iteration in range(99):
        print('## Iteration', iteration)
        # update z_0 and xi
        x_bar = horz_cat([b_bar.T, pad_zeros(n - m)]).T
        c_B = c[0:m]
        z_0 = c.T * x_bar
        xi = (c_B.T * A_bar - c.T).T
        xi[0:m] = 0
        assert(np.allclose(xi.T * x_bar, np.matrix([0])))
        print_tableau(names, xi, z_0, A_bar, b_bar)
        # test optimality
        if (xi <= 0).all() and (b_bar >= 0).all():
            print('optimal!')
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
        print('pivot k = %u, r = %u, theta = %s' % (k, r,
            str(fractions.Fraction(theta).limit_denominator())))
        # swap k, r rows in x_bar
        x_bar[[k,r]] = x_bar[[r,k]]
        # swap k, r columns in A_bar
        A_bar[:, [k,r]] = A_bar[:, [r,k]]
        # swap k, r rows in C_T
        c[[k,r]] = c[[r,k]]
        names[[k,r]] = names[[r,k]]
        # update x_bar and A_bar
        B = A_bar[:, 0:m]
        b_bar = x_bar[0:m]
        A_bar = inv(B) * A_bar

def main():
    # example input #1
    # names = np.array(['x' + str(i) for i in range(1, 6)])
    # c     = np.matrix([ 0.0, 0.0, 0.0,  -1.0,  2.0]).T
    # A_bar = np.matrix([[1.0, 0.0, 0.0,  -2.0,  1.0],
    #                    [0.0, 1.0, 0.0,   1.0, -3.0],
    #                    [0.0, 0.0, 1.0,   1.0, -1.0]])
    # b_bar = np.matrix([ 2.0, 1.0, 2.0]).T

    # example input #2
    # names = np.array(['x' + str(i) for i in range(1, 8)])
    # c     = np.matrix([ 0.0, 0.0, -2.0,  0.0, -8.0,  1.0,  1.0]).T
    # A_bar = np.matrix([[1.0, 0.0,  1.0, -1.0,  6.0, -1.0,  0.0],
    #                    [0.0, 1.0,  1.0,  1.0,  2.0,  0.0, -1.0]])
    # b_bar = np.matrix([ 2.0, 1.0]).T

    # example input #3
    # http://zhangxiaoyang.me/categories/intro-to-algorithms-tutorial/intro-to-algorithms-tutorial-6-1.html
    names = np.array( ['a1', 'a2',  'x1',   'x2', 's1', 's2'])
    M = 99999.0
    c     = np.matrix([   M,   M,    6.8,    2.7,  0.0,  0.0]).T
    A_bar = np.matrix([[1.0, 0.0,  668.0,  883.0, -1.0,  0.0],
                       [0.0, 1.0,    1.5,    1.2,  0.0, -1.0]])
    b_bar = np.matrix([ 16800.0, 525.0]).T

    # example input #4
    # https://www.hrwhisper.me/introduction-to-simplex-algorithm/
    # names = np.array(['x' + str(i) for i in [4, 5, 6, 7, 1, 2, 3, 4]])
    # c     = np.matrix([ 0.0, 0.0, 0.0, 0.0, -1.0, -14.0, -6.0, 0.0]).T
    # A_bar = np.matrix([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    #                    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    #                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    #                    [0.0, 0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0]])
    # b_bar = np.matrix([4.0, 2.0, 3.0, 6.0]).T

    # example input #5 (no lower bound)
    # https://www.hrwhisper.me/introduction-to-simplex-algorithm/
    # names = np.array( ['x3', 'x4', 'x1', 'x2'])
    # c     = np.matrix([ 0.0, 0.0, -1.0,  -1.0]).T
    # A_bar = np.matrix([[1.0, 0.0,  1.0,  -1.0],
    #                    [0.0, 1.0, -1.0,   1.0]])
    # b_bar = np.matrix([ 1.0, 1.0]).T

    simplex_tableau(names, c, A_bar, b_bar)
    #simplex_theory(names, c, A_bar, b_bar)

if __name__ == '__main__':
    main()
