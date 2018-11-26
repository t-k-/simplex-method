def simplex_tableau(names, c, A_bar, b_bar):
    # get dimensions
    m, n = A_bar.shape
    # check if it has redundant rows
    if m != matrix_rank(A_bar):
        print('not full rank matrix')
        quit()
    # initialize xi and z_0
    xi = -c
    z_0 = np.matrix([0.0])
    print_tableau(names, xi, z_0, A_bar, b_bar)
    # start iterations
    for iteration in range(99):
        # update objective row (zero those on eye matrix)
        A_bar_N = A_bar[:,m:n]
        xi_B = xi[0:m]
        xi_N = xi[m:n]
        xi_N_update = xi_N.T - xi_B.T * A_bar_N
        xi = horz_cat([pad_zeros(m), xi_N_update]).T
        # update objective row (update z_0)
        z_0 -= xi_B.T * b_bar
        # new iteration ready ...
        print('iteration', iteration)
        print_tableau(names, xi, z_0, A_bar, b_bar)
        if (xi <= 0).all() and (b_bar >= 0).all():
            print('optimal!')
            break
        # get entering position
        if (xi > 0).any(): # find k where xi_k first > 0
            k = (xi > 0).argmax() # Bland's rule
        else: # just choose the first in this case
            k = m + 0
        print('k = %u (entering)' % k)
        A_bar_k = A_bar[:, k]
        # lower bounded?
        if (A_bar_k <= 0).all() and (b_bar >= 0).all():
            print('no lower bound!')
            quit()
        # get exiting position
        np.seterr(divide='ignore') # ignore divided-by-zero error
        tmp = b_bar / A_bar_k
        if (b_bar >= 0).all():
            unselect = (A_bar_k <= 0)
            tmp[unselect] = float("inf")
        r = min_idx(tmp) # Bland's rule
        theta = amin(tmp)
        print('r = %u (exiting), theta = %f' % (r, theta))
        print('pivoting...')
        # swap k, r columns in A_bar, xi and names 
        A_bar[:, [k,r]] = A_bar[:, [r,k]]
        xi[[k,r]] = xi[[r,k]]
        names[[k,r]] = names[[r,k]]
        # update A_bar and b_bar using row operations
        B = A_bar[:, 0:m]
        A_bar = inv(B) * A_bar
        b_bar = inv(B) * b_bar
        print_tableau(names, xi, z_0, A_bar, b_bar)
