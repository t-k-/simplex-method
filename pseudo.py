def simplex_tableau(names, c, A_bar, b_bar):
    m, n = A_bar.shape
    # check inputs
    if m != matrix_rank(A_bar):
        raise Exception("not full rank matrix")
    if (b_bar < 0).any():
        raise Exception("initial b_bar not all >= 0")
    # initialize xi and z_0
    xi = -c
    z_0 = np.matrix([0.0])
    for iteration in range(99):
        # update objective row (zero those on eye matrix)
        A_bar_N = A_bar[:,m:n]
        xi_B, xi_N = xi[0:m], xi[m:n]
        xi_N_update = xi_N.T - xi_B.T * A_bar_N
        xi = horz_cat([pad_zeros(m), xi_N_update]).T
        # update objective row (update z_0)
        z_0 -= xi_B.T * b_bar
        # new iteration ready ...
        if (xi <= 0).all():
            return # optimal
        # get entering position
        k = (xi > 0).argmax() # Bland's rule
        A_bar_k = A_bar[:, k]
        # lower bounded?
        if (A_bar_k <= 0).all():
            raise Exception("no lower bound")
        # get exiting position
        tmp = b_bar / A_bar_k
        tmp[A_bar_k <= 0] = float("inf")
        r = min_idx(tmp) # Bland's rule
        # swap k, r columns in A_bar, xi and names 
        A_bar[:, [k,r]] = A_bar[:, [r,k]]
        xi[[k,r]] = xi[[r,k]]
        names[[k,r]] = names[[r,k]]
        # update A_bar and b_bar using row operations
        B = A_bar[:, 0:m] # must be linear independent, i.e. basis
        A_bar, b_bar = inv(B) * A_bar, inv(B) * b_bar
