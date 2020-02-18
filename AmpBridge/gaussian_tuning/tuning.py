def grid_search(lam_seq, X, y, q):
    """
    lam_seq has to be in descending order
    """
    k = lam_seq.shape[0]
    Beta_hat = Bridge(lam_seq, X, y, q)
    tau_seq = np.empty(k)
    for i in range(k):
        tau_seq[i] = tau(Beta_hat[:, i], X, y, q)
    return lam_seq[np.argmin(tau_seq)et filetype indent on

