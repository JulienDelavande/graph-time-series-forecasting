from scipy.linalg import expm
import numpy as np

def ar_diffusion_loss(params, X, p, tau, lambda_reg=0.0, l1=True):
    """
    Computes the loss for an AR(p) model with an optional diffusion term.

    This function implements the following model:
      X_pred(t) = sum_{k=1..p} alpha_k * X(t - k) + beta * (expm(-tau * L) @ X(t - 1)),
    where L is a matrix constrained to be a Laplacian (rows sum to zero), and the 
    exponential matrix function is used for the diffusion term.

    The loss is the square root of the sum of squared prediction errors 
    over all timesteps from p to K - 1:
      sqrt( sum_t || X(t) - X_pred(t) ||^2 ),
    plus an optional L1 (lasso-like) or L2 (ridge-like) penalty on the 
    flattened off-diagonal elements of L.

    Parameters
    ----------
    params : array-like of shape (p + 1 + N*N,)
        Model parameters in the following order:
        [alpha_1, alpha_2, ..., alpha_p, beta, L_flat],
        where L_flat is an N*N flattened Laplacian matrix.
        - alpha_k : float
            The AR coefficients for the past k steps.
        - beta : float
            The diffusion coefficient.
        - L_flat : array of shape (N*N,)
            Flattened N x N Laplacian matrix.

    X : ndarray of shape (N, K)
        The observed time series data:
        - N : number of nodes (dimensions).
        - K : number of time steps.

    p : int
        The AR model order (i.e. how many past time steps to include).

    tau : float
        Diffusion parameter. If tau=0, the diffusion term is deactivated.

    lambda_reg : float, default=0.0
        Regularization coefficient. If zero, no regularization is applied.

    l1 : bool, default=True
        Indicates the type of regularization if `lambda_reg > 0`. 
        - True for L1 regularization (sum of absolute values).
        - False for L2 regularization (sum of squares).

    Returns
    -------
    float
        The computed loss, which is the square root of the sum of squared 
        errors plus any regularization penalty.

    Notes
    -----
    - The function imposes a Laplacian structure on L by setting the diagonal 
      to be the negative sum of the other elements in each row.
    - If tau > 0, the diffusion term uses matrix exponentiation via 
      `scipy.linalg.expm`.
    - The sum of squared errors (SSE) is computed from time index p up to K - 1.
    - Regularization is either L1 (lasso) or L2 (ridge) on the flattened 
      off-diagonal entries of the matrix L.
    - The output is np.sqrt(SSE) + regularization_penalty.
    """
    N, K = X.shape # N=number of nodes, K=number of time steps

    alpha = params[0:p]                         # (p,)
    beta  = params[p]                           # scalare
    L_flat = params[p+1 : p+1 + N*N]            # (N*N,)
    L = L_flat.reshape((N, N))

    # Impose a Laplacian constraint (sum of rows=0)
    np.fill_diagonal(L, -np.sum(L, axis=1))

    if tau == 0:
        expm_L = None
    else:
        expm_L = expm(-tau * L)

    total_loss = 0.0

    for t in range(p, K):
        # --- AR(p) (active if p>0) ---
        ar_part = np.zeros(N)
        for k in range(1, p+1):
            ar_part += alpha[k-1] * X[:, t-k]

        # --- Diffusion term (active if tau>0) ---
        if tau > 0:
            diff_part = beta * (expm_L @ X[:, t-1])
            X_pred = ar_part + diff_part
        else:
            X_pred = ar_part

        total_loss += np.linalg.norm(X[:, t] - X_pred)**2

    # Regularisation L1 or L2
    if lambda_reg > 0.0:
        if l1:
            reg = lambda_reg * np.sum(np.abs(L_flat))
        else: # L2
            reg = lambda_reg * np.sum(L_flat**2)
    else:
        reg = 0.0

    return np.sqrt(total_loss) + reg
