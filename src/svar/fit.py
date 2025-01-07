import numpy as np
from scipy.optimize import minimize
from .loss import svar_loss

def estimate_adjacency_matrix_simplified(X, M, lambda1=1e-3, lambda2=1e-3,
                                         tol=1e-5, max_iter=300, method='L-BFGS-B',
                                         lambda1_l1=True, lambda2_l1=True):
    """
    Estimates a single adjacency matrix A and the polynomial coefficients c
    under the simplified algorithm from the CGP model.

    Parameters
    ----------
    X : ndarray of shape (N, K)
        Time series data, where N is the number of nodes and K is the length
        of the time series.
    M : int
        Lag order for autoregression.
    lambda1 : float
        Regularization parameter for adjacency matrix A (L1 penalty).
    lambda2 : float
        Regularization parameter for coefficients c (L1 penalty).
    tol : float
        Convergence tolerance for the optimizer.
    max_iter : int
        Maximum number of iterations for the optimizer.
    method : str
        The optimization method to use. This is passed directly to `scipy.optimize.minimize`.
        Common choices include:
        - "L-BFGS-B" for quasi-Newton methods with box constraints.
        - "TNC" for truncated Newton methods.
        - "Nelder-Mead" for the simplex method.
    lambda1_l1 : bool
        Indicates the type of regularization for A if `lambda1 > 0`. 
        - True for L1 regularization (sum of absolute values).
        - False for L2 regularization (sum of squares).
    lambda2_l1 : bool
        Indicates the type of regularization for c if `lambda2 > 0`. 
        - True for L1 regularization (sum of absolute values).
        - False for L2 regularization (sum of squares).

    Returns
    -------
    A_hat : ndarray of shape (N, N)
        Estimated adjacency matrix.
    c_hat : ndarray of shape (M, M+1)
        Estimated polynomial coefficients.
    """
    N, K = X.shape

    # -- Flatten initial guesses --
    #   A_init  : Flattened adjacency matrix of shape N*N
    #   c_init  : Flattened coefficients of shape M*(M+1)
    A_init = (0.1 * np.random.rand(N, N)).ravel()
    c_init = (0.1 * np.random.rand(M, M + 1)).ravel()
    init_params = np.concatenate([A_init, c_init])
    
    def loss(params):
        return svar_loss(params, X, M, lambda1, lambda2, lambda1_l1, lambda2_l1)

    result = minimize(
        loss,
        init_params,
        method=method,
        options={'maxiter': max_iter, 'ftol': tol}
    )

    A_hat_flat, c_hat_flat = np.split(result.x, [N * N])
    A_hat = A_hat_flat.reshape((N, N))
    c_hat = c_hat_flat.reshape((M, M + 1))

    return A_hat, c_hat