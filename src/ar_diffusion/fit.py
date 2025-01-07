from scipy.optimize import minimize
import numpy as np
from .loss import ar_diffusion_loss

def fit_ar_diffusion(X, p=2, tau=0.1, lambda_reg=0.0, l1=True, method="L-BFGS-B", max_iter=300):
    """
    Fits the parameters of an AR(p) model with a diffusion term.

    This function optimizes the parameters of the following model:
      X_pred(t) = sum_{k=1..p} alpha_k * X(t - k) + beta * (expm(-tau * L) @ X(t - 1)),
    where:
    - `alpha` are the autoregressive coefficients.
    - `beta` is the diffusion coefficient.
    - `L` is an N x N Laplacian matrix, constrained such that rows sum to zero.

    The parameters are fitted by minimizing the loss function `ar_diffusion_loss`,
    which combines the prediction error and an optional regularization term 
    on the Laplacian matrix.

    Parameters
    ----------
    X : ndarray of shape (N, K)
        The observed time series data:
        - N : number of nodes (dimensions).
        - K : number of time steps.

    p : int, default=2
        The AR model order (i.e., the number of past time steps to include).

    tau : float, default=0.1
        Diffusion parameter. If tau=0, the diffusion term is deactivated.

    lambda_reg : float, default=0.0
        Regularization coefficient. If zero, no regularization is applied.

    l1 : bool, default=True
        Indicates the type of regularization if `lambda_reg > 0`. 
        - True for L1 regularization (sum of absolute values).
        - False for L2 regularization (sum of squares).

    method : str, default="L-BFGS-B"
        The optimization method to use. This is passed directly to `scipy.optimize.minimize`.
        Common choices include:
        - "L-BFGS-B" for quasi-Newton methods with box constraints.
        - "TNC" for truncated Newton methods.
        - "Nelder-Mead" for the simplex method.

    max_iter : int, default=300
        Maximum number of iterations for the optimizer.

    Returns
    -------
    alpha_opt : ndarray of shape (p,)
        The optimized AR coefficients.

    beta_opt : float
        The optimized diffusion coefficient.

    L_opt : ndarray of shape (N, N)
        The optimized Laplacian matrix with rows constrained to sum to zero.

    res : OptimizeResult
        The result of the optimization process. Contains information about 
        convergence, final loss value, and other details.

    Notes
    -----
    - The initial parameters are set as follows:
        - `alpha` is initialized to small positive values.
        - `beta` is initialized to a small positive value.
        - `L` is initialized to a small random matrix.
    - The optimization problem may be sensitive to the initial parameters 
      and the choice of optimization method.
    - The function imposes a Laplacian constraint on `L` after optimization 
      by setting the diagonal to be the negative sum of the other elements in each row.
    """
    N, K = X.shape

    # Number of parameters = p (alpha) + 1 (beta) + N*N (L)
    alpha_init = 0.1 * np.ones(p)
    beta_init  = 0.1
    L_init     = 0.01 * np.random.randn(N, N)
    init_params = np.concatenate([alpha_init, [beta_init], L_init.ravel()])

    def objective(params):
        return ar_diffusion_loss(params, X, p, tau, lambda_reg, l1)

    res = minimize(
        objective,
        init_params,
        method=method,
        options={'maxiter': max_iter}
    )

    best_params = res.x
    alpha_opt = best_params[:p]
    beta_opt  = best_params[p]
    L_flat_opt = best_params[p+1 : p+1 + N*N]
    L_opt = L_flat_opt.reshape((N, N))

    # Impose a Laplacian constraint (sum of rows=0)
    np.fill_diagonal(L_opt, -np.sum(L_opt, axis=1))

    return alpha_opt, beta_opt, L_opt, res