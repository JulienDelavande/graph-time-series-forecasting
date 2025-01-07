from scipy.linalg import expm
import numpy as np

def predict_ar_diffusion_one_step(X_full, t, alpha, beta, L, tau, p):
    """

    Predicts the next state of the AR(p) diffusion model for a single time step.

    This function computes the predicted value `X_pred(t)` using the 
    AR(p) model with an optional diffusion term:
      X_pred(t) = sum_{k=1..p} alpha_k * X(t - k) + beta * (expm(-tau * L) @ X(t - 1)),
    where:
    - `alpha` are the autoregressive coefficients.
    - `beta` is the diffusion coefficient.
    - `L` is a Laplacian matrix for diffusion.
    - `tau` controls the intensity of the diffusion.

    Parameters
    ----------
    X_full : ndarray of shape (N, K)
        The full observed time series data:
        - N : number of nodes (dimensions).
        - K : number of time steps.

    t : int
        The current time index for which the prediction is made. 
        Must satisfy `t >= p` for the AR(p) model.

    alpha : ndarray of shape (p,)
        The AR coefficients for the past p steps.

    beta : float
        The diffusion coefficient.

    L : ndarray of shape (N, N)
        The Laplacian matrix used for the diffusion term. 
        Rows must sum to zero.

    tau : float
        The diffusion parameter. If `tau=0`, the diffusion term is disabled.

    p : int
        The AR model order (i.e., the number of past time steps to include).

    Returns
    -------
    X_pred : ndarray of shape (N,)
        The predicted value for the current time step.

    Notes
    -----
    - If `p=0`, only the diffusion term contributes to the prediction.
    - If `tau=0`, the diffusion term is deactivated, and the prediction is 
      based solely on the AR(p) model.
    - The Laplacian matrix `L` should be properly preprocessed to ensure it 
      satisfies the constraints (e.g., rows summing to zero).
    """
    N = L.shape[0]
    # --- AR(p) --- (if p>0)
    ar_part = np.zeros(N)
    for k in range(1, p+1):
        ar_part += alpha[k-1] * X_full[:, t-k]

    # --- Diffusion(tau) --- (if tau>0)
    if tau > 0:
        expm_L = expm(-tau * L)
        diff_part = beta * (expm_L @ X_full[:, t-1])
        return ar_part + diff_part
    else:
        return ar_part
