import numpy as np
def svar_loss(params, X, M, lambda1=1e-3, lambda2=1e-3, lambda1_l1=True, lambda2_l1=True):
    """
    Compute the loss function for the SVAR model.
    
    Parameters
    ----------
    params : array-like of shape (N*N + M*(M+1),)
        Flattened model parameters in the following order:
        [A_flat, c_flat],
        where:
        - A_flat : array of shape (N*N,)
            Flattened adjacency matrix A.
        - c_flat : array of shape (M*(M+1),)
            Flattened polynomial coefficients c.
            
    X : ndarray of shape (N, K)
        Time series data, where N is the number of nodes and K is the length
        of the time series.
        
    M : int
        Lag order for autoregression.
        
    lambda1 : float
        Regularization parameter for adjacency matrix A (L1 penalty).
        
    lambda2 : float
        Regularization parameter for coefficients c (L1 penalty).
        
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
    float
        The computed loss, which is the square root of the sum of squared
        errors plus any regularization penalty.
        """
    N, K = X.shape
    # Split the flat parameter vector into A and c
    A_flat, c_flat = np.split(params, [N * N])
    A = A_flat.reshape((N, N))
    c = c_flat.reshape((M, M + 1))

    # Build the residual starting at time index M
    residual = X[:, M:].copy()
    # Loop over lags from 1 to M
    for i in range(1, M + 1):
        # Polynomial in A up to power i, weighted by c[i-1, :]
        polynomial = sum(
            c[i - 1, j] * np.linalg.matrix_power(A, j)
            for j in range(i + 1)
        )
        # Subtract from residual
        residual -= polynomial @ X[:, M - i : K - i]

    # L1 or L2 penalty on A and c
    if lambda1_l1:
        reg_A = lambda1 * np.sum(np.abs(A))
    else:
        reg_A = lambda1 * np.sum(A**2)
    if lambda2_l1:
        reg_c = lambda2 * np.sum(np.abs(c))
    else:
        reg_c = lambda2 * np.sum(c**2)

    # Frobenius norm of residual plus regularizations
    return 0.5 * np.linalg.norm(residual, 'fro')**2 + reg_A + reg_c