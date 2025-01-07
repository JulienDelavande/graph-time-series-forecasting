import numpy as np

def predict_svar_one_step(x_last_M, A_hat, c_hat):
    """
    Predict the next value given the M last outputs using the
    polynomial expansions on a single adjacency matrix A_hat.

    Parameters
    ----------
    x_last_M : ndarray of shape (N, M)
        The last M outputs, columns are x[t-1], x[t-2], ..., x[t-M].
    A_hat : ndarray of shape (N, N)
        Estimated adjacency matrix.
    c_hat : ndarray of shape (M, M+1)
        Estimated polynomial coefficients.

    Returns
    -------
    x_pred : ndarray of shape (N,)
        Predicted next value.
    """
    N, M = x_last_M.shape
    x_pred = np.zeros(N)

    for i in range(1, M + 1):
        polynomial = sum(
            c_hat[i - 1, j] * np.linalg.matrix_power(A_hat, j)
            for j in range(i + 1)
        )
        # x_last_M[:, -i] corresponds to the most recent i-th column
        x_pred += polynomial @ x_last_M[:, -i]

    return x_pred