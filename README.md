


# Time Series Forecasting with Graph-Based Models

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juliendelavande/graph-time-series-forecasting/flue.ipynb)

This project explores graph-based approaches for time series forecasting, leveraging **graph learning** techniques to uncover meaningful relationships in data. We implemented two models:

1. **AR+Diffusion Model**: Combines autoregressive (AR) components with graph diffusion dynamics.
2. **SVAR Model**: Structural Vector Autoregression with polynomial expansions of the adjacency matrix.

### Data
The project uses COVID-19 weekly case counts from multiple countries, sourced from [Our World in Data](https://github.com/owid/covid-19-data). Data preprocessing includes:
- Aggregating daily cases into weekly data.
- Differencing and clipping negative values.
- Scaling with MinMaxScaler.

### Key Features
- **Graph Learning**: Infers graph structure using adjacency matrices and Laplacian operators.
- **Forecasting Models**: Implements AR+Diffusion and SVAR with grid search for hyperparameter tuning.
- **Visualization**: Generates insights into learned graphs and forecasting performance.

### Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/juliendelavande/graph-time-series-forecasting.git
   cd graph-time-series-forecasting
   ```

2. use the function to fit AR+Diffusion or SVAR modelss stored in the `src` folder:
   ```python
    # AR + Diffusion 
    from src.ar_diffision.fit import fit_ar_diffusion
    from src.ar_diffusion.predict import predict_ar_diffusion_one_step
    from sklearn.preprocessing import MinMaxScaler
    

    p = 3                   # order of the AR model
    tau = 0.001             # time constant of the diffusion
    lambda_reg = 0.0        # regularization parameter
    l1 = True               # use L1 regularization (True) or L2 regularization (False)
    method = "L-BFGS-B"     # optimization method
    X = data.values.T       # data matrix
    N, K = X.shape          # N=number of nodes and K=number of time points
    scaler = MinMaxScaler() # data normalization
    X_norm = scaler.fit_transform(X.T).T

    alpha_opt, beta_opt, L_opt, result = fit_ar_diffusion(
        X_norm, p=p, tau=tau, lambda_reg=lambda_reg, l1=l1, method=method
    )

    X_pred = predict_ar_diffusion_one_step(X_norm, i, alpha_opt, beta_opt, L_opt, tau, p)

    # SVAR
    from src.svar.predict import predict_svar_one_step
    from src.svar.fit import estimate_adjacency_matrix_simplified

    X = data.values.T
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X.T).T
    N, K= X_norm.shape[0], X_norm.shape[1]
    M = 3                   # lag order
    lambda1, lambda2 = 0, 0 # regularization parameters
    tol = 1e-5              # tolerance for optimization
    max_iter = 300          # maximum number of iterations    
    method = "L-BFGS-B"     # optimization method
    lambda1_l1, lambda2_l1 = True, True # use L1 regularization (True) or L2 regularization (False)

    A_hat, c_hat = estimate_adjacency_matrix_simplified(
        X_norm, M, lambda1, lambda2, tol=tol, max_iter=max_iter, 
        method=method, lambda1_l1=lambda1_l1, lambda2_l1=lambda2_l1
        )
    x_pred = predict_svar_one_step(X_norm[:, t - M:t], A_hat, c_hat)
   ```

### Requirements
- Python 3.8+
- Key libraries: `numpy`, `scipy`, `matplotlib`, `pandas`, `jupyter`, `scikit-learn`, `geopandas`


### Results
- **AR+Diffusion**: Best for moderate diffusion strength, capturing intra- and inter-country dependencies.
- **SVAR**: Superior performance, capturing key temporal and structural patterns.

### Contributions
- [Soël Megdoud](mailto:soel.megdoud@ens-paris-saclay.fr): AR+Diffusion implementation, paper review.
- [Julien Delavande](mailto:julien.delavande@ens-paris-saclay.fr): SVAR implementation, hyperparameter tuning, analysis.

### References
- Dong et al. (2019). *Learning Graphs from Data: A Signal Representation Perspective.*
- Additional references included in the [report](./report.pdf).
