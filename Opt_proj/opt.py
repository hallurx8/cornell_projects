import numpy as np
import cvxpy as cp

def adaptedStats(log_return, trade_date, horizon, rate_of_decay=0):
    """Projects filtered daily data to monthly holding data
    
    Args:
        log_return: filtered log_return from Kalman/ARIMA/GARCH/MCM
        trade_date: the day when rebalancing happens
        horizon: days of holding period (for monthly rebalacing, horizon=20)
        rate_of_decay: assigning weight to each return in history (the older, 
        the smaller weight). Default setting is 0

    Retuns:
        mu: arithmetic mean vector of historical returns 
        V: covariance matrix of arithmetic means
    """
    h = horizon
    c_r = log_return
    t_d = trade_date
    n_s = trade_date #use all the historical data available so far
    r_d = rate_of_decay

    s_d = np.array(range(t_d))
    s_c_r = c_r[s_d,:]  #sampled log return matrix

    #construct weights
    w = (1 - r_d)**np.array(range(n_s))
    w = w[::-1]/np.sum(w)

    #mean vector of compounded returns
    mean_c_r = s_c_r.T.dot(w)

    #covariance matrix of compounded returns
    cov_c_r = s_c_r.T.dot(np.diag(w)).dot(s_c_r) - mean_c_r.dot(mean_c_r.T)

    #adapted mean
    adapted_mean_c_r = h * mean_c_r

    #adapted covariance
    adapted_cov_c_r = h * cov_c_r

    #arithmetic mean for markowitz
    mu = np.exp(adapted_mean_c_r + 0.5*np.diag(adapted_cov_c_r)) - 1

    #covariance matrix of arithmetic mean
    V = mu.dot(mu.T)*(np.exp(adapted_cov_c_r) - 1)

    return mu, V

def benchmark(benchmark_xx0, benchmark_xx, trans_cost):
    """Calculates equally weighted benchmark portfolio
    
    Args:
        benchmark_xx0: wealth now in the bank
        benchmark_xx: wealth now invested in risky assets
        trans_cost: transactopm costs per unit of wealth invested/divested in assets

    Returns:
        benchmark_x0: new wealth currently in bank (scalar)
        benchmark_x: new wealth currently in risky assests (vector of size n)
    """
    n = len(benchmark_xx)
    w = cp.Variable()
    req = benchmark_xx0 + cp.sum(benchmark_xx) - trans_cost * cp.sum(cp.abs(w - benchmark_xx))
    prob = cp.Problem(cp.Maximize(w), [(n+1) * w <= req])
    

    prob.solve()
    print("benchmark optimization succesful")
        

    benchmark_x0 = w.value
    benchmark_x = w.value * np.ones((n,1))
    return benchmark_x0, benchmark_x

def markowitz(mu0, mu, V, sigma, xx, trans_cost):
    """Optimizes portfolio
    
    Args:
        mu0: risk-free rate
        mu: rate of return of risky assets
        V: covariance matrix of the risky assets
        sigma: user-defined sigma
        xx = risky assets currently owned
        trans_cost: transaction cost per unit of weatlh invested/divested in assets

    Returns:

        
    """
    n = len(mu)
    mu = np.array(mu.reshape(-1,1))
    # U = np.linalg.cholesky(V)
    e = np.ones((n, 1))

    x0, total_trans_cost = cp.Variable(), cp.Variable()
    x, y = cp.Variable(n), cp.Variable(n)

    prob = cp.Problem(cp.Maximize((1 + mu0) * x0 + (e + mu).T @ x),
                      [x0 + cp.sum(x) + total_trans_cost == 1,
                       x == xx + y, #y: asset amounts to be rebalanced
                       total_trans_cost >= trans_cost*cp.sum(cp.abs(y)),
                    #    cp.norm(U.dot(x)) <= sigma,
                       x.T @ V @ x <= sigma**2,
                       x0 >= 0])
    try:
        prob.solve()
        print('markowitz optimization successful')
    except cp.DCPError as e:
        print("ERROR occured:", str(e))
        
    return x0, x