import numpy as np
import cvxpy as cp

def benchmark(benchmark_xx, trans_cost, benchmark_xx0=None, risk_free=False):
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
    if risk_free:
        req = benchmark_xx0 + cp.sum(benchmark_xx) - trans_cost * cp.sum(cp.abs(w - benchmark_xx))
        prob = cp.Problem(cp.Maximize(w), [(n+1) * w <= req])
        prob.solve()
        
        benchmark_x = w.value * np.ones((n,1))
        return w.value, benchmark_x

    else:
        req =  cp.sum(benchmark_xx) - trans_cost * cp.sum(cp.abs(w - benchmark_xx))
        prob = cp.Problem(cp.Maximize(w), [(n) * w <= req])
        prob.solve()

        benchmark_x = w.value * np.ones((n,1))
        return  w.value, benchmark_x


def markowitz(mu, V, sigma, xx, trans_cost, mu0=None, risk_free=False):
    """Optimizes portfolio
    
    Args:
        mu0: risk-free rate
        mu: rate of return of risky assets (pred output for t+1 from model)
        V: covariance matrix of the risky assets (historical weekly return)
        sigma: user-defined sigma
        xx = risky assets currently owned
        trans_cost: transaction cost per unit of weatlh invested/divested in assets

    Returns:

        
    """
    n = len(mu)
    mu = np.array(mu.reshape(-1,1))

    for tries in range(1):
        try:
            U = np.linalg.cholesky(V)
            break
        except np.linalg.LinAlgError as err:
            print('*******', 'markowitz', err)
            print(np.linalg.eigvals(V))
            # V = V + np.eye(V.shape[0])
            raise np.linalg.LinAlgError


    x0, total_trans_cost = cp.Variable(), cp.Variable()
    x, y = cp.Variable(n), cp.Variable(n)

    if risk_free:
        prob = cp.Problem(cp.Maximize((1 + mu0) * x0 + (1 + mu).T @ x),
                        [x0 + cp.sum(x) + total_trans_cost == 1,
                        x == xx + y, #y: asset amounts to be rebalanced
                        total_trans_cost >= trans_cost*cp.sum(cp.abs(y)),
                        cp.norm(U @ x) <= sigma,
                        x0 >= 0])
        
        prob.solve(solver=cp.ECOS)
    
        return x0, x
    else:
        prob = cp.Problem(cp.Maximize((1 + mu).T @ x),
                        [cp.sum(x) + total_trans_cost == 1,
                        x == xx + y, #y: asset amounts to be rebalanced
                        total_trans_cost >= trans_cost*cp.sum(cp.abs(y)),
                        cp.norm(U @ x) <= sigma])
        
        prob.solve(solver=cp.ECOS)
    
        return x