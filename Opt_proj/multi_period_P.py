from KCA import fitKCA
from opt import markowitz, benchmark
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def multi_period_P(df_P, df_r, t, main_path, q, risk_free=False):
    """calculate cumulative wealth of portfolio and benchmark
        Args: 
            df_P: weekly exchange rate data (2018/05/07 - 2023/04/24)
            df_r: weekly exchang rate pct change
            t: np.array of time indices 
            main_path: current working path
            q: Kalman specific param
            risk_free: wether to use risk free asset in portfolio (defualt to False)
        
        Returns:
            wealth: final wealth of portfolio
            benchmark_wealth: final wealth of benchmark portfolio
            data: dictionary that stores wealth and allocation info at each day, t
    """

    n = df_P.shape[1]
    e = np.ones(n)
    #look-back window for training data
    inception = 105
    rebalance_dates = t[inception:]
    df_r_matrix = np.array(df_r)

    allowable_risk = 2 #1.5
    trans_cost = 0.005 #0.005
    wealth = 10000
    wealth0, benchmark_wealth = wealth,  wealth

    #dictionary to store wealth info @ each day
    data = {'day':[inception-1],
            'benchmark_wealth':[wealth],
            'bench_risky':[e/(n+1)],
            'portfolio_wealth':[wealth],
            'port_risky':[e/(n+1)],
            'pred_km':[np.nan]}

    if risk_free:
        risk_free_rate = pd.read_pickle(main_path+'\\data\\risk_free_rate.pkl')
        rf = pd.merge(risk_free_rate, df_P, left_index=True, right_index=True, how='right')['Close']
        rf = rf.fillna(method='ffill') 
        
        benchmark_xx0, xx0 = 1/(n+1), 1/(n+1) # %money in bank
        benchmark_xx, xx = e/(n+1), e/(n+1)  # %money in each of risky asset
    else:
         benchmark_xx, xx = e/(n), e/(n)
        
    # 52 weeks in 1 year
    for i in range(52*2):
        trigger = False
        if wealth < 0:
            print('BANKRUPT!!!')
            break
        if wealth > benchmark_wealth:
            print('portfolio wins', benchmark_wealth, wealth)
        if i%4 == 0:
            print('1m check', benchmark_wealth, wealth)

        trade_date = rebalance_dates[i]
        start_date = 0
        window = trade_date - start_date
        # start_date = trade_date - window
        print('#', trade_date)
        data['day'].append(trade_date)

        
        ########## mu, V estimiation ###########  
        # Jane  
        # use KCA predictor for next period mu
        # fit KCA to all assets up to the trading date

        P_km = np.zeros((window + 1, n)) #window
        for j in range(n):
            z_r = np.array(df_P.iloc[:,j])[start_date : trade_date] #window
            xr_point, xr_bands = fitKCA(t[start_date : trade_date], z_r, q=q, fwd=1)[:2] #window
            P_km[:,j] = xr_point[:,0]
          
        # calculate weekly percentage return 
        r_km = np.diff(P_km, axis=0) / P_km[:-1,:]
        mu = r_km[-1,:]  # np.array(ret_pred)[j, :]  ##Jason/Chuming
        data['pred_km'].append(mu)
        
        """jason
        mu = np.array(preds)[i,:]
        """
        # use historical return for return covariance
        V = np.array(df_r)[start_date : trade_date,:].T.dot(np.array(df_r)[start_date : trade_date,:]) - mu.dot(mu.T) #window
        
        ##########################################
        
        if risk_free:
            mu0 = (1 + 0.01 * rf[trade_date]) ** (1 / 52) - 1

            # rebalancing benchmark portfolio
            benchmark_x0, benchmark_x = benchmark(benchmark_xx, trans_cost, benchmark_xx0,)
            benchmark_risk = np.sqrt(benchmark_x.T @ V @ benchmark_x)
            sigma = allowable_risk * benchmark_risk
            sigma = sigma.flatten()[0]


            ########### rebalancing optimized portfolio ##########
            try:
                x0, x = markowitz(mu, V, sigma, xx, trans_cost, mu0)
            except Exception as e:
                # print('*********', e)
                # print(np.linalg.eigvals(V))
                print('r_km', r_km.shape)
                print(t[:trade_date+1].shape)
                plt.plot(t[:trade_date+1], r_km)
                plt.show()
                raise
            
            ########## count P&L ##########
            wealth_rf = wealth * ((1 + mu0) * x0.value)
            wealth_risky = wealth * (1 + df_r_matrix[trade_date,:]) * x.value
            wealth = wealth_rf + np.sum(wealth_risky)
            data['portfolio_wealth'].append(wealth)

            # %wealth in rf and risky assets
            xx0 = wealth_rf / wealth
            xx = wealth_risky / wealth
            data['port_risky'].append(xx)


            benchmark_rf = benchmark_wealth * ((1 + mu0) * benchmark_x0)
            benchmark_risky = benchmark_wealth * (1 + df_r_matrix[trade_date,:]) * benchmark_x
            benchmark_wealth = benchmark_rf + np.sum(benchmark_risky)
            data['benchmark_wealth'].append(benchmark_wealth)

            # %benchmark in rf and risky assets
            benchmark_xx0 = benchmark_rf / benchmark_wealth
            benchmark_xx = benchmark_risky / benchmark_wealth
            data['bench_risky'].append(benchmark_xx)
        
        else:
            # rebalancing benchmark portfolio
            benchmark_x_val, benchmark_x = benchmark(benchmark_xx, trans_cost, risk_free=risk_free)
            benchmark_risk = np.sqrt(benchmark_x.T @ V @ benchmark_x)
            sigma = allowable_risk * benchmark_risk
            sigma = sigma.flatten()[0]


            ########### rebalancing optimized portfolio ##########
            try:
                x = markowitz(mu, V, sigma, xx, trans_cost, risk_free=risk_free)
            except np.linalg.LinAlgError as e:
                """
                try:
                    print('use previous weights:', x.value)
                except UnboundLocalError as e2:
                    print('triggered')
                    x = benchmark_x_val
                    trigger = True
                # raise
                """
                print('triggered')
                x = benchmark_x_val
                trigger = True
            ########## count P&L ##########
            # print('111111', df_r.iloc[trade_date].name)
            # print('222222',df_P.iloc[:trade_date])
            if trigger:
                wealth_risky = wealth * (1 + df_r_matrix[trade_date,:]) * x
            else:        
                wealth_risky = wealth * (1 + df_r_matrix[trade_date,:]) * x.value

            wealth = np.sum(wealth_risky)
            data['portfolio_wealth'].append(wealth)

            # %wealth in rf and risky assets
            xx = wealth_risky / wealth
            data['port_risky'].append(xx)


            benchmark_risky = benchmark_wealth * (1 + df_r_matrix[trade_date,:]) * benchmark_x_val
            benchmark_wealth = np.sum(benchmark_risky)
            data['benchmark_wealth'].append(benchmark_wealth)

            # %benchmark in rf and risky assets
            benchmark_xx = benchmark_risky / benchmark_wealth
            data['bench_risky'].append(benchmark_xx)
    return wealth, benchmark_wealth, data