if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import os
    import datetime as dt
    from opt import adaptedStats, markowitz, benchmark

    #import data
    main_path = os.getcwd()
    start_date = pd.to_datetime('2018-03-01')
    end_date = start_date + dt.timedelta(5*365)
    period_pred = end_date + dt.timedelta(30)

    df_P = pd.read_pickle(main_path+'\\data\\exchange_rates.pkl')
    df_P = df_P.fillna(method='ffill').drop(df_P.index[0])
    df_P = df_P.loc[(df_P.index > start_date) & (df_P.index < end_date)]

    df_r = pd.read_pickle(main_path+'\\data\\log_return.pkl')
    df_r = df_r.fillna(method='ffill').dropna()
    df_r = df_r.loc[(df_r.index > start_date) & (df_r.index < end_date)]

    t_date = df_P.index
    t_day = np.array(range(len(t_date)))
    t = np.array(range(len(t_date)))/20 #20 trading days


    #rebalance params
    r_km = np.array(pd.read_pickle('r_km.pkl')) #import your own filtered log returns
    horizon = 20 # rebalance monthly (every 20 trading days)
    start = 250 #initial rebalancing strats at the 250th trading day
    number_rebalances = int((t_day[-1]-start)/horizon) + 1
    rebalance_dates = start + horizon * np.array(range(number_rebalances))
    rate_of_decay = 0 #no decay    


    #set up initial states of an equally-weighted portfolio
    risk_free_rate = pd.read_pickle(main_path+'\\data\\risk_free_rate.pkl')
    rf = pd.merge(risk_free_rate, df_r, left_index=True, right_index=True, how='right')['Close']
    rf = rf.fillna(method='ffill') 
    n = r_km.shape[1]
    e = np.ones(n)

    allowable_risk = 1.5
    trans_cost = 0.005
    wealth = 10000
    wealth0, benchmark_wealth = wealth,  wealth

    benchmark_xx0, xx0 = 1/(n+1), 1/(n+1) # %money in bank
    benchmark_xx, xx = e/(n+1), e/(n+1)  # %money in each of risky asset


    #generate montly mu and covariance data using filtered data
    trade_date = rebalance_dates[0]
    mu, V = adaptedStats(r_km, trade_date, horizon, rate_of_decay)    
    mu0 = (1 + 0.01 * rf[trade_date - 1]) ** (horizon / 270) - 1


    ##########################TEST####################################
    #only the first rebalnce is coded here

    # rebalancing benchmark portfolio
    benchmark_x0, benchmark_x = benchmark(benchmark_xx0, benchmark_xx, trans_cost)
    benchmark_risk = np.sqrt(benchmark_x.T.dot(V).dot(benchmark_x))
    sigma = allowable_risk * benchmark_risk

    # rebalancing optimized portfolio
    x0, x = markowitz(mu0, mu, V, sigma, xx, trans_cost)