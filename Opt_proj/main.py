if __name__ == '__main__':
    """ overhead function to store performance matrix dataframe
        Params:
            q: Kalman filter specific param
        Returns:
            df: portfolio and benchmark wealth and weights info at each time, t

    """

    import numpy as np
    import pandas as pd
    import os
    import datetime as dt
    from multi_period_P import multi_period_P

    ######### import data #########
    main_path = os.getcwd()

    """jason
    my_preds = pd.read_pickle(main_path+'/data/jason_preds_fit_sv_pf.pkl')
    my_preds = my_preds.drop(my_preds.index[0])
    """
    # #assets considered to be dropped
    # a_list = ['ZT', 'ZF']
    # #load price (exchange rate) data
    df_P = pd.read_pickle(main_path+'\\data\\future_price_bear.pkl')
    df_P = df_P.fillna(method='ffill').drop(df_P.index[0])


    # #weekly return
    df_r = pd.read_pickle(main_path+'\\data\\future_return_bear.pkl')
    df_r = df_r.fillna(method='ffill').dropna()


    
    ######## rebalancing ########
    t_date = df_P.index
    t = np.array(range(len(t_date)))

    """Jane"""
    for i, q in enumerate([1e-7, 1e-6, 1e-5]):
        wealth, benchmark_wealth, data = multi_period_P(df_P, df_r, t, main_path, q)

        print('final wealth of portfolio', wealth)
        print('final weatlh of benchmark', benchmark_wealth)
        df = pd.DataFrame(data=data)
        df.to_csv(f'bear2_performance_result_2y{i}.csv')
        df.to_pickle(f'bear2_performance_result_2y{i}.pkl')
    
    """jason
    wealth, benchmark_wealth, data = multi_period_P(df_P, df_r, t, main_path, my_preds)
    

    # print(data)
    print('final wealth of portfolio', wealth)
    print('final weatlh of benchmark', benchmark_wealth)

    df = pd.DataFrame(data=data)
    df.to_csv('Jason_performance_result_SV.csv')
    df.to_pickle('Jason_performance_result_SV.pkl')
    print('saved')
    """