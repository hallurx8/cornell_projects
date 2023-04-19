if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    import os
    import datetime as dt
    from KCA_mvp import fitKCA

    main_path = os.getcwd()
    preds = pd.DataFrame()

    df_r = pd.read_pickle(main_path+'\\data\\log_return.pkl')
    df_r = df_r.fillna(method='ffill').dropna()

    start_date = df_r.index[0]
    T_end = df_r.index[-1] - dt.timedelta(30)

    def rolling_window(start, end, length, df_r):
        """
        Input:
            start: 
                start of the enitre historical data used for training
            end: 
                end of the entire historical data used for training (end for the dataset - 30 days)
            length: 
                length of the look-back window (in years)
            df_r:
                dataframe of raw log return data
        """
        N = len(df_r.columns)
        start_date = start

        while start_date <= end:
            end_date = start_date + dt.timedelta(length*365)
            df_sub = df_r.loc[(df_r.index > start_date) & (df_r.index < end_date)]
            t_date = df_sub.index
            t = np.array(range(len(t_date)))/30
            fwd_steps = 30
            dt = t[1] - t[0]
            t_fwd = t.copy()

            for i in range(fwd_steps):
                 t_fwd = np.append(t_fwd, t_fwd[-1]+dt)

            for stock in range(N):
                z_r = np.array(df_r.iloc[:,stock])
                xr_point, xr_bands = fitKCA(t_fwd, z_r, q=1e-5, fwd=fwd_steps)[:2]
            
            start_date = start_date + dt.time



    start_date = pd.to_datetime('2018-03-01')
    end_date = start_date + dt.timedelta(5*365)
    period_pred = end_date + dt.timedelta(30)


    mean_accel_P = x_point[:,2].mean()
    mean_accel_r = xr_point[:,2].mean()

    std_accel_P = x_point[:,2].std()
    std_accel_r = xr_point[:,2].std()