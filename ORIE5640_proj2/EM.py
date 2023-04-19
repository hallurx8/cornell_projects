import numpy as np

def markow_switching_model(seq, mode, init):
    """
    Model:
        y_t - miu_s_t = beta_s_t + alpha_s_t * (y_t_1 - min_s_t_1) + e_t
        e_t~N(0,sigma_s_t)
    Input:
        seq: time sequence
        mode: model to use
        init: str
            "trend": upward (1) downward (0)
            "vol": high vol (1) low vol (0)
    Output:
        s_pred: Pr(S_t+1|S_t-1 = 1)
    """
    #initialize time sequence
    y = seq[~np.isnan(seq)]
    s0 = np.empty((len(y),1))

    #initialize S0 depending on init
    if init == 'trend':
        s0[0] = s0[1] = 0.5
        for t in range(2, len(y)):
            if (y[t] - y[t-1] > 0) and (y[t-1] - y[t-2] >0):
                s0[t] = 1 #set to upward trend if increase for 2 periods
            elif (y[t] - y[t-1] <0) and (y[t-1] - y[t-2] <0):
                s0[t] = 0 #set to downward trend if decreased for 2 periods
            else:
                s0[t] = 0.5 #neutural trend
    
    if init == 'vol':
        y_mean = np.mean(y)
        y_std = np.std(y)

        for t in range(0, len(y)):
            if abs(y[t] - y_mean) < y_std * 1.5:
                s0[t] = 0 #set to low vol
            elif abs(y[t] - y_mean) > y_std * 2.5:
                s0[t] = 1 #set to high vol
            else:
                s0[t] = 0.5 #neutural vol
    
    #selecting model
    if mode == 1:
        s_pred = MSMH2_AR0(y, s0)
    if mode == 2:
        s_pred = MSH2_AR1(y, s0)
    return s_pred

def MSMH2_AR0(y, s0):
    """
    Input:
        y: time series, numpy array
        s0: latent state vector, numpy array
    Output:
        s_pred: P(S_t+1|y_t,...,y_0)
    """
    T= len(y)
    alpha = 0
    beta = 0

    #define normal PDF (f) for MLE function
    def F(y_1, y_0, u_1, u_0, sigma, alpha, beta):
        return (1/(np.sqrt(2*np.pi)*sigma)) * \
            np.exp(-(y_1-u_1-beta-alpha*(y_0-u_0))**2/(2*sigma**2))
    #EM algo
    count = 0
    last_loglik = -999999

    while count < 100:
        """
        Maximization step
        """
        #constants for simplicity
        A = np.sum(1-s0[1:])
        B = np.sum(s0[1:])

        #params that do not depend on model specifications
        p_00 = np.dot(1-s0[1:], 1-s0[0:-1]) / np.sum(1-s0[0:-1])
        p_11 = np.dot(s0[1:], s0[0:-1]) / np.sum(s0[0:-1])
        Q = np.array([p_00, 1-p_11, 1-p_00, p_11]).reshape((2,2))

        #mu
        u_0 = np.dot(1-s0[1:], y[1:]) / A
        u_1 = np.dot(s0[1:], Y[1:])/ B

        #sigma
        sigma_0 = np.sqrt(np.dot(1-s0[1:], (y[1:]-u_0)**2) / A)
        sigma_1 = np.sqrt(np.dot(s0[1:], (y[2:]-u_1)**2) / B)

        """
        LogLik function
        """
        f00 = lambda y_1, y_0, sigma_0, alpha, beta : F(y_1, y_0, u_0, u_0, sigma_0, alpha, beta)
        f01 = lambda y_1, y_0, sigma_0, alpha, beta : F(y_1, y_0, u_0, u_1, sigma_0, alpha, beta)
        f10 = lambda y_1, y_0, sigma_1, alpha, beta : F(y_1, y_0, u_1, u_0, sigma_1, alpha, beta)
        f11 = lambda y_1, y_0, sigma_1, alpha, beta : F(y_1, y_0, u_1, u_1, sigma_1, alpha, beta)

        #filter
        gamma_00 = np.zeros((2,T-1))
        
        #forecast
        gamma_10 = np.zeros((2,T-1))

        #at t = 0
        gamma_00[:,0] = np.array([1-s0[0], s0[0]]).reshape((2,1))

        current_loglik = 0
        for t in range(1, T):
            gamma_10[:,t] = np.dot(Q, gamma_00[:,t-1])

            #calculate P(y_t|s_t=0, Y_t-1) and P(y_t|s_t=1, Y_t-1)
            Py_s0 = [f00(y[t], y[t-1], alpha, beta), f01(y[t], y[t-1], alpha, beta)].dot(gamma_00[:,t-1])
            Py_s1 = [f10(y[t], y[t-1], alpha, beta), f11(y[t], y[t-1], alpha, beta)].dot(gamma_00[:,t-1])

            #calculate current log likihood function
            C = np.array([Py_s0, Py_s1]).dot(gamma_10[:,t])
            current_loglik = current_loglik + np.log(C)

            #update gamma_00
            gamma_00[:,t] = np.array([Py_s0/C, Py_s1/C]).reshape((-1,1)) * gamma_10[:,t]

        """
        Break out of EM if converge
        """
        if abs(current_loglik - last_loglik) < 1e-6:
            break
        else:
            last_loglik = current_loglik
            count += 1

        """
        Expectation step
        """
        gamma_tT = np.zeros(2,T-1)
        gamma_tT[:,T-1] = gamma_00[:,T-1]

        for t in reversed(range(T-1)):
            gamma_tT[:,t] = Q.T.dot(gamma_tT[:,t+1] / gamma_10[:,t+1]) * gamma_00[:,t]

        s0 = gamma_tT[1,:].T

    """
    predicting next state s
    """
    s_pred = Q.dot(gamma_00[:,T])
    s_pred = np.array([gamma_00[1,1:].reshape((-1,1)), s_pred[1,:].reshape((-1,1))])

    return

def MSH2_AR1(y, s0):
    pass

