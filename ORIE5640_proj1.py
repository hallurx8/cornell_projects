#!/usr/bin/env python
# coding: utf-8

# In[2]:


import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from copy import copy
import random

def pd_readcsv(filename):
    """
    Reads the pandas dataframe from a filename, given the index is correctly labelled
    """
    ans=pd.read_csv(filename)
    ans.index=pd.to_datetime(ans['DATETIME'])
    del ans['DATETIME']
    ans.index.name=None
    
    return ans

def log_return(data):
    """
    input: an price dataframe
    output: dataframe with log return
    """
    price = data['ADJ'].ffill()
    r = np.log(price) - np.log(price.shift(1))
    
    return r


# In[3]:


import os

def get_code(directory):
    """
    get a list of instrument code from specified directory
    input: dir
    output: list
    """
    filelist = []
    directory = directory + '\\data'
    for filename in os.listdir(directory):
        if'_price.csv' in filename:
            filelist.append(filename)
    return filelist
        
directory = os.getcwd()
filelist = get_code(directory)


# # 1a. Historgram to check skew for single stock

# In[19]:


import seaborn as sns
from scipy.stats import skewnorm
from scipy.stats import lognorm
from scipy.optimize import minimize
from sstudentt import SST

log_returns_dict = {}
for stock in filelist:
    data = pd_readcsv('data/'+ stock)
    r = log_return(data)
    name = stock.replace('_price.csv','')
    log_returns_dict[name] = r
    
#fitting a lognormal distribution
data = log_returns_dict['VIX']
cleaned_data = data[np.isfinite(data)].values
nu, loc, scale = lognorm.fit(cleaned_data)

#lognormal distribution
def loglik_log(params,y):
    return -np.sum(np.log(lognorm.pdf(y, params[0], params[1], np.exp(params[2]))))

params_l = [0.1, 1,1]
result_l = minimize(loglik_log, params_l, args=(cleaned_data,), method='Nelder-Mead')

#fitting a skewed-t distribution
def loglik(params,y):
    dist = SST(mu=params[0], sigma=params[1], nu=params[2], tau=params[3])
    return -np.sum(np.log(dist.d(y)))

params0 = [1,1,1,5]
result = minimize(loglik, params0, args=(cleaned_data,), method='Nelder-Mead')
dist_plot = SST(mu=result.x[0], sigma=result.x[1], nu=result.x[2], tau=result.x[3])

#plotting
sns.set()
sns.set_theme(style='white')
sns.set_palette("Paired",9)
sns.histplot(data, bins=30, stat='density').set(title='log return distribution for VIX')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p_log = lognorm.pdf(x,nu,loc,scale)
sns.lineplot(x, p_log, linewidth=2, color='r')
p_skew = dist_plot.d(x)
sns.lineplot(x, p_skew, linewidth=2, color='b')
plt.legend(labels=["log normal fit", "skew-t fit"])
plt.show()


# In[14]:


log_returns_dict['VIX'].skew()


# In[9]:


result.x


# In[20]:


result_l.x


# In[84]:


#generate data for skew-t QQ plot
sorted_clean = np.sort(cleaned_data)
quant = np.linspace(0,1,len(sorted_clean))
theoretical_data = np.quantile(dist_plot.r(500), quant)

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(theoretical_data, sorted_clean)
# ax.scatter(empirical_q, sorted_clean, linestyle='-', marker='.')

from sklearn.linear_model import LinearRegression
model = LinearRegression()
x = theoretical_data.reshape(-1,1)
y = sorted_clean.reshape(-1,1)
model.fit(x, y)


# In[85]:


################
from scipy.stats import probplot
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

probplot(cleaned_data, dist=lognorm(s=nu, loc=loc, scale=scale), plot=ax1)

ax2.scatter(theoretical_data, sorted_clean)
ax2.plot(theoretical_data,model.predict(x),color='r')
ax2.set_xlabel('Theoretical quantiles')
ax2.set_ylabel('Ordered Values')
ax2.set_title('Probability Plot')
plt.show()


# # 1b. Resampled skewness for all futures

# In[17]:


def resampled_skew(data, niter=500):
    """
    resample the skew to lower estimate error
    """
    skew_distribution = []
    for i in range(niter):
        resample_index = [int(random.uniform(0,len(data))) for _ in range(len(data))] #resampe with replacement
        resample_data = data[resample_index]
        resample_skew = resample_data.skew()
        skew_distribution.append(resample_skew)
    return skew_distribution


# In[18]:


skew_distribution_dict = {}
for file in filelist:
    try:
        name = file.replace('_price.csv','')
        abs_path = os.getcwd() + '\\data\\' + file
        data = pd_readcsv(abs_path)
        r = log_return(data)
        x = resampled_skew(r)
        sd = np.std(x)
        x_norm = x/sd #normalized
        y = pd.Series(x_norm)
        skew_distribution_dict[name] = y
    except Exception as e:
        print(f"Error: {e}, at loading {name}")
        
    
df_skew_distribution = pd.DataFrame(skew_distribution_dict)
df_skew_distribution = df_skew_distribution.reindex(df_skew_distribution.mean().sort_values().index,axis=1)    


# In[64]:


matplotlib.rc_file_defaults()
plt.figure(figsize=(12,3))
df_skew_distribution.boxplot()
plt.xticks(rotation=90)
plt.show()


# In[58]:


plt.figure(figsize=(12,3))
new_df = df_skew_distribution.drop(columns=['LEANHOG','CRUDE_W'])
new_df.boxplot()
plt.xticks(rotation=90)
plt.show()


# # 2. Return vs. Skew

# ## a. Bootstrapped expected return distribution

# In[21]:


def resample_mean(data, niter=500):
    mean_est_distribution = []
    for _ in range(niter):
        resample_index = [int(random.uniform(0,len(data))) for _ in range(len(data))]
        resample_data = data[resample_index]
        mean_est = resample_data.mean()
        mean_est_distribution.append(mean_est)
        
    return mean_est_distribution

df_mean_distribution = {}
for file in filelist:
    try:
        name = file.replace('_price.csv','')
        abs_path = os.getcwd() + '\\data\\' + file
        data = pd_readcsv(abs_path)
        r = log_return(data)
        x = resample_mean(r)
        sd = np.std(x)
        x_norm = x/sd
        y = pd.Series(x_norm)
        df_mean_distribution[name] = y
    except Exception as e:
        print(f"Error: {e}, at loading {name}")
        


# In[67]:


plt.figure(figsize=(12,3))
df_mean_distribution = pd.DataFrame(df_mean_distribution)
df_mean_distribution = df_mean_distribution[new_df.columns]
df_mean_distribution.boxplot()
plt.xticks(rotation=90)
plt.show()


# ## b. Bootstrapped SR distribution

# In[23]:


def resample_SR(data, niter=500):
    SR_est_distribution = []
    for _ in range(niter):
        resample_index = [int(random.uniform(0,len(data))) for _ in range(len(data))]
        resample_data = data[resample_index]
        SR_est = resample_data.mean()/resample_data.std()
        SR_est_distribution.append(SR_est)
        
    return SR_est_distribution

df_SR_distribution = {}
for file in filelist:
    try:
        name = file.replace('_price.csv','')
        abs_path = os.getcwd() + '\\data\\' + file
        data = pd_readcsv(abs_path)
        r = log_return(data)
        x = resample_SR(r)
        y = pd.Series(x)
        df_SR_distribution[name] = y
    except Exception as e:
        print(f"Error: {e}, at loading {name}")


# In[65]:


plt.figure(figsize=(12,3))
df_SR_distribution = pd.DataFrame(df_SR_distribution)
df_SR_distribution = df_SR_distribution[new_df.columns]
df_SR_distribution.boxplot()
plt.xticks(rotation=90)
plt.show()


# ## c. Group stocks into skews above average and below average

# In[25]:


# calculate the mean skew for each security
skew_by_code = df_skew_distribution.mean()
# calculate the mean for all securities
avg_skew = np.mean(skew_by_code.values)
# find the stocks with lower/higher than avergage skew
low_skew_codes = list(skew_by_code[skew_by_code<avg_skew].index)
high_skew_codes = list(skew_by_code[skew_by_code>=avg_skew].index)


# In[26]:


def resampled_mean_estimator_multiple_codes(returns, code_list, niter=500):
    mean_estimate_distribution = []
    for _ in range(niter):
        # randomly choose a stock in the given list       
        code = code_list[int(random.uniform(0, len(code_list)))]
        data = returns[code]
        # resample the return of that stock
        resample_index = [int(random.uniform(0,len(data))) for _ in range(len(data))]
        resampled_data = data[resample_index]
        sample_mean_estimate = resampled_data.mean()
        mean_estimate_distribution.append(sample_mean_estimate)

    return mean_estimate_distribution

df_mean_distribution_multiple = dict()
df_mean_distribution_multiple['High skew'] = resampled_mean_estimator_multiple_codes(log_returns_dict,high_skew_codes,1000)
df_mean_distribution_multiple['Low skew'] = resampled_mean_estimator_multiple_codes(log_returns_dict,low_skew_codes,1000)


# In[68]:


df_mean_distribution_multiple = pd.DataFrame(df_mean_distribution_multiple)
df_mean_distribution_multiple.boxplot()
plt.ylim(-0.001,0.002)


# In[88]:


df_mean_distribution_multiple.median()*250


# In[28]:


def resampled_SR_estimator_multiple_codes(returns, code_list, niter=500):
    SR_estimate_distribution = []
    for _ in range(niter):
        # randomly choose a stock in the given list       
        code = code_list[int(random.uniform(0, len(code_list)))]
        data = returns[code]
        # resample the return of that stock
        resample_index = [int(random.uniform(0,len(data))) for _ in range(len(data))]
        resampled_data = data[resample_index]
        sample_SR_estimate = resampled_data.mean()/resampled_data.std()
        SR_estimate_distribution.append(sample_SR_estimate)

    return SR_estimate_distribution

df_SR_distribution_multiple = dict()
df_SR_distribution_multiple['High skew'] = resampled_SR_estimator_multiple_codes(log_returns_dict,high_skew_codes)
df_SR_distribution_multiple['Low skew'] = resampled_SR_estimator_multiple_codes(log_returns_dict,low_skew_codes)


# In[66]:


df_SR_distribution_multiple = pd.DataFrame(df_SR_distribution_multiple)
df_SR_distribution_multiple.boxplot()
# plt.ylim(-0.001,0.002)


# # 3a Assets with current negative skew outperform one with positive skew?

# In[45]:


from scipy.stats import ttest_ind
all_SR_list = []
count = 0
loop = 0

# t-stat tells how likely there exists a difference, but it does not tell the magnitude
all_tstats=[]

# Define "current" with multiple time horizon
all_frequencies = ["7D", "14D", "1M", "3M", "6M", "12M"]

for freq in all_frequencies:
    all_results = []
    for name in log_returns_dict:
        
            # rolling windows            
            log_returns = log_returns_dict[name]
            start_date = log_returns_dict[name].index[0]
            end_date = log_returns_dict[name].index[-1]

            periodstarts = list(pd.date_range(start_date, end_date, freq=freq)) + [end_date]
            
            for periodidx in range(len(periodstarts) - 2):               
                p_start = periodstarts[periodidx]+pd.DateOffset(-1)
                p_end = periodstarts[periodidx+1]+pd.DateOffset(-1)
                s_start = periodstarts[periodidx+1]
                s_end = periodstarts[periodidx+2]

                period_skew = log_returns_dict[name][p_start:p_end].skew()
                subsequent_return = log_returns_dict[name][s_start:s_end].mean()
                subsequent_vol = log_returns_dict[name][s_start:s_end].std()
                
                # there are 118 ZeroDivisionError out of 57000 datapoints
                try:
                    subsequent_SR = 16*(subsequent_return / subsequent_vol)
                except ZeroDivisionError:
                    subsequent_SR = np.nan
                    #print("Division by 0")
                    

                if np.isnan(subsequent_SR) or np.isnan(period_skew):
                    continue                
                else:
                    all_results.append([period_skew, subsequent_SR])
                    
    all_results=pd.DataFrame(all_results, columns=['x', 'y'])
    avg_skew=all_results.x.median()
    all_results[all_results.x>avg_skew].y.median()
    all_results[all_results.x<avg_skew].y.median()

    subsequent_distribution = dict()
#     subsequent_distribution['High_skew'] = all_results[all_results.x>=avg_skew].y
#     subsequent_distribution['Low_skew'] = all_results[all_results.x<avg_skew].y
    subsequent_distribution['Pos_skew'] = all_results[all_results.x>=0].y
    subsequent_distribution['Neg_skew'] = all_results[all_results.x<0].y
    
    subsequent_distribution = pd.DataFrame(subsequent_distribution)

    med_SR =subsequent_sr_distribution.median()
#     tstat = ttest_ind(subsequent_distribution.High_skew, subsequent_distribution.Low_skew, nan_policy="omit").statistic
    tstat = ttest_ind(subsequent_distribution.Pos_skew, subsequent_distribution.Neg_skew, nan_policy="omit").statistic

    all_SR_list.append(med_SR)
    all_tstats.append(tstat)
    
all_tstats = pd.Series(all_tstats, index=all_frequencies)
all_tstats.plot(x="Frequency", y="t-statistics")


# In[44]:


all_tstats.plot(x="Frequency", y="t-statistics")
# plt.title("t-stat for the null hypothesis that positive and negative skewness create same expected return")
plt.xlabel("Frequency")
plt.ylabel("t-statistics")




