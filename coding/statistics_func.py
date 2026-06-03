
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stattools

#########################
# Statistics functions ##
#########################

def acf_calc_plot(data, N_eff=None, plot=True, title="Autocorrelation Function am Sample Size vs Lag"):
    if not N_eff:
        N_eff = len(data)
    acf_values = stattools.acf(data, nlags=N_eff-1)
    lags = np.arange(len(acf_values))
    sample_sizes = N_eff - lags
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5.5))

        ax1.plot(acf_values, marker="o", linestyle="-")
        ax1.axhline(0, color="gray", linestyle="--")
        ax1.set_xticks(lags)
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Autocorrelation")

        ax2.scatter(lags, sample_sizes)
        ax2.set_xticks(lags)
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Sample Size")

        fig.suptitle(f"{title}")
        
    return acf_values, lags, sample_sizes


def integral_time_scale(data, del_t=10, method="0cross"):
    # del t is sampling interval in days, for rapid monthly data del t = 10 days
    # note all data has to be in the same time units!
    # get acf values for the whole dataset
    acf_values = stattools.acf(data, nlags=len(data)-1) 

    if method == "0cross":
        lag_0cross = np.where(np.diff(np.sign(acf_values)))[0][0] + 1
        # get acf values up to the neede lag 
        acf_values_0 = stattools.acf(data, nlags=lag_0cross) 
        max_lag = lag_0cross
    
    if method == "plateau":
        # find the first lag where the acf drops below 1/e
        lag_plateau = np.where(acf_values < 1/np.e)[0][0]
        acf_values_0 = stattools.acf(data, nlags=lag_plateau) 
        max_lag = lag_plateau
        
    #  calculate its using the N_eff recevied depeding on the method   
    its = del_t  * sum(1 + 2*(max_lag-j)/max_lag*acf_values_0[j] for j in range(1, max_lag-1)) # for normalized acf
    # its = del_t  * (1 + sum( 2*(N_eff-j)/N_eff*acf_values_0[j] for j in range(1, N_eff-1)) )# for normalized acf
    
    return its, max_lag

def standard_error(data, data_std, total_time, del_t=10, method="0cross", time_unit="days", ):
    
    its, N_eff = integral_time_scale(data, del_t=del_t, method=method)
    
    DOF = total_time / (2 * its) # degrees of freedom
    # total time = N * del t, 
    #where N is the number of data points, del t is the sampling interval in time units (e.g., days, months)
    se = data_std / np.sqrt(DOF)
    
    return se, its, N_eff, DOF