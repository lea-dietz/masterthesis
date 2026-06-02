
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stattools

#########################
# Statistics functions ##
#########################

def acf_calc_plot(data, N_eff=None, plot=True):
    if not N_eff:
        N_eff = len(data)
    acf_values = stattools.acf(data, nlags=N_eff-1)
    lags = np.arange(len(acf_values))
    sample_sizes = N_eff - lags
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 5.5))

        ax1.plot(acf_values, marker="o", linestyle="-")
        ax1.axhline(0, color="gray", linestyle="--")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("Autocorrelation")

        ax2.scatter(lags, sample_sizes)
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Sample Size")

        fig.suptitle("Autocorrelation Function and Sample Size vs Lag")
        
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
        N_eff = lag_0cross
        
    #  calculate its using the N_eff recevied depeding on the method   
    its = del_t  * sum(1 + 2*(N_eff-j)/N_eff*acf_values_0[j] for j in range(1, N_eff-1)) # for normalized acf
    # its = del_t  * (1 + sum( 2*(N_eff-j)/N_eff*acf_values_0[j] for j in range(1, N_eff-1)) )# for normalized acf
    
    return its, N_eff

def standard_error(data, data_std, del_t=10, method="0cross"):
    
    its, N_eff = integral_time_scale(data, del_t=del_t, method=method)
    
    T_days = (data.time.values[-1] - data.time.values[0]) / np.timedelta64(1, 'D') # total time in days
    DOF = T_days / (2 * its) # degrees of freedom
    se = data_std / np.sqrt(DOF)
    
    return se, its, N_eff, DOF