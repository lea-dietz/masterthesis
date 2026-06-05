
import numpy as np
import xarray as xr
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from scipy.stats import t as t_dist

import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import ccf

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



def critical_r(n_eff, alpha=0.05):
    # t_stat = r* np.sqrt(n_eff - 2) / np.sqrt(1 - r**2)
    # rearrage to get critical r 
    df = n_eff - 2 # loose 2 dof since mean is estimated for both ts
    # with alpha 0.05 we get 97.5 percentile of t distribution
    t_crit = t_dist.ppf(1 - alpha/2, df=df)
    r_crit = t_crit / np.sqrt(t_crit**2 + df)
    return r_crit


#########################
# Plotting   functions ##
#########################

   
def plot_hovmöller(mht_data, lats, anomalies=False, cmap="bwr", vmin=None, vmax=None, savefig=False):
    """
    Plot the MHT (can be anomalies or raw) as a function of time and latitude using a colormap.
    """
    if isinstance(mht_data, dict):
        MHT = mht_data["MHT_statistics"]["mean"] # this is mean over posterior samples!
        mht_all_mean = MHT.mean(dim="TIME")
    else:
        # asume its the mht mean over post samples
        MHT = mht_data
        mht_all_mean = MHT.mean(dim="TIME")
    # for anomalies
    # mht_all_mean = MHT_dict["MHT_statistics"]["time_mean"].mean(dim="posterior_samples") # this is time mean and over posterior samples mean

    fig, ax = plt.subplots(figsize=(9.5, 7))

    if anomalies:
        # cmap = "bwr"
        # Anomaly = value - mean
        anom = MHT - mht_all_mean.mean(dim="lat") # get overall mean over all latitudes!
        vmin = np.nanmin(anom.values)
        vmax = np.nanmax(anom.values)
        limit = max(abs(vmin), abs(vmax))
        vmin = -limit #- 0.5
        vmax = limit #+ 0.5
        mesh = ax.pcolormesh(lats, MHT.TIME, anom.values.T, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, label='MHT Anomaly from Mean (PW)')
        plt.title('Meridional Heat Transport Anomalies (MHT)')
        savelabel = "mht_time_lat_anomalies.png"

    else:
        if vmin is None and vmax is None:
            vmin = np.nanmin(MHT.values)
            vmax = np.nanmax(MHT.values)
            limit = max(abs(vmin), abs(vmax))
            vmin = -limit 
            vmax = limit 
        
        mesh = ax.pcolormesh(lats, MHT.TIME, MHT.values.T, shading='auto', cmap=cmap, vmin=0, vmax=vmax)
        plt.colorbar(mesh, label='MHT (PW)')
        plt.title('Meridional Heat Transport  (MHT)')
        savelabel = "mht_time_lat.png"

    xtick_pos    = [-35, -25, -11, -5,  5,  16, 26, 35, 45, 55, 60]
    xtick_labels = ["35S","25S","11S","5S","5N","16N","26N","35N","45N","55N","60N"]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=11)
    ax.set_ylabel('Time')        

    plt.grid(alpha=0.15, linestyle='-', color="black")
    if savefig:
        os.makedirs("figures/hovmöller", exist_ok=True)
        plt.savefig(f"figures/hovmöller/{savelabel}", dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_crosscorr(data, lats, lat_labels, cmap, significance=False, n_effs=None, savefig=False):
    corr_matrix = np.corrcoef(data.values)  # (lat, lat)
    
    fig, ax  = plt.subplots(figsize=(8, 6))
    
    cf = ax.pcolormesh(lats, lats, corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(cf, ax=ax, label='Correlation Coefficient')
    
    # put significance mask when wanted:
    if significance:
        significance_mask = np.zeros_like(corr_matrix, dtype=bool)
        for i, lat_i in enumerate(lats):
            for j, lat_j in enumerate(lats):
                min_neff = min(n_effs[lat_i], n_effs[lat_j])
                r_crit   = critical_r(min_neff)
                significance_mask[i, j] = np.abs(corr_matrix[i, j]) >= r_crit

        masked = np.where(~significance_mask, 1, np.nan)
        ax.pcolormesh(lats, lats, masked, cmap=mcolors.ListedColormap(['white']), alpha=0.8)
        savename = "mht_correlation_latlat_significant.png"
    else:
        savename = "mht_correlation_latlat.png"
    plt.xticks(lats, lat_labels, rotation=45)
    plt.yticks(lats, lat_labels)
    plt.xlabel('Latitude (°N)')
    plt.ylabel('Latitude (°N)')
    plt.title('Cross-Correlation of MHT between Latitude Bands')
    if savefig:
        plt.savefig(f"figures/cross_corr/{savename}", dpi=300, bbox_inches='tight')
    plt.show()


def zero_lag_corr(ts1, ts2, lats, lat_labels, dimension="TIME", plot=False, savefig=False, polar_band=True):
    """
    Compute the zero-lag Pearson correlation coefficient between two time series and plot the result as a function of latitude.
    
    """
    corr = xr.corr(ts1, ts2,  dim=dimension) 
    if plot:
        fig, ax = plt.subplots()
        ax.plot(lats, corr, marker='o')

        ax.set_xticks(lats)
        ax.set_xticklabels(lat_labels[1:], rotation=45)
        ax.set_xlabel('Latitude (°N)')
        ax.set_ylabel('Correlation Coefficient')
        if ts1.LATITUDE.size == 1:
            # then its one selected latitude band
            lat_sel = ts1.LATITUDE.values.item()
            ax.axvline(lat_sel, color='gray', linestyle='--', alpha=0.5)
        
        # 2 latitde bands with high inner covariability 
        band1_min, band1_max = 16, 35      # 16N–35N
        band2_min, band2_max = -35, 5      # 35S–5N
        band3_min, band3_max = 55, 60      # 55-65N
        
        y_min, y_max = ax.get_ylim()

        # Rectangle for 16N–35N
        rect1 = mpatches.Rectangle(
            (band1_min, y_min),
            band1_max - band1_min,
            y_max - y_min,
            color='red',
            alpha=0.2,
            label='16°N–35°N'
        )

        # Rectangle for 35S–5N
        rect2 = mpatches.Rectangle(
            (band2_min, y_min),
            band2_max - band2_min,
            y_max - y_min,
            color='orange',
            alpha=0.1,
            label='35°S–5°N'
        )
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        
        if polar_band:
            
            rect3 = mpatches.Rectangle(
                (band3_min, y_min),
                band3_max - band3_min,
                y_max - y_min,
                color='magenta',
                alpha=0.1,
                label='55°N–60°N'
            )
            ax.add_patch(rect3)
        
        ax.legend(loc='best')
        
        ax.set_title(f"Correlation of {ts1.name} with {ts2.name}")

        if savefig:
            plt.savefig(f"figures/cross_corr/corr_{ts1.name}_{ts2.name}.png", dpi=300, bbox_inches='tight')
        plt.show()
    return corr


