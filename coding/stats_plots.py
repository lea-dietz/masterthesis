
import numpy as np
import xarray as xr
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.patheffects as pe


import cmocean as cmo
import scipy.stats as stats
from scipy.stats import t as t_dist

import statsmodels.tsa.stattools as stattools
from statsmodels.tsa.stattools import ccf


#####
# get dim name
def get_lat_name(ds):
    for name in ("lat", "LATITUDE", "latitude", "LAT", "number_regions"):
        if name in ds.coords or name in ds.dims:
            return name
    raise ValueError(f"No latitude coordinate found. Available: {list(ds.coords)}")

## to change label name into float
def parse_lat_label(label):
    """'55°N' -> 55.0, '25°S' -> -25.0"""
    if "N" in label:
        label = label.replace("°N", "")
    elif "S" in label:
        label = label.replace("°S", "")
        label = "-" + label 
    return float(label)

#########################
#### plot timeseries ####
#########################
def plot_timeseries(
    data,
    lats,
    var="MHT",
    all_lats=None,
    labels=None,
    cmap=plt.cm.Spectral_r,
    alphas=None,
    linestyles=None,
    label="specific",
    anomalies=True,
    title=None,
    ylim=None,
    xlim=None,
    ylabel=None,
    xbase=1,
    show_trend=False,
    grid=False,
    annotate_x=None,
    loc_legend="upper center",
    timescale="years",
    lat_dim="lat",
    time_dim="time",
    savefig=False,
    savelabel="mht",
    savefolder="timeseries/anomalies/",
    ):
    """
    Plot MHT (or anomaly) time series per latitude, with optional linear trend overlay.
    also possible for several timeseries at once
    Parameters
    ----------
    data : xarray.DataArray or dict with multiple ts like "raw" or "processed"
        Data with a LATITUDE dim (selectable via .sel) or a 'lat' dim matching `lats`.
        
    lats : array-like
        Latitudes to plot.
    all_lats : array-like, optional
        Full latitude list used to look up colors by index. If None, colors are
        generated automatically from a coolwarm colormap over `lats`.
    show_trend : bool
        If True, fit and overlay a linear trend per latitude (dashed line, slope in legend).
    timescale : str
        "years" or "days" — controls slope units when show_trend=True.
   
    Returns
    -------
    fig, ax
    """
    
    if not isinstance(data, dict):
        data = {"data": data}
        
    if all_lats is None:
        raise ValueError("Need the list of all reference latitudes (all_lats).")

    lats = np.asarray(lats, dtype=float)
    all_lats = np.asarray(all_lats, dtype=float)
    
    if isinstance(cmap, (list, tuple)):
        colors = cmap
        dataset_colors = {name: cmap[i % len(cmap)] for i, name in enumerate(data.keys())}
    else:
        colors = [cmap(i / max(len(all_lats) - 1, 1)) for i in range(len(all_lats))]
        dataset_colors = {name: cmap(i / max(len(data) - 1, 1)) for i, name in enumerate(data.keys())}
    
    if alphas is None:
        dataset_alphas = {name: 1.0 for name in data.keys()}
    elif isinstance(alphas, dict):
        dataset_alphas = {name: alphas.get(name, 1.0) for name in data.keys()}
    else:  # list or tuple, cycled like colors
        dataset_alphas = {name: alphas[i % len(alphas)] for i, name in enumerate(data.keys())}

    lat_to_idx = {l: i for i, l in enumerate(all_lats)}

    default_styles = ["-", "-.", "-", "-"]
    if linestyles is None:
        linestyles = {name: default_styles[i % len(default_styles)]
                      for i, name in enumerate(data.keys())}

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, da in data.items():
        zorder_item = 3
        ls = linestyles.get(name, "-")
        alpha = dataset_alphas.get(name, 1.0)  # NEW: pull this dataset's alpha

        if show_trend:
            if timescale == "years":
                time_numeric = (da[time_dim] - da[time_dim][0]) / np.timedelta64(1, 'D') / 365.25
                slope_unit = "yr"
            elif timescale == "days":
                time_numeric = (da[time_dim] - da[time_dim][0]) / np.timedelta64(1, 'D')
                slope_unit = "day"
            else:
                raise ValueError("timescale must be 'years' or 'days'")

        for lat in lats:
            idx = lat_to_idx[lat]
            if len(lats) == 1:
                color = dataset_colors[name]
            else:
                color = colors[idx]

            lat_label = labels[idx] if labels else str(lat)
            full_label = name if len(lats) == 1 else (f"{lat_label} ({name})" if len(data) > 1 else lat_label)
            ts = da.isel({lat_dim: idx})

            if show_trend:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_numeric, ts.values
                )
                trend = slope * time_numeric + intercept
                ax.plot(da[time_dim], ts.values, color=color, alpha=alpha * 0.3, linestyle=ls)
                ax.plot(
                    da[time_dim], trend,
                    label=f"{full_label}, slope: {slope:.0e} PW/{slope_unit}",
                    color=color, linestyle=ls, zorder=zorder_item, linewidth=3,
                    alpha=alpha
                )
            else:
                ts.plot(ax=ax, label=full_label, color=color, linestyle=ls,
                         linewidth=3, zorder=zorder_item, alpha=alpha)  # NEW: alpha added here
        zorder_item += 1

    ax.set_title(f"{title} {label}")
    ax = plt.gca()  # or whatever your axis variable is

    ax.xaxis.set_major_locator(mdates.YearLocator(base=xbase))        # tick only at each year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # show only the year number
    
    ax.set_xlabel("")
    
    if ylabel is None:
        if anomalies:
            ax.set_ylabel(f"{var} anomaly (PW)")
        else: 
            ax.set_ylabel(f"{var} (PW)")
    else:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
        
    if xlim is not None:
        ax.set_xlim(*xlim)
        
    if loc_legend is not None:
        ax.legend(
            loc=loc_legend,
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(data),      # one row, one entry per dataset
            frameon=True,
            # fontsize=10,
        )   
    # else assume its on the right next to the plot
    else:
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            ncol=1,
            frameon=True,
        )
    
    # ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if grid: 
        plt.grid(alpha=0.5, linestyle="--")
    if annotate_x is not None:
        ax.annotate(
            'phase shift',
            xy=(annotate_x, 0), xycoords=('data', 'data'),
            xytext=(annotate_x, ax.get_ylim()[0]*0.8),
            arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
            fontsize=10, color='red', ha='center'
        )
    if savefig:
        os.makedirs(f"figures/{savefolder}/", exist_ok=True)
        plt.savefig(f"figures/{savefolder}/{savelabel}_timeseries_{label}.png",
                    dpi=300, bbox_inches='tight')

    #plt.show()

    return fig, ax


#########################
# Statistics functions ##
#########################

def acf_calc_plot(data, N_eff=None, plot=True, title="Autocorrelation Function am Sample Size vs Lag"):
    if not N_eff:
        N_eff = len(data)
    acf_values = stattools.acf(data, nlags=N_eff-1, adjusted=False)
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
    acf_values = stattools.acf(data, nlags=len(data)-1, adjusted=False) 
 
    if method == "0cross":
        lag_0cross = np.where(np.diff(np.sign(acf_values)))[0][0] + 1 
        acf_values_0 = stattools.acf(data, nlags=lag_0cross, adjusted=False) 
        # print(acf_values[lag_0cross])
        max_lag = lag_0cross

    if method == "0threshold":
        ## all lags where sign changes: 
        lags_0cross = np.where(np.diff(np.sign(acf_values)))[0] + 1 
        first_0cross = lags_0cross[0]
        if first_0cross != 1:
            lag_0cross = first_0cross
        else:
            lag_0cross = first_0cross + 1
        
        # get acf values up to the neede lag 
        acf_values_0 = stattools.acf(data, nlags=lag_0cross, adjusted=False) 
        max_lag = lag_0cross

    if method == "plateau":
        # find the first lag where the acf drops below 1/e
        lag_plateau = np.where(acf_values < 1/np.e)[0][0]
        acf_values_0 = stattools.acf(data, nlags=lag_plateau) 
        max_lag = lag_plateau
        
    #  calculate its using the N_eff recevied depeding on the method   
    its = del_t  * sum(1 + 2*(max_lag-j)/max_lag*acf_values_0[j] for j in range(1, max_lag)) # for normalized acf
    # 1 is outside of sum for next equation
    # its = del_t  * (1 + sum( 2*(max_lag-j)/max_lag*acf_values_0[j] for j in range(1, max_lag-1))) # for normalized acf
    
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

   
def compute_n_eff_and_r_crit(
        data,
        coord_values, # lats
        coord_labels, # lats labels for mht
        dim,
        del_t,
        time_unit,
        plateau_values=None,
        method="0cross",
        plot=False,
        printout=False
        ):

    N = data.sizes["time"]
    if printout:
        print(f"Total number of samples: {N}")

    n_effs = {}
    its_dict = {}

    if plateau_values is None:
        plateau_values = []

    for idx, value in enumerate(coord_values):

        ts = data.isel({dim: idx})

        # if value in plateau_values:
        #     method = "plateau"
        # else:

        its, max_lags = integral_time_scale(ts, del_t=del_t, method=method)
        
        if plot:
            acf_calc_plot(
                ts,
                N_eff=12,
                plot=True,
                title=f"Autocorrelation at {coord_labels[idx]}"
            )
            
        if printout:
           print(its)
        
        
        N_eff = N * del_t / its
        n_effs[value] = N_eff
        its_dict[value] = its
        if printout:
            print(
                f"{coord_labels[idx]}: ITS = {its:.2f} {time_unit}, "
                f"N_eff = {N_eff:.1f}"
            )


    r_crits = {value: critical_r(neff) for value, neff in n_effs.items()}
    
    return n_effs, r_crits, its_dict


def significance_mask(data, ref_lats, all_lats, all_lags, nlags, r_crits):
    corr_matrix = np.full((len(all_lats), len(all_lags)), np.nan)

    for ref_lat in ref_lats:

        ref_idx = np.argmin(np.abs(all_lats - ref_lat))
        ts_ref = data.isel(lat=ref_idx).values

        corr_matrix = np.full((len(all_lats), len(all_lags)), np.nan)
        significance_mask = np.zeros((len(all_lats), len(all_lags)), dtype=bool)

        r_ref = r_crits[ref_lat]

        for i, lat in enumerate(all_lats):

            ts = data.isel(lat=i).values

            pos_corr, _ = ccf(ts_ref, ts, nlags=nlags, alpha=0.05)
            neg_corr, _ = ccf(ts, ts_ref, nlags=nlags, alpha=0.05)

            full_corr = np.concatenate([neg_corr[1:][::-1], pos_corr])
            corr_matrix[i, :] = full_corr

            r_lat = r_crits[lat]

            # use the smaller effective sample size (larger critical r)
            r_crit = max(r_ref, r_lat)
            print(f"for ref lat: {ref_lat} and {lat}: r crit = {r_crit:.2f}")

            significance_mask[i, :] = np.abs(full_corr) >= r_crit
            

def between_block_significance(
        st_data, sa_data,
        periods,
        del_t, time_unit,
        method="0cross",
        alpha=0.05,
        plot=False
    ):
        """r_crit
        Correlation between ST and SA block-mean series per period,
        with significance based on each series' own effective sample size (autocorrelation-corrected).
        """
        rows = {}
        for period_name, (t0, t1) in periods.items():
            st_p = st_data.sel(time=slice(t0, t1))
            sa_p = sa_data.sel(time=slice(t0, t1))

            N = st_p.sizes["time"]  # same length for both, assuming aligned time axes

            its_st, _ = integral_time_scale(st_p, del_t=del_t, method=method)
            its_sa, _ = integral_time_scale(sa_p, del_t=del_t, method=method)

            n_eff_st = N * del_t / its_st
            n_eff_sa = N * del_t / its_sa
            n_eff_min = min(n_eff_st, n_eff_sa)   # conservative: less autocorrelated series sets the bound

            r_crit = critical_r(n_eff_min, alpha=alpha)
            r = float(np.corrcoef(st_p.values, sa_p.values)[0, 1])

            rows[period_name] = {
                "corr": r,
                "n_eff_ST": n_eff_st,
                "n_eff_SA": n_eff_sa,
                "n_eff_used": n_eff_min,
                "r_crit": r_crit,
                "significant": bool(abs(r) >= r_crit),
            }
            if plot:
                acf_calc_plot(
                    st_p,
                    N_eff=10,
                    plot=True,
                    title="ST (35-16°N) ACF and Sample Size"
                )
                acf_calc_plot(
                    sa_p,
                    N_eff=10,
                    plot=True,
                    title="SA (5°N-35°S) ACF and Sample Size"
                )
        return pd.DataFrame(rows).T

    
    

def significance_mask(data, ref_lats, lats, all_lags, nlags, r_crits):
    corr_matrix = np.full((len(lats), len(all_lags)), np.nan)

    for ref_lat in ref_lats:

        ref_idx = np.argmin(np.abs(lats - ref_lat))
        ts_ref = data.isel(lat=ref_idx).values

        corr_matrix = np.full((len(lats), len(all_lags)), np.nan)
        significance_mask = np.zeros((len(lats), len(all_lags)), dtype=bool)

        r_ref = r_crits[ref_lat]

        for i, lat in enumerate(lats):

            ts = data.isel(lat=i).values

            pos_corr, _ = ccf(ts_ref, ts, nlags=nlags, alpha=0.05)
            neg_corr, _ = ccf(ts, ts_ref, nlags=nlags, alpha=0.05)

            full_corr = np.concatenate([neg_corr[1:][::-1], pos_corr])
            corr_matrix[i, :] = full_corr

            r_lat = r_crits[lat]

            # use the smaller effective sample size (larger critical r)
            r_crit = max(r_ref, r_lat)
            print(f"for ref lat: {ref_lat} and {lat}: r crit = {r_crit:.2f}")

            significance_mask[i, :] = np.abs(full_corr) >= r_crit
        
    return significance_mask
            
#########################
# Plotting   functions ##
#########################

   
def plot_hovmöller(data,
                lats, lat_labels, 
                data_label="MHT",
                anomalies=False, 
                cmap="bwr",
                vlimits=None, 
                savefig=False
    ):
    """
    Plot the MHT (can be anomalies or raw) as a function of time and latitude using a colormap.
    """
    time_name = "time" if "time" in data.dims else "TIME"
    
    if isinstance(data, dict):
        MHT = data["MHT_statistics"]["mean"] # this is mean over posterior samples!
    else:
        # asume its the mht mean over post samples
        MHT = data
    # for anomalies
    # mht_all_mean = MHT_dict["MHT_statistics"]["time_mean"].mean(dim="posterior_samples") # this is time mean and over posterior samples mean

    fig, ax = plt.subplots(figsize=(16, 7))

    if anomalies:
    #     # cmap = "bwr"
    #     # Anomaly = value - mean
    #     anom = MHT - mht_all_mean#.mean(dim="lat") # get overall mean over all latitudes!
        if vlimits is not None:
            vmin, vmax = vlimits
        else:
            vmin = np.nanmin(data.values)
            vmax = np.nanmax(data.values)
            limit = max(abs(vmin), abs(vmax))
            vmin = -limit #- 0.5
            vmax = limit #+ 0.5
        mesh = ax.pcolormesh(MHT[time_name], lats ,data.values, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, label=f'{data_label} anomalies (PW)', extend="both")
        plt.title(f'{data_label} Anomalies')
        savelabel = f"{data_label}_time_lat_anomalies.png"

    else:
        if vlimits is not None:
            vmin, vmax = vlimits
        else:
            vmin = np.nanmin(MHT.values)
            vmax = np.nanmax(MHT.values)
            limit = max(abs(vmin), abs(vmax))
            vmin = -limit 
            vmax = limit 
    
        mesh = ax.pcolormesh(MHT[time_name], lats, MHT.values, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(mesh, label=f'{data_label} (PW)', extend="both")
        plt.title(f'{data_label}')
        savelabel = f"{data_label}_time_lat.png"

    ytick_labels = [label.replace(".0", "") for label in lat_labels]

    if np.all(np.diff(lats) < 0):      # decreasing, e.g. 60 -> -35
        ax.set_yticks(lats[::-1])
        ax.set_yticklabels(ytick_labels[::-1])
    else:                              # increasing, e.g. 0 -> 10 (region indices, 0 = north)
        ax.set_yticks(lats)
        ax.set_yticklabels(ytick_labels)   # keep original order — matches data row order
        ax.invert_yaxis() 
        
    ax.set_xlabel('Time')   
    ax.set_ylabel("Latitude")     

    if savefig:
        os.makedirs("figures/hovmöller", exist_ok=True)
        plt.savefig(f"figures/hovmöller/{savelabel}", dpi=300, bbox_inches='tight')
    plt.show()

    
    
def plot_crosscorr(
        data, 
        lats, lat_labels,
        cmap, 
        significance=False,
        n_effs=None,
        r_crits=None,
        title='Cross-Correlation of MHT anomalies',
        subtitle=None, 
        cbar_orientation='vertical', cbar_location='left',
        savefig=False, savename=None,
        ax=None,
        show=True,
        show_cbar=True,
        sharey=True,
    ):
    
    corr_matrix = np.corrcoef(data.values)  # (lat, lat)
    significance_mask = None
    
    if ax is None:
        fig, ax  = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    cf = ax.pcolormesh(lats, lats, corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    if show_cbar:
        if cbar_orientation == 'horizontal':
            cb = plt.colorbar(cf, ax=ax, label='Correlation Coefficient', orientation=cbar_orientation, location=cbar_location)
        else:
            cb = plt.colorbar(cf, ax=ax, label='Correlation Coefficient')
        cb.set_ticks([-1, -0.5, 0, 0.5, 1])
        cb.set_ticklabels(['-1', '-0.5', '0', '+0.5', '+1'])

    # put significance mask if wanted:
    if significance:
        significance_mask = np.zeros_like(corr_matrix, dtype=bool)
        for i, lat_i in enumerate(lats):
            for j, lat_j in enumerate(lats):
                min_neff = min(n_effs[lat_i], n_effs[lat_j])                
                if r_crits is None:
                    r_crit   = critical_r(min_neff)
                else:
                    r_crit = min(r_crits[lat_i], r_crits[lat_j])
                       
                significance_mask[i, j] = np.abs(corr_matrix[i, j]) >= r_crit

        sig_rows, sig_cols = np.where(significance_mask)   
        sig_x = lats[sig_cols]
        sig_y = lats[sig_rows]

        ax.scatter(sig_x, sig_y, marker='x', color='black', s=40, linewidths=1.2)
        
        if savename is None:
            savename = "mht_correlation_latlat_significant.png"
    else:
        if savename is None:
            savename = "mht_correlation_latlat.png"
            
    ax.set_xticks(lats)
    ax.set_xticklabels(lat_labels, rotation=45)
    ax.set_yticks(lats)
    ax.set_yticklabels(lat_labels)
    if not sharey:
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Latitude')
        
    if ax is None:
        plt.title(title)
    else:
        ax.set_title(subtitle)
    
    ax.set_aspect('equal')

    if savefig:
        plt.savefig(f"figures/cross_corr/{savename}", dpi=300, bbox_inches='tight')
    
    if show and ax is None:
        plt.show()
    
    return corr_matrix, significance_mask, cf

def plot_crosscorr_gif(
    data,
    lats, lat_labels,
    cmap,
    ax=None,
    significance=False,
    n_effs=None,
    title='Cross-Correlation of MHT anomalies',
    savefig=False, savename=None):

    corr_matrix = np.corrcoef(data.values)  # (lat, lat)

    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    cf = ax.pcolormesh(lats, lats, corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    cb = fig.colorbar(cf, ax=ax, label='Correlation Coefficient')
    cb.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb.set_ticklabels(['-1', '-0.5', '0', '+0.5', '+1'])

    if significance:
        significance_mask = np.zeros_like(corr_matrix, dtype=bool)
        for i, lat_i in enumerate(lats):
            for j, lat_j in enumerate(lats):
                min_neff = min(n_effs[lat_i], n_effs[lat_j])
                r_crit = critical_r(min_neff)
                significance_mask[i, j] = np.abs(corr_matrix[i, j]) >= r_crit

        sig_rows, sig_cols = np.where(significance_mask)  
        sig_x = lats[sig_cols]
        sig_y = lats[sig_rows]

        ax.scatter(sig_x, sig_y, marker='x', color='black', s=40, linewidths=1.2)

    ax.set_xticks(lats)
    ax.set_xticklabels(lat_labels, rotation=45)
    ax.set_yticks(lats)
    ax.set_yticklabels(lat_labels)
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)

    if owns_fig:
        if savefig:
            if savename is None:
                savename = "mht_correlation_latlat_significant.png" if significance else "mht_correlation_latlat.png"
            fig.savefig(f"figures/cross_corr/{savename}", dpi=300, bbox_inches='tight')
        plt.show()

    return ax

def plot_crosscorr_hf_mht(
    corr_matrix,          # (n_regions, n_mht_lats), already computed
    region_info,    # dict of idx, labels, regions 
    lats, lat_labels,      # MHT latitudes/labels
    cmap,
    significance=False,
    n_effs_hf=None,        # dict: region_bounds -> N_eff (or region label -> N_eff)
    n_effs_mht=None,       # dict: lat -> N_eff
    title='Cross-Correlation of HF regions vs MHT',
    savefig=False, savename=None):
    
    region_y = region_info["idx"]
    region_labels = region_info["labels"]
    region_bounds = region_info["bounds"]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cf = ax.pcolormesh(lats, region_y, corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(cf, ax=ax, label='Correlation Coefficient')

    if significance:
        significance_mask = np.zeros_like(corr_matrix, dtype=bool)
        for i in region_y:
            for j, lat_j in enumerate(lats):
                min_neff = min(n_effs_hf[i], n_effs_mht[lat_j])
                r_crit = critical_r(min_neff)
                significance_mask[i, j] = np.abs(corr_matrix[i, j]) >= r_crit

        sig_rows, sig_cols = np.where(significance_mask)   # row = region idx, col = lat idx
        sig_x = lats[sig_cols]
        sig_y = region_y[sig_rows]

        ax.scatter(sig_x, sig_y, marker='x', color='black', s=40, linewidths=1.2)
        
        if savename is None:
            savename = "hf_mht_correlation_significant.png"
    else:
        if savename is None:
            savename = "hf_mht_correlation.png"

    ax.set_xlabel('MHT Latitude (°N)')
    ax.set_xticks(lats)
    ax.set_xticklabels(lat_labels, rotation=45)
    
    ax.set_ylabel('HF Region')
    ax.set_yticks(region_y)
    ax.set_yticklabels(region_labels)
    ax.invert_yaxis() # invert y axis since regions are from 0 = north to 10 = south always
    ax.set_title(title)

    if savefig:
        os.makedirs("figures/cross_corr", exist_ok=True)
        plt.savefig(f"figures/cross_corr/{savename}", dpi=300, bbox_inches='tight')
    plt.show()
    
def zero_lag_corr(ts1, ts2, lats, lat_labels, dimension="TIME", plot=False, savefig=False, polar_band=True, title=None):
    """
    Compute the zero-lag Pearson correlation coefficient between two time series and plot the result as a function of latitude.
    
    """
    corr = xr.corr(ts1, ts2,  dim=dimension) 
    if plot:
        fig, ax = plt.subplots()
        ax.plot(lats, corr, marker='o')

        ax.set_xticks(lats)
        ax.set_xticklabels(lat_labels, rotation=45)
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Correlation Coefficient')
        
        lat_name = get_lat_name(ts1)
        if ts1[lat_name].size == 1:
            lat_sel = ts1[lat_name].values.item()
            ax.axvline(lat_sel, color='gray', linestyle='--', alpha=0.5)

            lat_sel = int(str(lat_sel).replace(".0",""))
            if lat_sel > 0:
                lat_label = f"{lat_sel}°N"
            else:
                lat_sel =  np.abs(lat_sel)
                lat_label = f"{lat_sel}°S"

        # 2 latitde bands with high inner covariability 
        band1_min, band1_max = 16, 35      # 16N–35N
        band2_min, band2_max = -35, 5      # 35S–5N
        band3_min, band3_max = 55, 60      # 55-65N
        
        ax.set_ylim(0, 1)

        y_min, y_max = ax.get_ylim()

        # Rectangle for 16N–35N
        rect1 = mpatches.Rectangle(
            (band1_min, y_min),
            band1_max - band1_min,
            y_max - y_min,
            color='red',
            alpha=0.2,
            #label='16°N–35°N'
        )

        # Rectangle for 35S–5N
        rect2 = mpatches.Rectangle(
            (band2_min, y_min),
            band2_max - band2_min,
            y_max - y_min,
            color='orange',
            alpha=0.1,
            #label='35°S–5°N'
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
                #label='55°N–60°N'
            )
            ax.add_patch(rect3)
        ax.legend(loc='best')
        if title is None:
            ax.set_title(f"0-lag correlation for {lat_label}")
        else:
            ax.set_title(title)
            
        if savefig:
            os.makedirs(f"figures/0lag/", exist_ok=True)
            plt.savefig(f"figures/0lag/corr_{lat_label}.png", dpi=300, bbox_inches='tight')
        plt.show()
    return corr

def zero_lag_corr_regions(
        ts1, ts2, 
        dimension="time",
        lat_idx=None, 
        lat_labels=None,
        plot=False, 
        savefig=False, 
        title=None,
        xlabel="Heat Flux Region Index",
    ):
    """
    Compute the zero-lag Pearson correlation coefficient between two time series,
    broadcasting over any remaining dimension (e.g. number_regions).
    """
    corr = xr.corr(ts1, ts2, dim=dimension)

    if plot:
        fig, ax = plt.subplots()
        
        # x-axis: whatever the remaining dim in corr is (e.g. number_regions)
        x_dim = corr.dims[0]
        x_vals = corr[x_dim].values
        
        ax.plot(x_vals, corr.values, marker='o')
        ax.set_xticks(x_vals)  
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Correlation Coefficient')
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(title if title else f"0-lag correlation: MHT {lat_labels[lat_idx]} vs Heat Flux by region")
        if np.all(np.diff(x_vals) < 0):      # decreasing, e.g. 60 -> -35
            ax.set_xticks(x_vals[::-1])
            # ax.set_yticklabels(ytick_labels[::-1])
        else:                              # increasing, e.g. 0 -> 10 (region indices, 0 = north)
            ax.set_xticks(x_vals)
            ax.set_xticklabels(x_vals)   # keep original order — matches data row order
            ax.invert_xaxis() 
            
        if savefig:
            os.makedirs("figures/0lag", exist_ok=True)
            plt.savefig("figures/0lag/corr_mht_hf_regions.png", dpi=300, bbox_inches='tight')
        plt.show()

    return corr


def zero_lag_corr_hf_mht(
        hf_region, mht,
        region_bounds, # like (16, 26) the boundaries for hf region  
        dimension="time",
        lats=None, lat_labels=None,
        plot=False, 
        savefig=False, 
        title=None,
    ):
    """
    Compute the zero-lag Pearson correlation coefficient between two time series.
    """
    corr = xr.corr(hf_region, mht, dim=dimension)

    if plot:
        fig, ax = plt.subplots()
        
        x_vals = lats
        ax.plot(x_vals, corr.values, marker='o')
        
        # shade the latitude span this HF region actually covers
        lower, higher = min(region_bounds), max(region_bounds)
        ax.axvspan(lower, higher, color='orange', alpha=0.2)

        ax.set_xticks(lats if lats is not None else x_vals)
        if lat_labels is not None:
            ax.set_xticklabels(lat_labels, rotation=45)
        ax.set_xlabel("MHT Latitude (°N)")
        ax.set_ylabel("Correlation Coefficient")
        ax.set_ylim(-1, 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        ax.set_title(title or f"HF {region_bounds} vs MHT (all latitudes)", y=1.02)

        if savefig:
            os.makedirs("figures/0lag", exist_ok=True)
            plt.savefig(f"figures/0lag/corr_hf_region_{lower}_{higher}_vs_mht.png", dpi=300, bbox_inches='tight')
        plt.show()

    return corr



def plot_shifted_timeseries(ds, ref_lat, lats, all_lats, lat_labels, colors, timedelta=6, label="specific", savefig=False, ylim=None):
    # ylim should be a tuple
    time_name = "time" if "time" in ds.dims else "TIME"

    fig, ax = plt.subplots(figsize=(12, 5))
    
    #reference time series:
    x = ds.sel(LATITUDE=ref_lat, method='nearest')
    x.plot(ax=ax, label=f"{lat_labels[ref_lat]} (reference)", color='black', marker='o', linestyle='--', zorder=5)
    for lat in lats:
        y= ds.sel(LATITUDE=lat, method='nearest').shift(TIME=timedelta)
        y.plot(ax=ax, label=lat_labels[lat], color=colors[np.where(all_lats == lat)[0][0]], marker='o', linestyle='-')
        # CORR BETWEEN REFERENCE AND SHIFTED TIME SERIES
        x_aligned, y_aligned = xr.align(x,y, join="inner")
        corr = xr.corr(x_aligned, y_aligned, dim=time_name)

        print(f"Correlation between {lat_labels[ref_lat]} and {lat_labels[lat]} with a shift of {timedelta / 4} years: {corr.values:.2f}")
            
    ax.set_title(f"MHT anomalies shifted by {timedelta / 4} years")
    ax.set_ylabel("MHT anomaly (PW)")
    ax.set_xlabel("Year")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axhline(y= 0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if savefig:
        os.makedirs(f"figures/timeseries/anomalies/", exist_ok=True)
        plt.savefig(f"figures/timeseries/anomalies/calafat_timeseries_shifted_{timedelta}_{label}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    

def lagged_std(ds, ref_lat1, ref_lat2, timedelta=4, max_lag=None, normalize=False):
    """
    Compute std of (x - y_shifted) for a range of lags.
    
    Parameters
    ----------
    x       : reference time series (1D array)
    y       : time series to shift (1D array)
    max_lag : maximum lag to test (default: len/4)
    normalize : normalize both series to unit variance first
    
    Returns
    -------
    lags : array of lag values (positive = y shifted forward)
    stds : std of (x - y_shifted) at each lag
    """
    # ds is already in anomalize usually
    x = ds.sel(LATITUDE=ref_lat1, method='nearest')#.plot(ax=ax, label=f"{lat_labels[ref_lat]} (reference)", color='black', marker='o', linestyle='--', zorder=5)
    y = ds.sel(LATITUDE=ref_lat2, method="nearest").shift(TIME=timedelta)
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
   
    # Optionally normalize
    if normalize:
        x = x / np.std(x)
        y = y / np.std(y)
    
    n = len(x)
    if max_lag is None:
        max_lag = n // 4  # sensible default
    
    lags = np.arange(-max_lag, max_lag + 1)
    stds = np.full(len(lags), np.nan)
    
    for i, lag in enumerate(lags):
        if lag == 0:
            diff = x - y
        elif lag > 0:
            # y is shifted forward: compare x[lag:] with y[:-lag]
            diff = x[lag:] - y[:-lag]
        else:
            # y is shifted backward: compare x[:lag] with y[-lag:]
            diff = x[:lag] - y[-lag:]
        
        stds[i] = np.std(diff)
    
    return x, y, lags, stds


def _plot_one_ref(
    ax, ref_idx, ds_ref, ds_target, n_effs_ref, n_effs_target,
    region_labels, all_lags, nlags, lag_units, significance, cmap,
    boundary_labels=None,   # NEW: list of n_target+1 latitude boundary labels, e.g. ["65°N","60°N",...,"35°S"]
):
    ts_ref = ds_ref.isel(number_regions=ref_idx).values
    n_eff_ref = n_effs_ref[ref_idx]

    n_target = ds_target.sizes["number_regions"]
    y_pos = np.arange(n_target)

    corr_matrix = np.full((n_target, len(all_lags)), np.nan)
    significance_mask = np.zeros((n_target, len(all_lags)), dtype=bool)

    for i in range(n_target):
        ts = ds_target.isel(number_regions=i).values

        pos_corr, _ = ccf(ts_ref, ts, nlags=nlags, alpha=0.05)
        neg_corr, _ = ccf(ts, ts_ref, nlags=nlags, alpha=0.05)
        lag0_corr = np.corrcoef(ts_ref, ts)[0, 1]

        full_corr = np.concatenate([neg_corr[::-1], [lag0_corr], pos_corr])
        corr_matrix[i, :] = full_corr

        n_eff_target = n_effs_target[i]
        min_n_eff = min(n_eff_ref, n_eff_target)
        r_crit = critical_r(min_n_eff)
        significance_mask[i, :] = np.abs(full_corr) >= r_crit

    lag_edges = np.arange(-nlags - 0.5, nlags + 1.5, 1)  # edges between lag columns
    y_edges = np.arange(-0.5, n_target + 0.5, 1)          # edges between region rows

    cf = ax.pcolormesh(-lag_edges, y_edges, corr_matrix, cmap=cmap, vmin=-1, vmax=1, shading='flat')

    if significance:
        sig_rows, sig_cols = np.where(significance_mask)
        sig_x = -all_lags[sig_cols]
        sig_y = y_pos[sig_rows]
        ax.scatter(sig_x, sig_y, marker='x', color='black', s=15, linewidths=0.8)

    ax.axhspan(ref_idx - 0.5, ref_idx + 0.5, facecolor='none',
               edgecolor='red', linewidth=1.5, zorder=5)
    ax.axvline(0, color='white', linewidth=0.8, linestyle='--')

    # --- boundary-based y ticks instead of region-center ticks ---
    if boundary_labels is not None:
        boundary_pos = np.arange(-0.5, n_target + 0.5, 1)  # n_target+1 edge positions
        ax.set_yticks(boundary_pos)
        ax.set_yticklabels(boundary_labels)
    else:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(region_labels)

    ax.invert_yaxis()

    if lag_units == "months":
        xticks = np.arange(all_lags[0], all_lags[-1] + 1, max(2, nlags // 6))
        xticklabels = xticks.astype(str)
    else:
        lag_step = 4
        xticks = np.arange(-nlags, nlags + 1, lag_step)
        xticklabels = (xticks // lag_step).astype(str)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ref_label = region_labels[ref_idx] if ref_idx < len(region_labels) else str(ref_idx)

    ax.annotate(f'← ref lags', xy=(0.20, -0.1), xycoords='axes fraction',
                ha='center', va='top')
    ax.annotate(f'ref leads →', xy=(0.80, -0.1), xycoords='axes fraction',
                ha='center', va='top')

    ax.set_ylabel('HF band')
    ax.set_title(f'HTC ref band {ref_label}')

    return cf

def plot_lead_lag_regions(
    ds_ref, ds_target, n_effs_ref, n_effs_target,
    region_labels=None, nlags=24, time_unit="months",
    significance=True, cmap=None, savefig=False,
    savename_prefix="lag_corr_regions", boundary_labels=None
):
    n_ref = ds_ref.sizes["number_regions"]
    n_target = ds_target.sizes["number_regions"]

    if region_labels is None:
        region_labels = [str(i) for i in range(n_target)]

    all_lags = np.arange(-nlags, nlags + 1)
    lag_units = "months" if time_unit == "months" else "years"

    n_cols = 4
    n_rows = int(np.ceil(n_ref / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows),
                              sharey=True, sharex=True)
    axes = np.atleast_1d(axes).flatten()

    cf = None
    for ref_idx in range(n_ref):
        cf = _plot_one_ref(
            axes[ref_idx], ref_idx, ds_ref, ds_target,
            n_effs_ref, n_effs_target, region_labels,
            all_lags, nlags, lag_units, significance, cmap,
            boundary_labels,
        )

    for j in range(n_ref, len(axes)):
        axes[j].set_visible(False)

    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='Correlation')

    ref_handle = mlines.Line2D([], [], color='red', linewidth=1.5, label='Reference region')
    eq_handle = mlines.Line2D([], [], color='white', linewidth=2, linestyle='--', label='Lag 0')
    fig.legend(handles=[ref_handle, eq_handle], loc='lower right', fontsize=12)

    fig.supxlabel(f'Lag ({lag_units})', fontsize=14)
    plt.tight_layout()

    savename = f"{savename_prefix}_significance.png" if significance else f"{savename_prefix}.png"
    if savefig:
        os.makedirs("figures/lead_lag", exist_ok=True)
        plt.savefig(f'figures/lead_lag/{savename}', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, axes


def plot_lead_lag_single_region(
    ds_ref, ds_target, ref_idx, n_effs_ref, n_effs_target,
    region_labels=None, nlags=24, time_unit="months",
    significance=True, cmap=None, savefig=False,
    savename_prefix="lag_corr_single", 
    boundary_labels=None
):
    n_target = ds_target.sizes["number_regions"]

    if region_labels is None:
        region_labels = [str(i) for i in range(n_target)]

    all_lags = np.arange(-nlags, nlags + 1)
    lag_units = "months" if time_unit == "months" else "years"

    fig, ax = plt.subplots(figsize=(8, 6))

    cf = _plot_one_ref(
        ax, ref_idx, ds_ref, ds_target,
        n_effs_ref, n_effs_target, region_labels,
        all_lags, nlags, lag_units, significance, cmap, 
        boundary_labels,
    )

    fig.colorbar(cf, ax=ax, label='Correlation')

    ref_handle = mlines.Line2D([], [], color='red', linewidth=1.5, label='Reference band')
    eq_handle = mlines.Line2D([], [], color='white', linewidth=2, linestyle='--', label='Lag 0')
    ax.legend(handles=[ref_handle, eq_handle], loc='lower right', fontsize=10)

    ax.set_xlabel(f'Lag ({lag_units})')
    plt.tight_layout()

    savename = f"{savename_prefix}_region{ref_idx}_significance.png" if significance else f"{savename_prefix}_region{ref_idx}.png"
    if savefig:
        os.makedirs("figures/lead_lag", exist_ok=True)
        plt.savefig(f'figures/lead_lag/{savename}', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, ax




def block_coherence_index(
        mht,
        r_crits,
        lat_idx,
        lats,
        periods, # dict: period name -> (start time, end time)
        lat_dim="lat",
        time_dim="time"):
    """
    Calcutes the mean coherence index for a block of latitudes over specified periods.
    The coherence index is defined as the mean of the pairwise correlation coefficients between latitudes in the block, 
    after applying a significance threshold based on critical r values.
    Also only use upper triangle of correlation matrix (without diagonal) to avoid double counting.
    
    """
    # select block from lat idx (0 == 60°N)
    block = mht.isel({lat_dim: lat_idx})
    block_lats = lats[lat_idx]
    
    results = {}
    
    for period_name, (t_start, t_end) in periods.items():
        
        period_data = block.sel({time_dim: slice(t_start, t_end)})
        
        vals = period_data.transpose(time_dim, lat_dim).values
        
        corr_mat = np.corrcoef(vals.T)
        # shape of corr_mat is (n lats in block , n lats in block)
        # correlation between each pair lats in block in that time window
        # Apply significance threshold
        for i in range(len(block_lats)):
            for j in range(i+1, len(block_lats)):
                r_crit = max(r_crits[block_lats[i]], r_crits[block_lats[j]])
                if np.abs(corr_mat[i, j]) < r_crit:
                    corr_mat[i, j] = np.nan
                    corr_mat[j, i] = np.nan

        # triu indices gives upper triangles of matrix, k=1 is offset 1 (without diagonal!)
        iu = np.triu_indices(len(block_lats), k=1)
        pair_vals = corr_mat[iu]
        
        mean_coherence = np.nanmean(pair_vals) if np.sum(~np.isnan(pair_vals)) >= 2 else np.nan
        n_valid = np.sum(~np.isnan(pair_vals))

        results[period_name] = {
            "mean_coherence": mean_coherence,
            "n_valid_pairs": n_valid,
            "n_total_pairs": len(pair_vals)
        }
        
        
    return pd.DataFrame(results).T

def cross_block_coherence_index(
        mht,
        r_crits,
        lat_idx_a,   # e.g. NA lat indices
        lat_idx_b,   # e.g. SA lat indices
        lats,
        periods,
        lat_dim="lat",
        time_dim="time"):
    """
    Computes mean within-block coherence for block A and block B separately,
    plus mean cross-block coherence between A and B, over specified periods.
    Uses the same significance thresholding (r_crits) as block_coherence_index.

    Within-block: uses upper triangle (k=1) of the pairwise correlation matrix,
    excluding the diagonal, to avoid double counting (same-latitude pairs).
    Cross-block: uses the full A x B matrix since A and B are distinct latitude sets.
    """
    block_a = mht.isel({lat_dim: lat_idx_a})
    block_b = mht.isel({lat_dim: lat_idx_b})
    lats_a = lats[lat_idx_a]
    lats_b = lats[lat_idx_b]

    def within_block_coherence(vals, block_lats):
        # vals: (time, lat) array for this block
        corr_mat = np.corrcoef(vals.T)
        n = len(block_lats)
        for i in range(n):
            for j in range(i + 1, n):
                r_crit = max(r_crits[block_lats[i]], r_crits[block_lats[j]])
                if np.abs(corr_mat[i, j]) < r_crit:
                    corr_mat[i, j] = np.nan
                    corr_mat[j, i] = np.nan
        iu = np.triu_indices(n, k=1)
        pair_vals = corr_mat[iu]
        mean_coh = np.nanmean(pair_vals) if np.sum(~np.isnan(pair_vals)) >= 2 else np.nan
        n_valid = np.sum(~np.isnan(pair_vals))
        return mean_coh, n_valid, len(pair_vals)

    def cross_block_coherence(a_vals, b_vals, lats_a, lats_b):
        n_a, n_b = len(lats_a), len(lats_b)
        cross_mat = np.full((n_a, n_b), np.nan)
        for i in range(n_a):
            for j in range(n_b):
                r = np.corrcoef(a_vals[:, i], b_vals[:, j])[0, 1]
                r_crit = max(r_crits[lats_a[i]], r_crits[lats_b[j]])
                cross_mat[i, j] = r if np.abs(r) >= r_crit else np.nan
        pair_vals = cross_mat.flatten()
        mean_coh = np.nanmean(pair_vals) if np.sum(~np.isnan(pair_vals)) >= 2 else np.nan
        n_valid = np.sum(~np.isnan(pair_vals))
        return mean_coh, n_valid, len(pair_vals)

    results = {}
    for period_name, (t_start, t_end) in periods.items():
        a_data = block_a.sel({time_dim: slice(t_start, t_end)}).transpose(time_dim, lat_dim).values
        b_data = block_b.sel({time_dim: slice(t_start, t_end)}).transpose(time_dim, lat_dim).values

        within_a_coh, within_a_n, within_a_total = within_block_coherence(a_data, lats_a)
        within_b_coh, within_b_n, within_b_total = within_block_coherence(b_data, lats_b)
        cross_coh, cross_n, cross_total = cross_block_coherence(a_data, b_data, lats_a, lats_b)

        results[period_name] = {
            "within_A_coherence": within_a_coh,
            "within_A_n_valid": within_a_n,
            "within_A_n_total": within_a_total,
            "within_B_coherence": within_b_coh,
            "within_B_n_valid": within_b_n,
            "within_B_n_total": within_b_total,
            "cross_AB_coherence": cross_coh,
            "cross_AB_n_valid": cross_n,
            "cross_AB_n_total": cross_total,
        }
    return pd.DataFrame(results).T


def block_mean_amplitude(
        mht,
        lat_idx,
        lats,
        periods,
        lat_dim="lat",
        time_dim="time"):
    """
    Computes mean amplitude (std of each latitude's own time series) for a block
    of latitudes, averaged over period. Amplitude here = temporal std per latitude,
    i.e. how much that latitude's signal swings, NOT spread across latitudes.
    """
    block = mht.isel({lat_dim: lat_idx})
    block_lats = lats[lat_idx]

    results = {}
    for period_name, (t_start, t_end) in periods.items():
        period_data = block.sel({time_dim: slice(t_start, t_end)})

        # std per latitude (over time), then mean across latitudes in the block
        per_lat_std = period_data.std(dim=time_dim)  # shape: (lat,)
        mean_amp = float(per_lat_std.mean())
        std_of_amp = float(per_lat_std.std())  # how much amplitude itself varies within block

        ### mean of block and then std of that mean
        block_mean_ts = period_data.mean(dim=lat_dim)   # average across lat FIRST
        amp_of_block_curve = float(block_mean_ts.std(dim=time_dim))
        
        results[period_name] = {
            "mean_amplitude": mean_amp,
            "std_across_lats": std_of_amp,
            "mean_block_amplitude": amp_of_block_curve,
        }
    return pd.DataFrame(results).T


#####################
###### LEAD LAG #####
#####################

def plot_all_leadlag(
    data, lats, lat_labels, all_lats, all_lat_labels,
    nlags, all_lags, lag_units='years',
    cmap=None, significance=False, n_effs=None,
    ncols=4,
    savefig=False, savename=None
):
    lats = np.asarray(lats)
    all_lats = np.asarray(all_lats)

    n_ref = len(lats)
    n_target = len(all_lats)

    nrows = int(np.ceil(n_ref / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7*ncols, 7*nrows),
        sharey=True
    )

    if n_ref == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
            
            
    for ax, ref_lat, ref_label in zip(axes, lats, lat_labels):
        ref_idx = np.argmin(np.abs(all_lats - ref_lat))
        
        ts_ref  = data.isel(lat=ref_idx).values
        n_eff_ref = n_effs[ref_lat] if significance else None

        corr_matrix = np.full((n_target, len(all_lags)), np.nan)
        significance_mask = np.zeros((n_target, len(all_lags)), dtype=bool)

        for i, (lat, lat_label) in enumerate(zip(all_lats, all_lat_labels)):
            ts = data.isel(lat=i).values
            
            pos_corr, _ = ccf(ts_ref, ts, nlags=nlags, alpha=0.05)   
            neg_corr, _ = ccf(ts, ts_ref, nlags=nlags, alpha=0.05)  
            ### ccf gives also lag 0
            full_corr = np.concatenate([neg_corr[1:][::-1], pos_corr])
            
            corr_matrix[i, :] = full_corr
            if significance:
                min_n_eff = min(n_eff_ref, n_effs[lat])
                r_crit = critical_r(min_n_eff)
                significance_mask[i, :] = np.abs(full_corr) >= r_crit

            # if ref_lat == 45:
            #     print(f"Reference latitude: {ref_lat}°N")
            #     print(all_lags)
            #     print(np.round(full_corr, 2))
            #     print(all_lags[np.argmax(np.abs(full_corr))])
                
            max_corr_idx = np.argmax(full_corr)
            max_lag = all_lags[max_corr_idx]
            
            ax.scatter(-max_lag, lat, color='black', s=70, marker='o', edgecolor='white', zorder=5)
            
        cf = ax.contourf(
            -all_lags, all_lats, corr_matrix,
            levels=np.linspace(-1, 1, 41),
            cmap=cmap,
        )
        if significance:
            ax.contourf(-all_lags, all_lats,
                        np.where(~significance_mask, 1, np.nan),
                        levels=[0.5, 1.5],
                        colors='white', alpha=0.7)
            savename = "lag_corr_significance.png"
        else:
            savename = f'lag_corr.png'
        ax.axhline(ref_lat, color='red', linewidth=1.5)
        ax.axvline(0, color='white', linewidth=1.5, linestyle='--')
        ax.axhline(0, color='white', linewidth=1.5)

        ax.set_yticks(all_lats)
        ax.set_yticklabels(all_lat_labels, fontsize=20)
        
        if lag_units == "months":
            xticks = np.arange(all_lags[0], all_lags[-1] + 1, 2)
            xticklabels = xticks.astype(str)
        else:
            # assume its years!!
            lag_step = 4
            xticks = np.arange(-nlags, nlags + 1, lag_step)
            xticklabels = (xticks // lag_step).astype(str)
            
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize=20)

        ax.annotate(f'← {ref_label} lags', xy=(0.25, +0.05), xycoords='axes fraction',
                    ha='center', va='top', fontsize=20)
        ax.annotate(f'{ref_label} leads →', xy=(0.75, +0.05), xycoords='axes fraction',
                    ha='center', va='top', fontsize=20)

        ax.set_ylabel('Latitude', fontsize=20)
        ax.set_title(f'{ref_label}')


    # hide any unused axes
    for ax in axes[n_ref:]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='Correlation')

    ref_lat_handle = mlines.Line2D([], [], color='red', linewidth=1.5, label='Reference Latitude')
    # eq_handle = mlines.Line2D([], [], color='white', linewidth=2, linestyle='-',
    #         label="Equator / Lag 0")
    lag0_handle = mlines.Line2D([], [], 
        color='white',
        linewidth=1.5, 
        linestyle='--',
        label='Lag 0',
        path_effects=[pe.withStroke(linewidth=4, foreground='black')]
    )
    significance_handle = mpatches.Patch(
        facecolor='white',
        edgecolor='black',
        linewidth=1.5,
        alpha=1,
        label='Not significant'
    )
    
    max_lag_hanlde = mlines.Line2D([], [], color='black', marker='o', markersize=8, label='Max correlation lag', markeredgecolor='white')
    
    fig.supxlabel(
        'Lag in years' if lag_units == 'years' else 'Lag in months',
        fontsize=18,
        y=0.08
    )

    fig.legend(
        handles=[ref_lat_handle, lag0_handle, significance_handle, max_lag_hanlde],
        loc='lower center',
        bbox_to_anchor=(0.6, 0.0),
        ncol=4,
        frameon=True,
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    

    if savefig:
        if savename is None:
            savename = "lag_corr_significance.png" if significance else "lag_corr.png"
        os.makedirs("figures/lead_lag", exist_ok=True)
        plt.savefig(f'figures/lead_lag/{savename}', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, axes

def plot_max_lag(
    data, lats, lat_labels,
    nlags, all_lags, lag_units='years',
    cmap_lag=cmo.cm.curl, cmap_corr=None,
    significance=False, n_effs=None,
    title='Lag of Maximum Correlation',
    savefig=False, savename=None
):
    n = len(lats)
    lag_matrix     = np.full((n, n), np.nan)
    maxcorr_matrix = np.full((n, n), np.nan)
    significance_mask = np.zeros((n, n), dtype=bool)

    for i, lat_i in enumerate(lats):
        # print(lat_i)
        ts_i = data.isel(lat=i).values
        for j, lat_j in enumerate(lats):
            # print(f"  {lat_j}")
            ts_j = data.isel(lat=j).values

            ##### ccf givves lag 0 so no need to add it manually
            # pos_corr, _ = ccf(ts_i, ts_j, nlags=nlags, alpha=0.05)
            # neg_corr, _ = ccf(ts_j, ts_i, nlags=nlags, alpha=0.05)
            # lag0_corr = np.corrcoef(ts_i,  ts_j)[0, 1]
            # full_corr = np.concatenate([neg_corr[::-1], [lag0_corr], pos_corr])
            pos_corr, _ = ccf(ts_i, ts_j, nlags=nlags, alpha=0.05)   
            neg_corr, _ = ccf(ts_j, ts_i, nlags=nlags, alpha=0.05)   

            full_corr = np.concatenate([neg_corr[1:][::-1], pos_corr])
            
            ## accounting for the fact that lag -1 and +1 are often 0.99999
            abs_corr = np.abs(full_corr)
            max_val = np.max(abs_corr)
            tol = 1e-5  # or something a bit looser, e.g. 1e-4, if you have float noise
            candidate_idx = np.where(abs_corr >= max_val - tol)[0]
            # among tied candidates, pick the one with lag closest to zero
            idx_max = candidate_idx[np.argmin(np.abs(np.array(all_lags)[candidate_idx]))]            
            
            lag_matrix[i, j] = all_lags[idx_max]
            maxcorr_matrix[i, j] = full_corr[idx_max]
            # if i == 4:
            #     if lat_i == lat_j:
                    
            #         for idx in range(len(full_corr)):
            #             print(f"Lag {all_lags[idx]}: Correlation {full_corr[idx]}")
            #         # print(f"Max correlation at same latitude {lat_i}°N: {full_corr[idx_max]:.3f} at lag {all_lags[idx_max]} ({lag_units})")
            
            if significance:
                min_neff = min(n_effs[lat_i], n_effs[lat_j])
                r_crit   = critical_r(min_neff)
                significance_mask[i, j] = np.abs(full_corr[idx_max]) >= r_crit

    if cmap_corr is None:
        cmap_corr = cmap_lag

    fig, axes = plt.subplots(1, 2, figsize=(18, 8.5), sharey=True)

    # max lag plot
    ax = axes[0]
    
    cf1 = ax.pcolormesh(lats, lats, lag_matrix, cmap=cmap_lag,
                         vmin=-nlags, vmax=nlags)
    
    cb1 = plt.colorbar(cf1, ax=ax, label=f'Lag of max corr ({lag_units})')

    lag_step = 4  # 4 quarters = 1 year
    ticks = np.arange(-nlags, nlags + 1, lag_step)

    cb1.set_ticks(ticks)
    cb1.set_ticklabels((ticks / lag_step).astype(int))

    cb1.set_label("Lag of max correlation (years)")

    if significance:
        sig_rows, sig_cols = np.where(significance_mask)
        ax.scatter(lats[sig_cols], lats[sig_rows], marker='x',
                    color='black', s=60, linewidths=1.5)

    ax.set_xticks(lats); ax.set_xticklabels(lat_labels, rotation=45)
    ax.set_yticks(lats); ax.set_yticklabels(lat_labels)
    ax.set_xlabel('Latitude'); ax.set_ylabel('Latitude')
    ax.set_title('Lag of maximum abs. Correlation')

    # correlation at that lag plot
    ax = axes[1]
    cf2 = ax.pcolormesh(lats, lats, maxcorr_matrix, cmap=cmap_corr,
                         vmin=-1, vmax=1)
    cb2 = plt.colorbar(cf2, ax=ax, label='Correlation Coefficient')
    cb2.set_ticks([-1, -0.5, 0, 0.5, 1])
    cb2.set_ticklabels(['-1', '-0.5', '0', '+0.5', '+1'])

    if significance:
        sig_rows, sig_cols = np.where(significance_mask)
        ax.scatter(lats[sig_cols], lats[sig_rows], marker='x',
                    color='black', s=60, linewidths=1.5)

    ax.set_xticks(lats); ax.set_xticklabels(lat_labels, rotation=45)
    ax.set_xlabel('Latitude')
    ax.set_title('Correlation at that Lag')

    # fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if savefig:
        if savename is None:
            savename = "mht_lag_of_max_corr.png"
        os.makedirs("figures/max_lag", exist_ok=True)
        plt.savefig(f"figures/max_lag/{savename}", dpi=300, bbox_inches='tight')
    plt.show()

    return lag_matrix, maxcorr_matrix, significance_mask



def block_leadlag(
    st_data, sa_data,
    periods,
    nlags,
    all_lags,
    time_dim="time",
    n_eff_st=None, n_eff_sa=None,   # dict per period, for significance
    plot=True
):
    """
    Cross-correlation between ST and SA block-mean anomaly series,
    per period. Returns max |corr|, the lag at which it occurs, and
    whether ST leads or lags SA at that point.

    Convention: positive lag => SA leads ST (SA's past predicts ST's future)
                negative lag => ST leads SA
    (matches ccf(x, y) = corr(x_t, y_{t+lag}); we build the symmetric
    version the same way your plot_all_leadlag does)
    """
    rows = {}
    curves = {}

    for period_name, (t_start, t_end) in periods.items():
        st_timeperiod = st_data.sel({time_dim: slice(t_start, t_end)}).values
        sa_timeperiod = sa_data.sel({time_dim: slice(t_start, t_end)}).values

        pos_corr, _ = ccf(st_timeperiod, sa_timeperiod, nlags=nlags, alpha=0.05)
        neg_corr, _ = ccf(sa_timeperiod, st_timeperiod, nlags=nlags, alpha=0.05)
        full_corr = np.concatenate([neg_corr[1:][::-1], pos_corr])

        max_idx = np.argmax(np.abs(full_corr))
        max_lag = all_lags[max_idx]
        max_corr = full_corr[max_idx]

        row = {
            "max_abs_corr": max_corr,
            "lag_at_max": max_lag,
            "leader": "SA leads" if max_lag > 0 else ("ST leads" if max_lag < 0 else "in phase"),
        }

        if n_eff_st is not None and n_eff_sa is not None:
            min_n_eff = min(n_eff_st, n_eff_sa)
            r_crit = critical_r(min_n_eff)   # reuse your existing functio
            print(f"minimal n eff for {period_name}: ", min_n_eff)
            row["r_crit"] = r_crit
            row["significant"] = bool(abs(max_corr) >= r_crit)

        rows[period_name] = row
        curves[period_name] = (all_lags, full_corr)

    result_df = pd.DataFrame(rows).T

    if plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        for period_name, (all_lags, corr) in curves.items():
            ax.plot(all_lags, corr, marker='o', label=period_name)
        ax.axhline(0, color='gray', lw=0.8)
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel("Lag (SA relative to ST)")
        ax.set_ylabel("Cross-correlation")
        ax.legend()
        ax.set_title("ST–SA lead/lag by period")
        plt.tight_layout()
        plt.show()

    return result_df, curves