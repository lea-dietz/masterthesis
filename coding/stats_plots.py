
import numpy as np
import xarray as xr
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

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
    all_lats=None,
    labels=None,
    cmap=plt.cm.coolwarm,
    linestyles=None,
    label="specific",
    title=None,
    ylim=None,
    show_trend=False,
    timescale="years",
    lat_dim="lat",
    time_dim="time",
    savefig=False,
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
    labels : dict, optional
        Mapping lat -> legend label. If None, str(lat) is used.
    colors : list, optional
        Colors indexed to match `all_lats`. If None, auto-generated.
    label : str
        Used in the title/filename when show_trend=False (legacy v1 behavior).
    title : str, optional
        Overrides the auto-generated title.
    ylim : tuple, optional
        (ymin, ymax).
    show_trend : bool
        If True, fit and overlay a linear trend per latitude (dashed line, slope in legend).
    timescale : str
        "years" or "days" — controls slope units when show_trend=True.
    savefig : bool
        If True, save PNG to figures/timeseries/anomalies/.
    show : bool
        If True, call plt.show(). Set False if you want to keep customizing fig/ax.

    Returns
    -------
    fig, ax
    """
    
    if all_lats is None:
        raise ValueError("Need the list of all reference latitudes (all_lats).")

    lats = np.asarray(lats, dtype=float)
    all_lats = np.asarray(all_lats, dtype=float)
    colors = [cmap(i / max(len(all_lats) - 1, 1)) for i in range(len(all_lats))]    

    lat_to_idx = {l: i for i, l in enumerate(all_lats)}
    if not isinstance(data, dict):
        data = {"data": data}
    
    default_styles = ["-", "--", ":", "-."]
    if linestyles is None:
        linestyles = {name: default_styles[i % len(default_styles)]
                      for i, name in enumerate(data.keys())}

    fig, ax = plt.subplots(figsize=(12, 6))

    for name, da in data.items():
        zorder_item = 3
        ls = linestyles.get(name, "-")
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
            color = colors[idx]
            lat_label = labels[idx] if labels else str(lat)
            full_label = f"{lat_label} ({name})" if len(data) > 1 else lat_label
            ts = da.isel({lat_dim: idx})

            if show_trend:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_numeric, ts.values
                )
                trend = slope * time_numeric + intercept
                ax.plot(da[time_dim], ts.values, color=color, alpha=0.3, linestyle=ls)
                ax.plot(
                    da[time_dim], trend,
                    label=f"{full_label}, slope: {slope:.0e} PW/{slope_unit}",
                    color=color, linestyle=ls, zorder=5, linewidth=2,
                )
            else:
                ts.plot(ax=ax, label=full_label, color=color, linestyle=ls, linewidth=2, zorder=zorder_item)
        zorder_item += 1

    ax.set_title(f"{title} timeseries {label} latitude(s)")
    ax.set_xlabel("Year" if not show_trend else "Time")
    ax.set_ylabel("MHT anomaly (PW)")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if savefig:
        os.makedirs("figures/timeseries/anomalies/", exist_ok=True)
        plt.savefig(f"figures/timeseries/anomalies/calafat_timeseries_{label}.png",
                    dpi=300, bbox_inches='tight')

    plt.show()

    return fig, ax


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
        plot=False):

    N = data.sizes["time"]
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
        method = "0cross"

        its, max_lags = integral_time_scale(ts, del_t=del_t, method=method)

        N_eff = N * del_t / its
        n_effs[value] = N_eff
        its_dict[value] = its

        print(
            f"{coord_labels[idx]}: ITS = {its:.2f} {time_unit}, "
            f"N_eff = {N_eff:.1f}"
        )

        if plot:
            acf_calc_plot(
                ts,
                N_eff=12,
                plot=True,
                title=f"Autocorrelation at {coord_labels[idx]}"
            )

    r_crits = {value: critical_r(neff) for value, neff in n_effs.items()}
    
    return n_effs, r_crits, its_dict


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
                vmin=None, vmax=None, 
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
        if vmin is None and vmax is None:
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
    title='Cross-Correlation of MHT anomalies', 
    savefig=False, savename=None):
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

        sig_rows, sig_cols = np.where(significance_mask)   
        sig_x = lats[sig_cols]
        sig_y = lats[sig_rows]

        ax.scatter(sig_x, sig_y, marker='x', color='black', s=40, linewidths=1.2)
        
        if savename is None:
            savename = "mht_correlation_latlat_significant.png"
    else:
        if savename is None:
            savename = "mht_correlation_latlat.png"
            
    plt.xticks(lats, lat_labels, rotation=45)
    plt.yticks(lats, lat_labels)
    plt.xlabel('Latitude (°N)')
    plt.ylabel('Latitude (°N)')
    plt.title(title)
    if savefig:
        plt.savefig(f"figures/cross_corr/{savename}", dpi=300, bbox_inches='tight')
    plt.show()

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
    fig.colorbar(cf, ax=ax, label='Correlation Coefficient')

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
    ax.set_xlabel('Latitude (°N)')
    ax.set_ylabel('Latitude (°N)')
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

