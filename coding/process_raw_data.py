import numpy as np
import xarray as xr
import pandas as pd
import os
# # plottng
import cmocean as cmo
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.transforms import Bbox
# from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
# import matplotlib.ticker as mticker

# import cartopy.crs as ccrs  # Projections list
# import cartopy.feature as cfeature

# import scipy
from scipy import signal, stats


# from scipy.stats import pearsonr
# from amocatlas import read



#########################
# data handling& select #
#########################


def calc_ci(data, ci=90):
    if not "posterior_samples" in data.dims:
        return print("Can't compute CI without distribution.")
    else:
        lower = data.quantile((100 - ci) / 200, dim="posterior_samples", skipna=True)   
        upper = data.quantile(1 - (100 - ci) / 200, dim="posterior_samples", skipna=True)

        data_mean = data.mean(dim="posterior_samples", skipna=True)
        
        lower_err = data_mean - lower   # distance below mean
        upper_err = upper - data_mean   # distance above mean

        return np.array([lower_err, upper_err])    
  

def decimal_year_to_datetime(decimal_year):
    decimal_year = np.asarray(decimal_year)
    year = np.floor(decimal_year).astype(int)
    remainder = decimal_year - year

    start_of_year = pd.to_datetime(year.astype(str) + '-01-01')
    start_of_next_year = pd.to_datetime((year + 1).astype(str) + '-01-01')
    year_length = start_of_next_year - start_of_year

    return start_of_year + remainder * year_length


def detrend_dataset(ds):
    return xr.apply_ufunc(
        signal.detrend, ds, kwargs={"axis": -1},
        input_core_dims=[["time"]],
        output_core_dims=[["time"]]
    )
    
# def apply_filtfilt(data, b, a):
#     return signal.filtfilt(b, a, data, axis=-1)

# def butterworth_filter(ds, cutoff=1/1.5, fs=4):
#     b, a = signal.butter(
#         4,
#         cutoff,
#         btype="low",
#         fs=fs
#     )
#     return xr.apply_ufunc(
#         apply_filtfilt, ds, kwargs={'b': b, 'a': a},
#         input_core_dims=[['time']], output_core_dims=[['time']],
#         vectorize=True
#     )
    
def tukey_filter(data, window=4, alpha=0.5, min_periods=4):
    data_transpose = data.transpose("time", "lat")
    ds_pd = data_transpose.to_pandas()

    data_filtered = ds_pd.rolling(
        window=window,  #timedelta, 4 = 4 quarters = 1 year
        min_periods=min_periods,
        center=True,
        win_type="tukey",
    ).mean(alpha=alpha)
    
    return xr.DataArray(
        data_filtered.values,
        dims=["time", "lat"],
        coords={"time": data_filtered.index, "lat": data_filtered.columns},
    )
    
    
def lowpass_filter(data, method="tukey", window=4, alpha=0.5):
    if method == "rolling":
        data_transpose = data.transpose("time", "lat")
        ds_pd = data_transpose.to_pandas()

        data_smooth_pd = ds_pd.rolling(
            window=window,  #timedelta, 4 = 4 quarters = 1 year
            min_periods=4,
            center=True,
        ).mean()
        
        data_smooth_xr = xr.DataArray(
            data_smooth_pd.values,
            dims=["time", "lat"],
            coords={"time": data_smooth_pd.index, "lat": data_smooth_pd.columns},
        )
        # 4-quarter running mean filter
        data_smooth = data_smooth_xr.dropna(dim="time", how="any")
        
    # elif method == "butter":
    #     data_smooth = butterworth_filter(data)
    
    elif method == "tukey":
        data_smooth = tukey_filter(data, window, alpha).dropna(dim="time", how="any")
        data_smooth = data_smooth.transpose("lat", "time")

        
    return data_smooth  
    
def process_data(ds,
                detrended=False,
                method="tukey", 
                restore_mean=False,
                window=4,
                alpha=0.5,
    ):
    overall_mean = ds.mean(dim="time")  
    
    # no deseasonalize- all data is already deseasonalized!!
    # # 1. deseasonalize
    # seasonal_clim = ds.groupby("time.quarter").mean(dim="time")
    # anom = ds.groupby("time.quarter") - seasonal_clim

    # if anomalies:
    #     # return seasonal_clim
    #     return anom + overall_mean if restore_mean else anom

    # 2. detrend
    detrended_data = detrend_dataset(ds)
    ## if not restore mean then always anomalies
    if detrended:
        return detrended_data + overall_mean if restore_mean else detrended_data
    
    # 3. lowpass filter
    data_filtered = lowpass_filter(detrended_data, method=method, window=window, alpha=alpha)
    
    return data_filtered + overall_mean if restore_mean else data_filtered


def plot_filter_response(cutoff=1/1.5, fs=4, order=4, worN=2000,
                         its_period_range=(0.7, 3)):
    """
    Plot the frequency response of the Butterworth filter used in `butterworth_filter`,
    for both single-pass and zero-phase (filtfilt) application.
    """
    b, a = signal.butter(order, cutoff, btype="low", fs=fs)
    w, h = signal.freqz(b, a, worN=worN, fs=fs)

    H2_singlepass = np.abs(h) ** 2
    # h2 is the fravtion of power that survives at a given frequency f , that if not low pass filtered here
    # goes from 0 - 1, 0 fill blocked, 1 fully passed
    H2_filtfilt = H2_singlepass ** 2  # filtfilt = forward+backward -> response squared

    its_min_period, its_max_period = its_period_range
    its_freq_low = 1 / its_max_period   # longer period -> lower frequency
    its_freq_high = 1 / its_min_period  # shorter period -> higher frequency

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.loglog(w, H2_singlepass, label=f'Butterworth, single-pass (order {order})')
    # ax.loglog(w, H2_filtfilt, label=f'Butterworth, zero-phase / filtfilt (eff. order {2*order})')
    ax.axhline(0.5, color='gray', linestyle=':',alpha=0.7, label='half-power point')
    # cutoff
    ax.axvline(cutoff, color='black', linestyle='--', alpha=0.8,
                              label=f'cutoff = {cutoff:.3f} cyc/yr ({1/cutoff:.2g} yr period)')
    # ITS period band -> frequency band
    ax.axvline(its_freq_low, color='green', linestyle=':', alpha=0.8,
               label=f'ITS band: {its_max_period}yr -> {its_freq_low:.3f} cyc/yr')
    ax.axvline(its_freq_high, color='orange', linestyle=':', alpha=0.8,
               label=f'ITS band: {its_min_period}yr -> {its_freq_high:.3f} cyc/yr')
    ax.axvspan(its_freq_low, its_freq_high, color='green', alpha=0.1)

    ax.set_xlabel(f'frequency (cycles per year, fs={fs})')
    ax.set_ylabel(r'$|H(f)|^2$')
    ax.set_title(f'Butterworth low-pass response (cutoff={cutoff:.3g}, fs={fs})')
    ax.legend()
    plt.tight_layout()
    plt.show()
    return fig, ax

def MHT_selection(
        ds,
        statistics=False, 
        time_mean=False,
        anomalies=False, 
        ci=90, 
        drop65=True, 
        trend=False,
        printout=False
    ):
    """
    Select and process MHT data for the full latitude range.
    Returns mean, std, CI, anomalies, and time statistics.
    """
    def _name(da, label):
        da.name = label
        return da
    
    if drop65:
        ds = ds.where(ds.LATITUDE != 65, drop=True)

    lats = ds.LATITUDE.values
    lat_labels = [f"{np.abs(lat)}{'N' if lat > 0 else 'S'}" for lat in lats]
    MHT = ds.MHT
    
    if printout:
        print("Initial MHT dims: ", MHT.dims)
        nan_mask = np.isnan(MHT).any(dim=["lat", "posterior_samples"])
        print("Timesteps with NaN: ", MHT.TIME.where(nan_mask, drop=True).values)
    
    MHT = MHT.dropna(dim="TIME", how="any")

    statistics_results = {}
    if statistics:
        MHT_mean = MHT.mean(dim="posterior_samples", skipna=True)
        MHT_std = MHT.std(dim="posterior_samples",  skipna=True)
        statistics_results["mean"] = MHT_mean
        statistics_results["std"] = MHT_std
        
        if ci:
            statistics_results["ci"] = calc_ci(MHT, ci)
            
        if time_mean:
            mht_time_mean = MHT.mean(dim="TIME", skipna=True)
            mht_time_std = MHT.std(dim="TIME",  skipna=True)
            statistics_results["time_mean"] = _name(mht_time_mean, "MHT time mean")
            statistics_results["time_std"] = _name(mht_time_std,  "MHT time std")

    mht_anom = None
    if anomalies:
        mht_anom = MHT - MHT.mean(dim="TIME", skipna=True)
        mht_anom = _name(mht_anom, "MHT anomalies")

    if trend:
        pass

    if printout:
        target = mht_anom if anomalies else MHT
        print(f"Output shape: {target.shape}")
        print(f"Any NaNs: {np.isnan(target).any().item()}")

    return {
        "lats" : lats,
        "lat_labels" : lat_labels,
        "MHT" : MHT,
        "MHT_statistics" : statistics_results,
        "MHT_anom" : mht_anom,
    }


def MHT_lat_selection(
        MHT_dict,
        lat,
        ci=90,
        anomalies=False,
        time_mean=True,
        printout=False
    ):
    """
    Select a single latitude band from an already-processed MHT_dict
    (output of MHT_selection) and compute statistics for that band.
    """
    def _name(da, label):
        da.name = label
        return da

    lats = MHT_dict["lats"]
    lat_labels = MHT_dict["lat_labels"]
    MHT = MHT_dict["MHT"]
    mht_anom = MHT_dict["MHT_anom"]

    lat_idx = np.argmin(np.abs(lats - lat))
    label = lat_labels[lat_idx]

    MHT_lat = MHT.isel(lat=lat_idx)          # shape: (TIME, posterior_samples)
    MHT_lat = _name(MHT_lat, f"MHT lat {label}")

    stats = {}
    stats["mean"] = MHT_lat.mean(dim="posterior_samples", skipna=True)  # (TIME,)
    stats["std"] = MHT_lat.std(dim="posterior_samples",  skipna=True)

    if time_mean:
        stats["time_mean"] = MHT_lat.mean(dim="TIME", skipna=True)
        stats["time_std"] = MHT_lat.std(dim="TIME",  skipna=True)

    if ci:
        stats["ci"] = calc_ci(MHT_lat, ci)

    if anomalies:
        if mht_anom is None:
            raise ValueError("No anomalies found in MHT_dict. Run MHT_selection with anomalies=True first.")
        MHT_lat_anom = mht_anom.isel(lat=lat_idx)
        stats["anom"] = _name(MHT_lat_anom, f"MHT lat anom {label}")

    if printout:
        print(f"Selected latitude: {label} (index {lat_idx})")
        print(f"MHT_lat shape: {MHT_lat.shape}")

    return {
        "lat_label" : label,
        "MHT_lat" : MHT_lat,
        "MHT_lat_statistics" : stats,
    }
 