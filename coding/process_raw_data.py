import numpy as np
import xarray as xr
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
import scipy.stats as stats

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


def lowpass_filter(data, method="4q-rollingmean"):
    # 4-quarter running mean filter
    data_smooth = data.rolling(TIME=4, center=True, min_periods=4).mean()
    data_smooth = data_smooth.dropna(dim="TIME", how="any")

    return data_smooth    

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
 