import numpy as np
# import xarray as xr
# # plottng
# import cmocean as cmo
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.transforms import Bbox
# from matplotlib.colors import ListedColormap
# import matplotlib.patches as mpatches
# import matplotlib.ticker as mticker

# import cartopy.crs as ccrs  # Projections list
# import cartopy.feature as cfeature

# import scipy
# import scipy.stats as stats
# from scipy.stats import pearsonr
# from amocatlas import read



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


def MHT_selection(
        ds,
        statistics=False, 
        time_mean=False,
        anomalies=False, 
        one_band_sel=False, 
        lat=None, 
        ci=95, 
        drop65=True, 
        trend=False,
        printout=False
    ):
    """
    Select and process MHT data
    get anomalies and statistics,
    select a specific latitude band
    
    """
    def _name(da, label):
        da.name = label
        return da
    
    if drop65:
        ds = ds.where(ds.LATITUDE != 65, drop=True)
    # define labels for the latitdues
    lats = ds.LATITUDE.values
    n_lats = len(lats)
    lat_labels = [f"{np.abs(lat)}{'N' if lat > 0 else 'S'}" for lat in lats]
    MHT = ds.MHT
    
    if printout:
        print("Initial MHT dims: ", MHT.dims)
        nan_mask = np.isnan(MHT).any(dim=["lat", "posterior_samples"])
        print("Timesteps with NaN: ", MHT.TIME.where(nan_mask, drop=True).values)# Average over posterior samples or keep all
    
    
    # Drop timesteps with any NaN
    MHT = MHT.dropna(dim="TIME", how="any")
    # get mean and std
    statistics_results = {}
    if statistics:
        # get mean over post. samples  
        MHT_mean = MHT.mean(dim="posterior_samples", skipna=True)
        MHT_std = MHT.std(dim="posterior_samples", skipna=True)
        
        statistics_results["mean"] = MHT_mean
        statistics_results["std"] = MHT_std
        
        if ci:
            ci_array = calc_ci(MHT, 90,)
            statistics_results["ci"] = ci_array
            
        if time_mean:
            mht_time_mean = MHT.mean(dim="TIME", skipna=True)
            mht_time_std = MHT.std(dim="TIME", skipna=True)
            
            mht_time_mean = _name(mht_time_mean, "MHT time mean")
            mht_time_std = _name(mht_time_std, "MHT time std")
            
            statistics_results["time_mean"] = mht_time_mean
            statistics_results["time_std"] = mht_time_std
        
        
    # Anomalies
    mht_anom = None
    if anomalies:
        mht_mean = MHT.mean(dim="TIME", skipna=True)
        mht_anom = MHT - mht_mean
        mht_anom = _name(mht_anom, "MHT anomalies")

    # Latitude selection (applied to anomalies if available, else raw MHT)
    statistics_results_lat = {}

    MHT_lat = None
    if one_band_sel:
        if lat is None:
            raise ValueError("Please provide a latitude value for one_band_sel.")
        
        lat_idx = np.argmin(np.abs(lats - lat))
        
        # Keep posterior samples for CI/distribution at this latitude
        MHT_lat_prob = MHT.isel(lat=lat_idx)   # shape: (TIME, posterior_samples)
        MHT_lat = MHT_lat_prob
        MHT_lat_mean      = MHT.isel(lat=lat_idx).mean(dim="posterior_samples", skipna=True) # shape: (TIME,) — mean over samples
        MHT_lat_std = MHT.isel(lat=lat_idx).std(dim="posterior_samples", skipna=True) 
        
        MHT_lat_anom = mht_anom.isel(lat=lat_idx) if anomalies else None

        MHT_lat      = _name(MHT_lat, f"MHT lat {lat_labels[lat_idx]}")

        statistics_results_lat["mean"] = MHT_lat_mean
        statistics_results_lat["std"] = MHT_lat_std
        
        # CI for this specific latitude
        if ci:
            lat_ci = calc_ci(MHT_lat_prob, 90)
            statistics_results_lat["ci"] = lat_ci
        if anomalies:
            MHT_lat_anom = _name(MHT_lat_anom, f"MHT lat anom {lat_labels[lat_idx]}")
            statistics_results_lat["anom"] = MHT_lat_anom

            
    if trend:
        # compute the trend for the dataset over time
        pass
    
    if printout:
        target = mht_anom if anomalies else MHT
        print(f"Output shape: {target.shape}")
        print(f"Any NaNs: {np.isnan(target).any().item()}")
        lat_sel = MHT_lat
        print(f"Latitude selection shape: {lat_sel.shape if lat_sel is not None else 'N/A'}")

    return {
        "lats"         : lats,
        "MHT"          : MHT, # no means just drop Nan
        "MHT_statistics": statistics_results,
        "MHT_anom"     : mht_anom,
        "MHT_lat"      : MHT_lat,
        "MHT_lat_statistics"   : statistics_results_lat if one_band_sel and ci else None, 
    }
    
    
