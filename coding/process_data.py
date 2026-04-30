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





def MHT_selection(ds, average=True, anomalies=False, one_band_sel=False, lat=None, statistics=False, ci=95, drop65=True, printout=False):
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
    MHT_prob = ds.MHT
    
    if printout:
        print("Initial MHT dims: ", MHT_prob.dims)
        nan_mask = np.isnan(MHT_prob).any(dim=["lat", "posterior_samples"])
        print("Timesteps with NaN: ", MHT_prob.TIME.where(nan_mask, drop=True).values)# Average over posterior samples or keep all
        
    MHT = MHT_prob.mean(dim="posterior_samples", skipna=True) if average else MHT_prob
    MHT_prob = MHT_prob.dropna(dim="TIME", how="any")  # Drop NaN timesteps from the probability data as well
    
    # Drop timesteps with any NaN
    MHT = MHT.dropna(dim="TIME", how="any")
    # get mean and std
    MHt = _name(MHT, "MHT (all bands)")
    statistics_results = None
    if statistics:
        mht_mean = MHT.mean(dim="TIME", skipna=True)
        mht_std = MHT.std(dim="TIME", skipna=True)
        
        mht_mean = _name(mht_mean, "MHT mean")
        mht_std = _name(mht_std, "MHT std")
        
        statistics_results = {
            "mean": mht_mean,
            "std": mht_std
        }
        
        if ci:
            
            lower_bound = MHT_prob.quantile((100 - ci) / 200, dim="posterior_samples", skipna=True) # this gives 2.5 for 95% CI
            upper_bound = MHT_prob.quantile(1 - (100 - ci) / 200, dim="posterior_samples", skipna=True) # this gives 97.5 for 95% CI
            
            lower_bound = lower_bound.mean(dim="TIME", skipna=True)
            upper_bound = upper_bound.mean(dim="TIME", skipna=True)
            
            lower_bound = _name(lower_bound, f"MHT {ci}% CI lower")
            upper_bound = _name(upper_bound, f"MHT {ci}% CI upper")
            if printout:
                print(f"{ci}% CI: {(100 - ci) / 200*100}% - {100 - (100 - ci) / 200*100}%")
            statistics_results["ci"] = np.array([mht_mean - lower_bound, upper_bound - mht_mean])
        
    # Anomalies
    mht_anom = None
    if anomalies:
        mht_mean = MHT.mean(dim="TIME", skipna=True)
        mht_anom = MHT - mht_mean
        mht_anom = _name(mht_anom, "MHT anomalies")

    # Latitude selection (applied to anomalies if available, else raw MHT)
    MHT_lat = None
    if one_band_sel:
        if lat is None:
            raise ValueError("Please provide a latitude value for one_band_sel.")
        lat_idx = np.argmin(np.abs(lats - lat))
        MHT_lat = MHT.isel(lat=lat_idx) 
        MHT_lat_anom = mht_anom.isel(lat=lat_idx) if anomalies else None
        MHT_lat = _name(MHT_lat, f"MHT lat {lat_labels[lat_idx]}")
        if anomalies:
            MHT_lat_anom = _name(MHT_lat_anom, f"MHT lat anom {lat_labels[lat_idx]}")
        if printout:
            print(f"Selected latitude: {lats[lat_idx]}°N")

    
    if printout:
        target = mht_anom if anomalies else MHT
        print(f"Output shape: {target.shape}")
        print(f"Any NaNs: {np.isnan(target).any().item()}")
        lat_sel = MHT_lat
        print(f"Latitude selection shape: {lat_sel.shape if lat_sel is not None else 'N/A'}")

    return {
        "lats" : lats,
        "MHT": MHT,
        "MHT_anom": mht_anom,
        "MHT_lat": MHT_lat,
        "MHT_lat_anom": MHT_lat_anom if anomalies else None,
        "MHT_statistics": statistics_results
    }
    