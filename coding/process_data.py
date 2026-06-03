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
# import scipy.stats as stats

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
    
    
def plot_crosscorr(data, lats, lat_labels, cmap, savefig=False):
    corr_matrix = np.corrcoef(data.values)  # (lat, lat)
    fig = plt.figure(figsize=(8, 6))
    plt.pcolormesh(lats, lats, corr_matrix, cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(lats, lat_labels, rotation=45)
    plt.yticks(lats, lat_labels)
    plt.xlabel('Latitude (°N)')
    plt.ylabel('Latitude (°N)')
    plt.title('Cross-Correlation of MHT between Latitude Bands')
    if savefig:
        plt.savefig("figures/cross_corr/mht_correlation_latlat.png", dpi=300, bbox_inches='tight')
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


