import numpy as np
import xarray as xr
import os

import pycwt #  as wavelet
# from pycwt.helpers import find

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


import cmocean as cmo
import palettable.colorbrewer as cb 


###########################
#### Coherence analysis ###
###########################

def wct_analysis(
    y1, y2, 
    time, 
    sig=True, 
    normalize=True,
    ):
    
    N = len(time) # all have the same time
    dt = 0.25
    s0 = 2 * dt
    dj = 1 / 10

    J = int(1/dj * np.log2(N * dt / s0)) 
    
    WCT, arrowsWCT, COI, freq, sig = pycwt.wct(
        y1=y1,
        y2=y2,
        dt=dt,
        dj=dj,
        s0=s0,
        J=J,
        sig=sig,
        normalize=normalize, # default : true, normalized with STD!
        cache=False
    )
    
    period = 1 / freq

    return WCT, arrowsWCT, COI, period, sig
    
    
def wct_plot(
    WCT, arrowsWCT, COI, period, sig,
    time, 
    cmap=cb.sequential.YlGnBu_9.mpl_colormap,
    vmin=0,
    vmax=1,
    title="26°N and 11°S",
    analysis_type="Coherence", # can be coherence or power ( for individual CWT plot)
    savefig=False,
    savelabel="cwt_26_11S",
    ):
        
    fig, ax  = plt.subplots(figsize=(8.5, 6))

    if vmin is not None and vmax is not None:
        cf = ax.pcolormesh(time, period, WCT, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    else:
        cf = ax.pcolormesh(time, period, WCT, cmap=cmap, shading='auto')

    fig.colorbar(cf, label=analysis_type)

    ax.plot(time, COI, 'k--', linewidth=1.5, label="COI")
    ax.fill_between(time, COI, period.max(), color='white', alpha=0.6, zorder=5)

    # significance contour (sig=1 marks 95% significance boundary)
    # 
    if sig is not None:
        sig_2d = np.ones_like(WCT) * sig[:, np.newaxis]  # broadcast to (scales, time)
        # shading the NOT significant part:
        sig_ratio = WCT / sig_2d
        # Shade non-significant regions (ratio < 1) 
        sig_shading = ax.contourf(time, period, sig_ratio, levels=[0, 1], colors='None', hatches=['//'], label="Region of no significance")
        sig_shading.set_edgecolor('white')
        sig_shading.set_linewidth(0)
            
        # ax.contourf(time, period, sig_ratio, levels=[0, 1], colors='white', alpha=0.4)
        ax.contour(time, period, sig_ratio, [1], colors='white', linewidths=1.5, label="Sig. 95% (AR(1))")
    
    # phase arrows (subsample so arrows aren't too dense)
    skip_t = 3
    skip_p = 3
    # sliciing the time and period and arrows , ::step
    # quiver doesnt take angle directly but x and y direction
    # x = cos(angle), y ( sin(angle) )
    if arrowsWCT is not None:
        ax.quiver(time[::skip_t], period[::skip_p],
                np.cos(arrowsWCT)[::skip_p, ::skip_t], np.sin(arrowsWCT)[::skip_p, ::skip_t],
                scale=30, width=0.003)

    ax.set_yscale('log', base=2)
    # change ticks to be not 2^0 but 1
    yticks = 2 ** np.arange(np.floor(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(y)}' if y >= 1 else f'{y:.1f}' for y in yticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())  # removes minor tick clutter

    ax.set_ylim(float(0.5), float(8.0))
    
    ax.invert_yaxis()  # short periods on top, like the book figure

    
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))        # tick only at each year
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))  # show only the year number
        
    ax.set_xlabel('Time')
    ax.set_ylabel('Period (years)')
    ax.set_title(f"{title}")
    
    ## handles for the legend
    if arrowsWCT is not None and sig is not None:
        legend_elements = [
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Cone of influence'),
            Patch(facecolor='none', edgecolor='gray', hatch='//', label='Not significant (95%)'),
            # Line2D([0], [0], color='gray', linewidth=1.5, label='Sig. 95%'),
            Line2D([0], [0], color='black', linestyle="None", marker=r"$\longrightarrow$", markersize=15, label='Phase'),
        ]
    if arrowsWCT is None:
        legend_elements = [
            Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Cone of influence'),
        ]

    legend = ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        frameon=True,
        fontsize=14,
        # facecolor="gray",
        # edgec olor="black",
    )
    # for text in legend.get_texts():
    #     text.set_color('white')
    
    # plt.grid("grey", linestyle="--", alpha=0.5)

    if savefig:
        os.makedirs("figures/wct/", exist_ok=True)
        plt.savefig(f"figures/wct/{savelabel}.png", dpi=300, bbox_inches='tight')
    plt.show()


################################
#### Wavelet transform only  ###
################################


def cwt_plot(
    power, COI, period,
    time,
    timeseries,
    cmap=cb.sequential.YlGnBu_9.mpl_colormap,
    title="16°N CWT",
    timeseries_label="Index",
    savefig=False,
    savelabel="cwt_plot",
):
    fig = plt.figure(figsize=(6.5, 7.5))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[30, 1],     # main plot vs colorbar column
        height_ratios=[3, 1],     # wavelet panel taller than timeseries panel
        hspace=0.15,              # whitespace between the two panels
        wspace=0.04,
    )
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, 0], sharex=ax)

    tick_levels = [1/8, 1/4, 1/2, 1, 2, 4, 8]

    # Many fine, log-spaced levels for a smooth-looking fill
    fine_levels = np.logspace(np.log10(tick_levels[0]), np.log10(tick_levels[-1]), 100)

    cf = ax.contourf(time, period, power, levels=fine_levels,
                norm=LogNorm(vmin=tick_levels[0], vmax=tick_levels[-1]),
                cmap=cmap,
                )

    cbar = fig.colorbar(cf, cax=cax, label='Power', ticks=tick_levels)
    cbar.ax.set_yticklabels([f'{l:.3g}' if l < 1 else f'{int(l)}' for l in tick_levels])
    
    ax.plot(time, COI, 'k--', linewidth=1.5, label="COI")
    ax.fill_between(time, COI, period.max(), color='white', alpha=0.6, zorder=5)

    ax.set_yscale('log', base=2)
    yticks = 2 ** np.arange(np.floor(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(yticks)
    ax.set_yticklabels([f'{int(y)}' if y >= 1 else f'{y:.1f}' for y in yticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())

    ax.set_ylim(float(0.5), float(8.0))
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_ylabel('Period (years)')
    ax.set_title(f"{title}")

    plt.setp(ax.get_xticklabels(), visible=False)  # top panel: no x labels, bottom panel carries them
    ax.set_xlabel("")

    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Cone of influence'),
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=len(legend_elements),
        frameon=True,
        fontsize=14,
    )

    # timeseries subfigure ( following idea from :A. Grinsted1, J. C. Moore1, and S. Jevrejeva2 2004) 
    
    ax_ts.plot(time, timeseries, 'k-', linewidth=1)

    ax_ts.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    ax_ts.set_ylabel(timeseries_label)
    ax_ts.set_xlabel('Time')
    # ax_ts.set_ylim(-0.53, 0.53)
    ax_ts.xaxis.set_major_locator(mdates.YearLocator(base=2))
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if savefig:
        os.makedirs("figures/wct/", exist_ok=True)
        plt.savefig(f"figures/wct/{savelabel}.png", dpi=300, bbox_inches='tight')
    plt.show()
