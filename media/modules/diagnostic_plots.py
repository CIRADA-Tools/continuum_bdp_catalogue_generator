import re
import pandas as pd

# Need to trap matplotlib UserWarning...
# notes: https://docs.astropy.org/en/stable/warnings.html
import warnings
warnings.simplefilter("error",UserWarning)

import numpy as np, matplotlib.pyplot as plt
from astropy.coordinates import Angle
from astropy import units as u
from matplotlib.colors import LogNorm


###functions to make key diagnostic plots for component cat


################################################################################
################################################################################
###define parameters
#plt.interactive(True) ###set to false if just runnning script to create figures in directory


####example bins for analysing parameter space
nnbins = np.logspace (-1, 3.5, 500) ##nn separations in arcsec
majbin = np.logspace(0, 2, 500)   ##major axis size in arcsec
tfbins = np.logspace(-1, 4, 300) ###flux bins in mJy
rmsbins = np.linspace(0.07, 0.35, 500) ###noise bins in mJy


################################################################################
################################################################################
###define functions


def dplot_dhist(xdat, bins=nnbins, title='', bsize=2.5, save=False, fname='test', hcolor='C0'):
    ##plots histogram of nearest neighbour distance
    plt.figure(figsize=(7, 6))
    plt.hist(xdat, bins, color=hcolor)
    plt.xlim(0.1, 1200)
    plt.xscale('log')
    
    ##plot 2.5 and 60"
    ##need to find ymax and add 5%
    ymx = 1.05*max(np.histogram(xdat, bins)[0])
    
    plt.plot([bsize, bsize], [0, ymx], ls='-.', color='r', label=str(bsize)+'"')
    plt.plot([60, 60], [0, ymx], ls=':', color='r', label='60"')
    
    plt.legend(loc=2)
    
    plt.ylim(0, ymx)
    
    plt.xticks([0.1, 1, 10, 100, 1000, 1000], [0.1, 1, 10, 100, 1000, 1000])
    
    plt.xlabel('NN sep [arcsec]', fontsize=12)
    plt.ylabel('N', fontsize=12)
    plt.title(title, fontsize=14)
    
    if save == True:
        plt.savefig(fname, dpi=150)
        plt.close()
    return



def log_hist_plot(xdat, bins, plot_beam=True, beam_size=2.5, ymax=1200,
                  colour='C0'):
    ##plots histogram of nearest neighbour distance
    plt.figure(figsize=(7, 6))
    plt.hist(xdat, bins, color=colour)
    plt.xlim(min(bins)-0.1*min(bins), max(bins)+0.1*max(bins))
    plt.ylim(0.5, ymax)
    plt.xscale('log')
    plt.yscale('log')
    
    if plot_beam==True:
        plt.plot([beam_size, beam_size], [0.5, ymax], ls='--', color='C3',
                  label=str(beam_size)+' arcsec')
    
    return


def log_N_log_S(data, bins, ymax=10**5, stype='Total', save=False, fname='test',
                hcolor='C0'):

    ##set up plot
    log_hist_plot(xdat=data, bins=bins, plot_beam=False, ymax=ymax, colour=hcolor)
    
    smax=1.1*max(bins)
    smin=0.9*min(bins)
    
    plt.xlim(smin, smax)
    
    ###set upticks
    tmin, tmax = np.log10(min(bins)), np.log10(max(bins))
    xt = 10**np.arange(tmin, tmax+1)
    xt = [float(i) for i in xt]
    ###make int if >=1
    for i in range(len(xt)):
        if xt[i]>=1:
            xt[i]=int(xt[i])

    plt.xticks(xt, xt)

    unit_lable = '[mJy]'
    if stype != 'Total':
        unit_lable = '[mJy/beam]'
    
    plt.xlabel(r'$S_{\rm{' + stype + '}}$ ' + unit_lable, fontsize=14)
    plt.ylabel('N', fontsize=14)
    
    if save == True:
        plt.savefig(fname, dpi=150)
        plt.close()
    return


def compsize_dist(data, bins=majbin, bsize=2.5, atype='Maj', hcolor='C0', ymax=10**6,
                  save=False, fname='test'):
    ###set up plot
    log_hist_plot(xdat=data, bins=bins, plot_beam=True, beam_size=bsize,
                  ymax=ymax, colour=hcolor)

    #plot labels and ticks
    xt = [1, 3, 10, 30]
    
    
    plt.xticks(xt, xt)
    
    plt.xlabel(atype + ' [arcsec]', fontsize=14)
    plt.ylabel('N', fontsize=14)
    
    plt.legend(loc=1)
    
    if save == True:
        plt.savefig(fname, dpi=150)
        plt.close()
    
    return


def rms_dist(data, bins, ymax=40000, xlab='Isl rms', hcolor='C0',
             save=False, fname='test', inunit=u.mJy, outunit=u.uJy):
    ###convert rms to correct unit
    data = np.array(data)*inunit
    data = data.to(outunit).value
    
    ###convert bins
    bins = bins*inunit
    bins = bins.to(outunit).value
    
    ###set up figure
    plt.figure(figsize=(7, 6))
    plt.hist(data, bins, color=hcolor)
    plt.xlim(min(bins), max(bins))
    plt.ylim(0, ymax)
    
    ###set up xlabel in correct units
    labu = ' [' + outunit.to_string() + '/beam]'
    
    ##make mu instead of u if uJy
    if outunit == 'uJy':
        labu = r' [$\mu$Jy/beam]'
    
    plt.xlabel(xlab + labu, fontsize=14)
    plt.ylabel('N', fontsize=14)
    
    if save == True:
        plt.savefig(fname, dpi=150)
        plt.close()
    
    return



def aitoff_scatter(ra, dec, colorby, zmin=100, zmax=200, save=False, fname='test',
                   cmap='viridis', marker_size=1, cbarlab='Isl rms',
                   inunit=u.mJy, outunit=u.uJy, per_beam=True):
    'set up ra/dec scatter plot coloured by a third variable on aitoff projection'
    ###use aitoff rather than mollweide as better suited and doesn't limit at dec+pi
    
    ###ensure dtypes = array
    ra, dec, colorby = np.array(ra), np.array(dec), np.array(colorby)*inunit
    
    ###scale rms from inunit to outunit
    colorby = colorby.to(outunit).value
    
    ###set colorbar label to be consistent
    labu = ' [' + outunit.to_string() + '/beam]'
    
    ##make mu instead of u if uJy
    if outunit == 'uJy':
        labu = r' [$\mu$Jy/beam]'
    
    if per_beam == False:
        labu = labu[:len(labu)-6]+']'
    
    cbarlab = cbarlab + labu
    
    #just subtract 180 from ra, then reverse sign to get ra in right direction
    ra = ra-180
    ra = -ra
    
    ###make astropy angle and wrap at 180deg
    ra = Angle(ra*u.deg)
    dec = Angle(dec*u.deg)
    
    ###convert to radians and remove units
    ra = ra.to('rad').value
    dec = dec.to('rad').value
    
    ###set colorby limits
    colorby[colorby<zmin] = zmin
    colorby[colorby>zmax] = zmax
    
    ###setup ticks and lables
    d2r = 1*u.deg.to('rad')
    aticks = -np.arange(120, -180, -60)*d2r
    dticks = np.arange(-75, 95, 25)*d2r
    atlabs = ['20hr', '16hr', '12hr', '8hr', '4hr']
    
    ###set up figure
    fig = plt.figure(figsize=(8,6), tight_layout=True)
    ax = fig.add_subplot(111, projection='aitoff')
    
    splot = ax.scatter(ra, dec, c=colorby, s=marker_size, cmap=cmap,
                       norm=LogNorm(), rasterized=True)
    
    ax.set_xticks(aticks)
    ax.set_xticklabels(atlabs)
    ax.set_yticks(dticks)
    ax.grid()
    
    plt.colorbar(splot, orientation='horizontal').set_label(label=cbarlab, size=14)
    
    if save == True:
        plt.savefig(fname, dpi=150)
        plt.close()
    
    return


################################################################################
################################################################################



################################################################################
#
#    * * *    S U B T I L E   I N F O .   T A B L E   P L O T S     * * *
#

def get_subtile_info_table_plot_params():
    # define plot layout paramters
    params = {
        #'Subtile',
        'Mean_isl_rms':  {
            'xlabel': '(Mean Island-RMS)/Subtile (mJy/Beam)', 
            'scale': 'log', 'range': [0.00008,0.0012],'bins': 75,
            'filename': 'mean_island_rms.png'
        },
        'SD_isl_rms':   {
            'xlabel': '(SD Island-RMS)/Subtile (mJy/Beam)', 
            'scale': 'log', 'range': [0.000001,0.005],'bins': 75,
            'filename': 'stdev_island_rms.png'
        },
        'Peak_flux_p25':  {
            'xlabel': '(Q1 Peak Flux Density)/Subtile (mJy/Beam)', 
            'scale': 'log', 'range': [0.0005,0.006],'bins': 75,
            'filename': 'q1_peak_flux_density.png'
        },
        'Peak_flux_p50':  {
            'xlabel': '(Q2 Peak Flux Density)/Subtile (mJy/Beam)', 
            'scale': 'log', 'range': [0.0005,0.010],'bins': 75,
            'filename': 'q2_peak_flux_density.png'
        },
        'Peak_flux_p75':  {
            'xlabel': '(Q3 Peak Flux Density)/Subtile (mJy/Beam)', 
            'scale': 'log', 'range': [0.0005,0.025],'bins': 75,
            'filename': 'q3_peak_flux_density.png'
        },
        'N_components':       {
            'xlabel': 'Components/Subtile', 
            'scale': 'linear', 'range': [0.0,250.0],'bins': 75,
            'filename': 'components.png'
        },
        'N_empty_islands':    {
            'xlabel': '(Empty Islands)/Subtile', 
            'scale': 'log', 'range': [0.7,1000],'bins': 25,
            'filename': 'empty_islands.png'
        },
        'Peak_flux_max': {
            'xlabel': '(Max Peak FLux)/Subtile (mJy/Beam)', 
            'scale': 'log', 'range': [0.001,7],'bins': 75,
            'filename': 'max_peak_flux.png'
        },
    }
    return params

def create_subtile_info_table_plots(
    info_table,
    plots_dir,
    iteration_callback = None,
    is_stub_test = False # This is for debugging
):
    # if debugging product nice diagnostics plots, as in test mode we have very little data
    if is_stub_test:
        try:
            info_table = pd.read_csv("media/modules/subtile_summaries_stub_test.csv.gz")
        except:
            info_table = pd.read_csv("media/modules/subtile_summaries_stub_test.csv")

    # define plot layout paramters
    params = get_subtile_info_table_plot_params() 

    # create plot files
    print(f"Creating Subtile Info. Table plots...")
    for param in params.keys():
        #plt.figure(figsize=(10,5))
        plt.figure()
        png_file = f"{plots_dir}/{params[param]['filename']}"
        if params[param]['scale'] == 'log':
            try:
                logbins = np.logspace(np.log10(params[param]['range'][0]),np.log10(params[param]['range'][1]),params[param]['bins'])
                plt.hist(info_table[param],bins=logbins)
                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel(params[param]['xlabel'])
                plt.ylabel('N')
            except UserWarning as e:
                # we should only end up here when doing testing; except, if we set is_stub_test=True during testing/debugging.
                error_string = re.sub(r"\.$","",f"{e}")
                print(f"> WARNING: {error_string}: Doing linear-scaled plot...")
                plt.figure()
                plt.hist(info_table[param],bins=105)
                plt.xlabel(params[param]['xlabel'])
                plt.ylabel('N')
        else:
            n,bins,patches=plt.hist(info_table[param],bins=105)
            plt.xlabel(params[param]['xlabel'])
            plt.ylabel('N')
        plt.tight_layout()
        print(f"> Saving: {png_file}")
        plt.savefig(png_file)
        if not iteration_callback is None:
            iteration_callback()
    print("[Done]")

    # build html table
    cols = 2
    html_table = list()
    html_table.append("<table>")
    col_fields = list()
    def make_row(c_fields):
        return "   <tr><td>"+"</td><td>".join(c_fields)+"</td></tr>"
    for idx,param in enumerate(params.keys()):
        filename = params[param]['filename']
        col_fields.append(f"<img src=\"{filename}\">")
        if not ((idx+1) % cols):
            html_table.append(make_row(col_fields))
            col_fields = list()
    if len(col_fields) > 0:
        html_table.append(make_row(col_fields))
    html_table.append("</table>")
    html_table = "\n".join(html_table)
    #print(html_table)

    return html_table

