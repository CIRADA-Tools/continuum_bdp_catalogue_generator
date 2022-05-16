import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import coordinates as coords
from astropy.wcs import WCS
import pandas as pd

##x/y positions (requires subtile info)
###need to create WCS info for each image - WCS(dict of series)
def yg_recover_xy(cdata, imdat, acol='RA', dcol='DEC', apcol='RA_max', dpcol='DEC_max',
               eacol='E_RA', edcol='E_DEC', eapcol='E_RA_max', edpcol='E_DEC_max'):
    ##make colnames generic within function
    ###recover X/Yposn from image header info
    #fstart_time = time.time()
    
    #cunit, ctype, cdelt, crval, crpix, naxis
    ###values that are constant for ALL QL images
    nax = 3722
    cta = 'RA---SIN'
    ctd = 'DEC--SIN'
    cu = 'deg'
    crp = 1861.0
    cda = -0.0002777777777778
    cdd = 0.0002777777777778
    
    ###create a list of WCS transforms
    ###create a base 2D WCS for all images - need to move inside of loop
    wcslist = []
    for i in range(len(imdat)):
        aref, dref = imdat.CRVAL1.iloc[i], imdat.CRVAL2.iloc[i]
        ql_wcs = WCS({'NAXIS':2, 'NAXIS1':nax, 'NAXIS2':nax})
        ql_wcs.wcs.ctype = [cta, ctd]
        ql_wcs.wcs.cunit = [cu, cu]
        ql_wcs.wcs.crpix = [crp, crp]
        ql_wcs.wcs.crval = [aref, dref]
        ql_wcs.wcs.cdelt = [cda, cdd]
        wcslist.append(ql_wcs)
    
    ###obtain index of subtile in imdat to determine wcs to use for row in catalogue
    stlist = list(imdat['Subtile'])

    ##need to loop through cdata and append x/y coords to list
    cpos_cat = SkyCoord(ra=np.array(cdata[acol]), dec=np.array(cdata[dcol]), unit='deg')
    mpos_cat = SkyCoord(ra=np.array(cdata[apcol]), dec=np.array(cdata[dpcol]), unit='deg')
    xpos, ypos, xpmax, ypmax = [], [], [], []
    for i in range(len(cdata)):
        skypos = cpos_cat[i]
        maxpos = mpos_cat[i]
        stile = cdata.iloc[i]['Subtile']
        sti = stlist.index(stile)
        pxcoords = wcslist[sti].world_to_pixel(skypos)
        pxcoords_max = wcslist[sti].world_to_pixel(maxpos)
        
        xpos.append(float(pxcoords[0]))
        ypos.append(float(pxcoords[1]))

        xpmax.append(float(pxcoords_max[0]))
        ypmax.append(float(pxcoords_max[1]))

    ##add x/y columns to cdata
    ###errors in px coords estimate via error in pos/|cdelt|
    ###e.g. E_Xposn = E_RA/|CDELT1|
    cdata = cdata.assign(Xposn = xpos)
    cdata = cdata.assign(E_Xposn = np.array(cdata[eacol])/cdd)
    cdata = cdata.assign(Yposn = ypos)
    cdata = cdata.assign(E_Yposn = np.array(cdata[edcol])/cdd)

    cdata = cdata.assign(Xposn_max = xpmax)
    cdata = cdata.assign(E_Xposn_max = np.array(cdata[eapcol])/cdd)
    cdata = cdata.assign(Yposn_max = ypmax)
    cdata = cdata.assign(E_Yposn_max = np.array(cdata[edpcol])/cdd)
    
    return(cdata)

def xy_positions(vlad_table,meta,metrics,info_table):
    df = vlad_table.to_pandas().copy()
    datum = yg_recover_xy(df,info_table.to_pandas())
    fields = ['Xposn','E_Xposn','Yposn','E_Yposn','Xposn_max','E_Xposn_max','Yposn_max','E_Yposn_max']
    xy_pos = {f:list(datum[f].copy()*u.pixel) for f in fields}
    return xy_pos

def yg_find_duplicates(df, acol='RA', dcol='DEC', pos_err=2*u.arcsec):
    ###find duplicates and flag

    ###create SN column to sort by - may replace with q_flag later
    df['SN'] = df['Peak_flux']/df['Isl_rms']
    
    #2) sort by SN/qflag, subset dist<2"
    #df = df.sort_values(by='SN', ascending=False).reset_index(drop=True)
    df = df.sort_values(by='SN', ascending=False).reset_index(drop=False)

    #####DONT subset duplicates!
    ###tun search around on entire catalogue (sorted) and use index you dumbass!
    dfpos = SkyCoord(ra=np.array(df[acol]), dec=np.array(df[dcol]), unit='deg')
    dsearch = dfpos.search_around_sky(dfpos, seplimit=pos_err)
    
    ###create dataframe for easy manipulation - not actually neccesary just cleaner
    dsdf = pd.DataFrame({'ix1': dsearch[0], 'ix2': dsearch[1], 'ix3': dsearch[2].arcsec})
    
    ###subset to ix1 != ix2 - reduces 4M to 500k
    dsdf = dsdf[(dsdf['ix1']!=dsdf['ix2'])].reset_index(drop=True)
    
    ###is index of preferred components where fist instance in ix1 occurs before ix2? - I think so
    ix1, ix2 = list(dsdf['ix1']), list(dsdf['ix2'])
    prefcomp = [i for i in ix1 if ix1.index(i) < ix2.index(i)] ##this takes a while
    
    ###use pref comp to filter dup array and reflag
    dupflag = np.zeros(len(df)) ##all set to zero
    dupflag[np.unique(ix1)] = 2 ##flags all duplicates
    dupflag[prefcomp] = 1 ##reflags preferred duplicates
    
    df['Duplicate_flag'] = dupflag
    
    ###re-sort to old index and drop column 'index'
    df = df.sort_values(by='index').drop('index', axis=1).reset_index(drop=True)
    
    return(df)

def find_duplicates(vlad_table,meta,metrics,info_table):
    df = vlad_table.to_pandas().copy()
    position_error = metrics['duplicate_flagging']['duplicate_search']['radius']
    position_error *= eval(f"u.{metrics['duplicate_flagging']['duplicate_search']['units']}")
    duplicate_flags = list(yg_find_duplicates(df,pos_err=position_error)['Duplicate_flag'].copy())
    return duplicate_flags

def yg_q_flag(df, snmin=5, prmax=2, prdist=20):
    ###Q_flag
    ##3 fold:
    ## 1) Tot < Peak (GFIT)
    ## 2) Peak < 5*Isl_rms (SN)
    ## 3) dNN >= 20" && Peak_to_ring < 2 (PR)
    
    ###combine in to single flag value via binary bit addition
    ## PR = 1; SN = 2; GFIT = 4
    ## weights GFIT highest, then SN, then P2R
    gfit, sn, pr = np.zeros(len(df)), np.zeros(len(df)), np.zeros(len(df))
    
    ###necessary column arrays
    stot = np.array(df['Total_flux'])
    speak = np.array(df['Peak_flux'])
    rms = np.array(df['Isl_rms'])
    dnn = np.array(df['NN_dist'])
    ptr = np.array(df['Peak_to_ring'])
    
    ###flag individual critera
    gfit[(speak > stot)] = 4
    sn[(speak < snmin*rms)] = 2
    pr[(dnn>=prdist) & (ptr<prmax)] = 1
    
    qflag = gfit + sn + pr
    
    df = df.assign(Quality_flag = qflag)

    return(df)

def quality_flag(vlad_table,meta,metrics,info_table):
    df = vlad_table.to_pandas().copy()
    q_flag = list(yg_q_flag(df)['Quality_flag'].copy())
    return q_flag

def source_name(vlad_table,meta,metrics,info_table):
    return 'N/A'

def source_type(vlad_table,meta,metrics,info_table):
    return 'N/A'

####nn dist needs to be limited to unique components that aren't empty islands
def yg_add_nn_dist(df, acol='RA', dcol='DEC'):
    
    ###create column for row number - allows easy remerge
    df['dfix'] = df.index
    
    ###subset those to be used in the NN search (D_flag<2 & S_Code!='E')
    ucomps = df[(df['Duplicate_flag']<2) & (df['S_Code']!='E')].reset_index(drop=True)
    
    ##create sky position catalogue and self match to nearest OTHER component
    poscat = SkyCoord(ra=np.array(ucomps[acol]), dec=np.array(ucomps[dcol]), unit='deg')
    self_x = coords.match_coordinates_sky(poscat, poscat, nthneighbor=2)
    
    ###create new column in ucomps
    ucomps = ucomps.assign(NN_dist = self_x[1].arcsec)
    
    ###merge with df, fill na with -99
    df = pd.merge(df, ucomps[['dfix', 'NN_dist']], on='dfix', how='left')
    df['NN_dist'] = df['NN_dist'].fillna(-99)
    
    return(df)

def nn_distance(vlad_table,meta,metrics,info_table):
    df = vlad_table.to_pandas().copy()
    nn_dist = list(yg_add_nn_dist(df)['NN_dist'].copy())
    return  nn_dist


