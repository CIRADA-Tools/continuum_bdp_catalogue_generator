import os
import re
import numpy as np
from astropy.io import fits
from astropy.table import QTable
from astropy.coordinates import SkyCoord
from astropy import units as u
from .fits_io import fitsImageFileHandle

def first_distance(vlad_table,meta,surveys):
    # load first table
    ra  = surveys['FIRST']['header']['ra']['field']
    dec = surveys['FIRST']['header']['dec']['field']
    first = QTable.read(surveys['FIRST']['file'])[ra,dec]

    # get first coords
    ra_units  = eval(f"u.{surveys['FIRST']['header']['ra']['units']}")
    dec_units = eval(f"u.{surveys['FIRST']['header']['dec']['units']}")
    first_coords = SkyCoord(first[ra].value,first[dec].value,unit=(ra_units,dec_units))

    # get vlad coords
    vlad_coords  = SkyCoord(vlad_table['RA'],vlad_table['DEC'])

    # match coords
    matched = vlad_coords.match_to_catalog_sky(first_coords)
    first_dist = matched[1].to(eval(f"u.{meta['FIRST_distance']['units']}"))

    return first_dist

def nvss_distance(vlad_table,meta,surveys):
    # load first table
    ra  = surveys['NVSS']['header']['ra']['field']
    dec = surveys['NVSS']['header']['dec']['field']
    first = QTable.read(surveys['NVSS']['file'])[ra,dec]

    # get first coords
    ra_units  = eval(f"u.{surveys['NVSS']['header']['ra']['units']}")
    dec_units = eval(f"u.{surveys['NVSS']['header']['dec']['units']}")
    nvss_coords = SkyCoord(first[ra].value,first[dec].value,unit=(ra_units,dec_units))

    # get vlad coords
    vlad_coords  = SkyCoord(vlad_table['RA'],vlad_table['DEC'])

    # match coords
    matched = vlad_coords.match_to_catalog_sky(nvss_coords)
    nvss_dist = matched[1].to(eval(f"u.{meta['FIRST_distance']['units']}"))

    return nvss_dist

def get_url(vlad_filename,urls):
    fits_filename = re.sub(r"(.*?/)*","",re.sub(r"(\.vlad\.csv|\.csv)",".fits",vlad_filename))
    fits_filename = re.sub(r"\.","\\.",fits_filename) # escape the regular express dots ('.').
    fits_url = ""
    for url in urls:
        if re.search(r"%s$" % fits_filename, url):
            fits_url=url
            break
    return fits_url

def build_subtile_info_table(vlad_csv_files,urls,iteration_callback=None):
    def required_files_exist(csv_file):
        fits_file = re.sub(r"(\.vlad\.csv|\.csv)$",".fits",csv_file)
        is_csv  = os.path.isfile(csv_file)
        is_fits = os.path.isfile(fits_file)
        return is_csv and is_fits

    # subtile info template-meta data /w helper function
    def slurp(txt):
        txt = '"'+"\\n".join([re.sub(r"(\"|')",r"\\\1",t) for t in str(txt).split("\n")])+'"'
        return txt 
    def s1_filter(txt):
        txt =  re.sub(r"( +)$","",txt)
        return txt if txt else "''"
    def opt_filter(hdr,key,dummy_return=''):
        optional = [    # Example: VLASS1.1.ql.T01t01.J000228-363000.10.2048.v1.I.iter1.image.pbcor.tt0.subim.fits
            'FIELD',    # 4922
            'FILNAM01', # VLASS1
            'FILNAM02', # 1
            'FILNAM03', # ql
            'FILNAM04', # Tile
            'FILNAM05', # SubTile
            'FILNAM06', # 10
            'FILNAM07', # 2048
            'FILNAM08', # v1
            'FILNAM09', # I
            'FILNAM10', # iter1
            'FILNAM11', # image
            'FILNAM12', # tt0
            'NFILNAM',  # 12
            'INTENT',   # TARGET
            'ITER',     # 1
            'SPECMODE', # mfs
            'SPW',      # '2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17'
            'TYPE',     # image
        ]
        if key in optional and not key in hdr:
            return dummy_return
        return hdr[key]
    def get_tile(csv_file):
        return re.sub(r"^.*?(T\d\dt\d\d).*$",r"\1",csv_file)
    def get_subtile(csv_file):
        return re.sub(r"^.*?(T\d\dt\d\d)\.(J\d+[+-]\d+)\..*$",r"\2",csv_file)
    def get_epoch(csv_file):
        epoch = re.search("VLASS(\d+\.\d+)\.",csv_file)
        return epoch.group(1) if epoch else -1000
    def get_version(csv_file):
        version = re.search("VLASS.*?\.v(\d+)\.",csv_file)
        return version.group(1) if version else '-1000'

    meta = {
        ##'Subtile':         [lambda vlf,hdr,vld: hdr['FILNAM05'], 'S1'],
        #'Subtile':         [lambda vlf,hdr,vld: opt_filter(hdr,'FILNAM05',get_subtile(vlf)), 'S1'],
        ##'Image_version':   [lambda vlf,hdr,vld: re.sub('v','',hdr['FILNAM08']), 'S1'],
        #'Image_version':   [lambda vlf,hdr,vld: re.sub('v','',opt_filter(hdr,'FILNAM08','-1000')), 'S1'],
        ##'Tile':            [lambda vlf,hdr,vld: hdr['FILNAM04'], 'S1'],
        #'Tile':            [lambda vlf,hdr,vld: opt_filter(hdr,'FILNAM04',get_tile(vlf)), 'S1'],
        #'Epoch':           [lambda vlf,hdr,vld: 1+float(hdr['FILNAM02'])/10, np.dtype(float)  ],
        'Subtile':         [lambda vlf,hdr,vld: get_subtile(vlf), 'S1'],
        'Image_version':   [lambda vlf,hdr,vld: get_version(vlf), 'S1'],
        'Tile':            [lambda vlf,hdr,vld: get_tile(vlf), 'S1'],
        'Epoch':           [lambda vlf,hdr,vld: float(get_epoch(vlf)), np.dtype(float)  ],
        'NAXIS':           [lambda vlf,hdr,vld: hdr['NAXIS'],  np.dtype(int)],
        'NAXIS1':          [lambda vlf,hdr,vld: hdr['NAXIS1'], np.dtype(int)],
        'NAXIS2':          [lambda vlf,hdr,vld: hdr['NAXIS2'], np.dtype(int)],
        'NAXIS3':          [lambda vlf,hdr,vld: hdr['NAXIS3'], np.dtype(int)],
        'NAXIS4':          [lambda vlf,hdr,vld: hdr['NAXIS4'], np.dtype(int)],
        'BSCALE':          [lambda vlf,hdr,vld: hdr['BSCALE'], np.dtype(float)],
        'BZERO':           [lambda vlf,hdr,vld: hdr['BZERO'], np.dtype(float)],
        'BMAJ':            [lambda vlf,hdr,vld: hdr['BMAJ'], np.dtype(float)],
        'BMIN':            [lambda vlf,hdr,vld: hdr['BMIN'], np.dtype(float)],
        'BPA':             [lambda vlf,hdr,vld: hdr['BPA'], np.dtype(float)],
        'BTYPE':           [lambda vlf,hdr,vld: s1_filter(hdr['BTYPE']), 'S1'],
        'BUNIT':           [lambda vlf,hdr,vld: s1_filter(hdr['BUNIT']), 'S1'],
        'OBJECT':          [lambda vlf,hdr,vld: s1_filter(hdr['OBJECT']), 'S1'],
        'EQUINOX':         [lambda vlf,hdr,vld: hdr['EQUINOX'], np.dtype(float)],
        'RADESYS':         [lambda vlf,hdr,vld: hdr['RADESYS'], 'S1'],
        'LATPOLE':         [lambda vlf,hdr,vld: hdr['LATPOLE'], np.dtype(float)],
        'LONPOLE':         [lambda vlf,hdr,vld: hdr['LONPOLE'], np.dtype(float)],
        'PC1_1':           [lambda vlf,hdr,vld: hdr['PC1_1'], np.dtype(float)],
        'PC2_1':           [lambda vlf,hdr,vld: hdr['PC2_1'], np.dtype(float)],
        'PC3_1':           [lambda vlf,hdr,vld: hdr['PC3_1'], np.dtype(float)],
        'PC4_1':           [lambda vlf,hdr,vld: hdr['PC4_1'], np.dtype(float)],
        'PC1_2':           [lambda vlf,hdr,vld: hdr['PC1_2'], np.dtype(float)],
        'PC2_2':           [lambda vlf,hdr,vld: hdr['PC2_2'], np.dtype(float)],
        'PC3_2':           [lambda vlf,hdr,vld: hdr['PC3_2'], np.dtype(float)],
        'PC4_2':           [lambda vlf,hdr,vld: hdr['PC4_2'], np.dtype(float)],
        'PC1_3':           [lambda vlf,hdr,vld: hdr['PC1_3'], np.dtype(float)],
        'PC2_3':           [lambda vlf,hdr,vld: hdr['PC2_3'], np.dtype(float)],
        'PC3_3':           [lambda vlf,hdr,vld: hdr['PC3_3'], np.dtype(float)],
        'PC4_3':           [lambda vlf,hdr,vld: hdr['PC4_3'], np.dtype(float)],
        'PC1_4':           [lambda vlf,hdr,vld: hdr['PC1_4'], np.dtype(float)],
        'PC2_4':           [lambda vlf,hdr,vld: hdr['PC2_4'], np.dtype(float)],
        'PC3_4':           [lambda vlf,hdr,vld: hdr['PC3_4'], np.dtype(float)],
        'PC4_4':           [lambda vlf,hdr,vld: hdr['PC4_4'], np.dtype(float)],
        'CTYPE1':          [lambda vlf,hdr,vld: s1_filter(hdr['CTYPE1']), 'S1'],
        'CRVAL1':          [lambda vlf,hdr,vld: hdr['CRVAL1'], np.dtype(float)],
        'CDELT1':          [lambda vlf,hdr,vld: hdr['CDELT1'], np.dtype(float)],
        'CRPIX1':          [lambda vlf,hdr,vld: hdr['CRPIX1'], np.dtype(float)],
        'CUNIT1':          [lambda vlf,hdr,vld: s1_filter(hdr['CUNIT1']), 'S1'],
        'CTYPE2':          [lambda vlf,hdr,vld: s1_filter(hdr['CTYPE2']), 'S1'],
        'CRVAL2':          [lambda vlf,hdr,vld: hdr['CRVAL2'], np.dtype(float)],
        'CDELT2':          [lambda vlf,hdr,vld: hdr['CDELT2'], np.dtype(float)],
        'CRPIX2':          [lambda vlf,hdr,vld: hdr['CRPIX2'], np.dtype(float)],
        'CUNIT2':          [lambda vlf,hdr,vld: s1_filter(hdr['CUNIT2']), 'S1'],
        'CTYPE3':          [lambda vlf,hdr,vld: s1_filter(hdr['CTYPE3']), 'S1'],
        'CRVAL3':          [lambda vlf,hdr,vld: hdr['CRVAL3'], np.dtype(float)],
        'CDELT3':          [lambda vlf,hdr,vld: hdr['CDELT3'], np.dtype(float)],
        'CRPIX3':          [lambda vlf,hdr,vld: hdr['CRPIX3'], np.dtype(float)],
        'CUNIT3':          [lambda vlf,hdr,vld: s1_filter(hdr['CUNIT3']), 'S1'],
        'CTYPE4':          [lambda vlf,hdr,vld: s1_filter(hdr['CTYPE4']), 'S1'],
        'CRVAL4':          [lambda vlf,hdr,vld: hdr['CRVAL4'], np.dtype(float)],
        'CDELT4':          [lambda vlf,hdr,vld: hdr['CDELT4'], np.dtype(float)],
        'CRPIX4':          [lambda vlf,hdr,vld: hdr['CRPIX4'], np.dtype(float)],
        'CUNIT4':          [lambda vlf,hdr,vld: s1_filter(hdr['CUNIT4']), 'S1'],
        'PV2_1':           [lambda vlf,hdr,vld: hdr['PV2_1'], np.dtype(float)],
        'PV2_2':           [lambda vlf,hdr,vld: hdr['PV2_2'], np.dtype(float)],
        'RESTFRQ':         [lambda vlf,hdr,vld: hdr['RESTFRQ'], np.dtype(float)],
        'SPECSYS':         [lambda vlf,hdr,vld: s1_filter(hdr['SPECSYS']), 'S1'],
        'ALTRVAL':         [lambda vlf,hdr,vld: hdr['ALTRVAL'], np.dtype(float)],
        'ALTRPIX':         [lambda vlf,hdr,vld: hdr['ALTRPIX'], np.dtype(float)],
        'VELREF':          [lambda vlf,hdr,vld: hdr['VELREF'], np.dtype(int)],
        'COMMENT':         [lambda vlf,hdr,vld: slurp(hdr['COMMENT']), 'S1'],
        'TELESCOP':        [lambda vlf,hdr,vld: s1_filter(f'"{hdr["TELESCOP"]}"'), 'S1'],
        'OBSERVER':        [lambda vlf,hdr,vld: s1_filter(f'"{hdr["OBSERVER"]}"'), 'S1'],
        'DATEOBS':         [lambda vlf,hdr,vld: hdr['DATE-OBS'], 'S1'],
        'TIMESYS':         [lambda vlf,hdr,vld: s1_filter(hdr['TIMESYS']), 'S1'],
        'OBSRA':           [lambda vlf,hdr,vld: hdr['OBSRA'], np.dtype(float)],
        'OBSDEC':          [lambda vlf,hdr,vld: hdr['OBSDEC'], np.dtype(float)],
        'OBSGEOX':         [lambda vlf,hdr,vld: hdr['OBSGEO-X'], np.dtype(float)],
        'OBSGEOY':         [lambda vlf,hdr,vld: hdr['OBSGEO-Y'], np.dtype(float)],
        'OBSGEOZ':         [lambda vlf,hdr,vld: hdr['OBSGEO-Z'], np.dtype(float)],
        'FIELD':           [lambda vlf,hdr,vld: opt_filter(hdr,'FIELD',-1000), np.dtype(int)],
        #'FILNAM01':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM01']), 'S1'],
        #'FILNAM02':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM02']), 'S1'],
        #'FILNAM03':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM03']), 'S1'],
        #'FILNAM04':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM04']), 'S1'],
        #'FILNAM05':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM05']), 'S1'],
        #'FILNAM06':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM06']), 'S1'],
        #'FILNAM07':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM07']), 'S1'],
        #'FILNAM08':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM08']), 'S1'],
        #'FILNAM09':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM09']), 'S1'],
        #'FILNAM10':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM10']), 'S1'],
        #'FILNAM11':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM11']), 'S1'],
        #'FILNAM12':        [lambda vlf,hdr,vld: s1_filter(hdr['FILNAM12']), 'S1'],
        #'INTENT':          [lambda vlf,hdr,vld: s1_filter(hdr['INTENT']), 'S1'],
        #'ITER':            [lambda vlf,hdr,vld: hdr['ITER'], np.dtype(int)],
        #'NFILNAM':         [lambda vlf,hdr,vld: hdr['NFILNAM'], np.dtype(int)],
        #'SPECMODE':        [lambda vlf,hdr,vld: s1_filter(hdr['SPECMODE']), 'S1'],
        #'SPW':             [lambda vlf,hdr,vld: f'"{hdr["SPW"]}"', 'S1'],
        'FILNAM01':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM01')), 'S1'],
        'FILNAM02':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM02')), 'S1'],
        'FILNAM03':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM03')), 'S1'],
        'FILNAM04':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM04')), 'S1'],
        'FILNAM05':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM05')), 'S1'],
        'FILNAM06':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM06')), 'S1'],
        'FILNAM07':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM07')), 'S1'],
        'FILNAM08':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM08')), 'S1'],
        'FILNAM09':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM09')), 'S1'],
        'FILNAM10':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM10')), 'S1'],
        'FILNAM11':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM11')), 'S1'],
        'FILNAM12':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'FILNAM12')), 'S1'],
        'INTENT':          [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'INTENT')), 'S1'],
        'ITER':            [lambda vlf,hdr,vld: opt_filter(hdr,'ITER',-1000), np.dtype(int)],
        'NFILNAM':         [lambda vlf,hdr,vld: opt_filter(hdr,'NFILNAM',-1000), np.dtype(int)],
        'SPECMODE':        [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'SPECMODE')), 'S1'],
        'SPW':             [lambda vlf,hdr,vld: f'"{opt_filter(hdr,"SPW")}"', 'S1'],
        #'TYPE':            [lambda vlf,hdr,vld: s1_filter(hdr['TYPE']), 'S1'],
        'TYPE':            [lambda vlf,hdr,vld: s1_filter(opt_filter(hdr,'TYPE')), 'S1'],
        'DATE':            [lambda vlf,hdr,vld: hdr['DATE'], 'S1'],
        'ORIGIN':          [lambda vlf,hdr,vld: f'"{hdr["ORIGIN"]}"', 'S1'],
        'HISTORY':         [lambda vlf,hdr,vld: slurp(hdr['HISTORY']), 'S1'],
        'Mean_isl_rms':    [lambda vlf,hdr,vld: np.mean(vld['Isl_rms'])*1000.0, np.dtype(float)],
        'SD_isl_rms':      [lambda vlf,hdr,vld: np.std(vld['Isl_rms'])*1000.0, np.dtype(float)],
        'Peak_flux_p25':   [lambda vlf,hdr,vld: np.quantile(vld['Peak_flux'],0.25)*1000.0, np.dtype(float)],
        'Peak_flux_p50':   [lambda vlf,hdr,vld: np.quantile(vld['Peak_flux'],0.50)*1000.0, np.dtype(float)],
        'Peak_flux_p75':   [lambda vlf,hdr,vld: np.quantile(vld['Peak_flux'],0.75)*1000.0, np.dtype(float)],
        'Peak_flux_max':   [lambda vlf,hdr,vld: np.max(vld['Peak_flux'])*1000.0, np.dtype(float)],
        'N_components':    [lambda vlf,hdr,vld: len(vld[(vld['Component_id'] > -1),]), np.dtype(int)],
        'N_empty_islands': [lambda vlf,hdr,vld: len(vld[(vld['Component_id'] >  0),]), np.dtype(int)],
        'Subtile_url':     [lambda vlf,hdr,vld: get_url(vlf,urls),'S1'],
    }


    # build the table
    print("Building Subtile Info. Table...")
    info_table = QTable(names=meta.keys(),dtype=[v[1] for v in meta.values()])
    for vlad_csv_file in vlad_csv_files:
        print(f"> Processing: {vlad_csv_file}")
        if required_files_exist(vlad_csv_file):
            vladtb = QTable.read(vlad_csv_file)
            header = fitsImageFileHandle(vlad_csv_file).header()
            record = [meta[k][0](vlad_csv_file,header,vladtb) for k in meta.keys()]
            info_table.add_row(record)
        else:
            print("> WARNING: Either .csv or .fits file is missing, skipping...")
        if not iteration_callback is None:
            iteration_callback()
    if len(info_table)>0:
        print(f"> Output table format:")
        print(info_table.info)
    print("[Done]")

    return info_table

