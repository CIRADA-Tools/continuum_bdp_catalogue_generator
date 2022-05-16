import re
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils import SkyCircularAperture
from photutils import SkyCircularAnnulus
from .fits_io import fitsImageFileHandle

def get_sexadecimal_string(position,dp_ra=0,dp_dec=0,is_prefixed=True):
    ##YG altered to truncate rather than round coordinates
#    sexadecimal = f"%02d%02d%05.{dp_ra}f" % position.ra.hms+re.sub(r"([+-])\d",r"\1",f"%+d%02d%02d%04.{dp_dec}f" % position.dec.signed_dms)
    precision_pad = 2 ###prevent rounding errors prior to truncation
    
    astring = position.ra.to_string(sep='', unit='hour', precision=dp_ra+precision_pad,
                                    pad=True)
    dstring = position.dec.to_string(sep='', precision=dp_dec+precision_pad,
                                    pad=True, alwayssign=True)
                                    
    ###truncate
    astring = astring[:len(astring)-precision_pad]
    dstring = dstring[:len(dstring)-precision_pad]
    sexadecimal = astring + dstring

    return ("J" if is_prefixed else "")+sexadecimal

def pybdsf_component_name(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    # break apart the iau name format
    prefix, sexadecimal   = re.sub(r"\s+"," ",meta['Component_name']['iau_format']).split(' ')
    ra_format, dec_format = re.sub(r"^J","",re.sub(r"\s+","",sexadecimal)).split('+')
    
    # get arscec decimal precisions
    ra_arcsec_precision  = re.sub(r"^.*?(\.|$)","",ra_format.lower() ).count('s')
    dec_arcsec_precision = re.sub(r"^.*?(\.|$)","",dec_format.lower()).count('s')

    # build the component names list
    coords = SkyCoord(vlad_table['RA'],vlad_table['DEC'])
    names  = [prefix+' '+get_sexadecimal_string(coord,ra_arcsec_precision,dec_arcsec_precision) for coord in coords]

    return names

def s_code(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    s_code = np.array([],dtype=vlad_table['S_Code'].dtype)
    for cid,code in zip(vlad_table['Component_id'],vlad_table['S_Code']):
        s_code = np.append(s_code,'E' if cid < 0 else code)
    return s_code

def tile_name(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    tile_name = re.sub(r"^.*?(T\d\dt\d\d).*$",r"\1",pybdsf_output_csv_filename)
    return tile_name

def subtile_name(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    subtile_name = re.sub(r"^.*?(J.*?)\..*$",r"\1",pybdsf_output_csv_filename)
    return subtile_name

def peak_to_ring(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    def warning(text):
        print(f"WARNING::VLAD::PEAK_TO_RING: {text}")

    # defaults
    r_units_default = u.arcsec
    r_inner_default = 5
    r_outer_default = 10

    # extract peak_to_ring annulus metrics
    annulus = metrics['peak_to_ring_annulus']
    try:
        r_units = eval(f"u.{annulus['units']}")
    except:
        warning(f"'{annulus['units']}' invalid astropy unit, using {r_units_default}!")
        r_units = r_units_default
    r_inner = annulus['r_inner']
    r_outer = annulus['r_outer']
    if r_inner==0 or r_outer == 0 or r_inner > r_outer:
        warning(f"r_inner={r_inner} and r_outer={r_outer} are invalid: using defaults, r_inner={r_inner_default} and r_outer={r_outer_default}")
        r_inner = r_inner_default
        r_outer = r_outer_default

    # set up datums
    fh = fitsImageFileHandle(pybdsf_output_csv_filename)
    positions = SkyCoord(vlad_table['RA'],vlad_table['DEC'])

    # get the inner circle (core) and outter annulus (ring)
    r_core = (r_inner * r_units).to(u.deg)
    r_ring = (r_outer * r_units).to(u.deg)
    a_core = SkyCircularAperture(positions,r_core).to_pixel(fh.wcs())
    a_ring = SkyCircularAnnulus(positions,r_core,r_ring).to_pixel(fh.wcs())

    # get their peak values
    pv_core = np.array([np.max(v.multiply(fh.data())) for v in a_core.to_mask(method='center')])
    pv_ring = np.array([np.max(v.multiply(fh.data())) for v in a_ring.to_mask(method='center')])

    # compute ratio
    # TO-DO: Handle nan's...
    peak_to_ring = pv_core/pv_ring

    return peak_to_ring

def ql_image_ra(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    fh = fitsImageFileHandle(pybdsf_output_csv_filename)
    ra = fh.header()['CRVAL1']
    return ra

def ql_image_dec(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    fh = fitsImageFileHandle(pybdsf_output_csv_filename)
    dec = fh.header()['CRVAL2']
    return dec

def ql_cutout(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    sexadecimals = [n.split(' ').pop() for n in pybdsf_component_name(vlad_table,meta,metrics,pybdsf_output_csv_filename)]
    urls = ["http://{base_url}/"+f"{s}_s3arcmin_VLASS.png" for s in sexadecimals]
    return urls

def VLAD_BMAJ(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    fh = fitsImageFileHandle(pybdsf_output_csv_filename)
    beam_major_axis = fh.header()['BMAJ']
    return beam_major_axis

def VLAD_BMIN(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    fh = fitsImageFileHandle(pybdsf_output_csv_filename)
    beam_minor_axis = fh.header()['BMIN']
    return beam_minor_axis

def VLAD_BPA(vlad_table,meta,metrics,pybdsf_output_csv_filename):
    fh = fitsImageFileHandle(pybdsf_output_csv_filename)
    beam_position_angle = fh.header()['BPA']
    return beam_position_angle 

