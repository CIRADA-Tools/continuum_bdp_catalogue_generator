import re
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

class fitsImageFileHandle:
    def __init__(self,csv_file):
        # convert .vlad.csv and .csv to .fits; ignores others
        self.fits_file = re.sub(r"(\.vlad\.csv|\.csv)$",".fits",csv_file)

        self.fits_hdu    = None
        self.fits_header = None
        self.fits_data   = None
        self.fits_wcs    = None

    def __get_handle(self):
        if self.fits_hdu is None:
            self.fits_hdu = fits.open(self.fits_file)[0]
            self.fits_header = self.fits_hdu.header
            self.fits_data = np.squeeze(self.fits_hdu.data)

            # get the wcs header
            self.fits_wcs = WCS(self.fits_header)
            naxis = self.fits_wcs.naxis
            while naxis > 2:
                self.fits_wcs = self.fits_wcs.dropaxis(2)
                naxis -= 1
        return self

    def get_fits_filename(self):
        return self.fits_file

    def hdu(self):
        return self.__get_handle().fits_hdu

    def header(self):
        return self.__get_handle().fits_header

    def data(self):
        return self.__get_handle().fits_data

    def wcs(self):
        return self.__get_handle().fits_wcs
