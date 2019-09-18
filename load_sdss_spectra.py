import numpy
import download_sdss_spectra as dss
from astropy.io import fits
from multiprocessing import Pool
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

BASE_DATA_PATH = '../data/'

def load_spectra(df, common_wave):
    df = dss.get_run2d(df)
    nof_objects = df.shape[0]
    res = Parallel(n_jobs=-1, verbose = 10)(delayed(single_object_load)(df.loc[i], common_wave)
                            for i in range(nof_objects))

    return numpy.vstack(res)



#def single_object_load(object_id, spectra_files_path, i):
def single_object_load(object_id, common_wave):
    """
    Load data from the fits file for a single objects
    :param object_id:
    :return:
    """

    plate  = str(object_id['plate'])
    mjd  = str(object_id['mjd'])
    fiber  = str(object_id['fiberid'])
    run2d  = object_id['run2d']
    #z = object_id['z']
    #ebv = object_id['ebv']

    spectrum_folder = dss.spectrum_path(plate, run2d)
    spectrum_file_name = dss.spectrum_file_name(plate,mjd,fiber)
    full_path = BASE_DATA_PATH + spectrum_folder + spectrum_file_name
    #full_path =  spectrum_folder + spectrum_file_name

    try:
        #print(full_path)
        hdulist = fits.open(full_path)

        spec = hdulist[1].data['flux']
        wave = wavelength_ex(hdulist)
        #spec = deredden_spectrum(wave, spec, ebv)

        #wave = wave/(1+z)
        ivar = hdulist[1].data['ivar']
        sky = hdulist[1].data['sky']

        if False:
            spec[ivar <  (1 / (numpy.nanmedian(spec)**2))]    = numpy.nan
            spec[sky > 2*numpy.nanmedian(spec)] = numpy.nan
            spec_i = numpy.interp(common_wave, wave, spec)
            #spec = spec_i.astype('f2')


        if True:
            spec_i = numpy.interp(common_wave, wave, spec)
            ivar_i = numpy.interp(common_wave, wave, ivar,left = 0, right=0)
            sky_i  = numpy.interp(common_wave, wave, sky)


            #spec_i[ivar_i <  (1 / (numpy.nanmedian(spec_i)**2))]    = numpy.nan
            spec_i[ivar_i < (1 / (numpy.nanmedian(abs(spec_i))**2))] = numpy.nan
            #spec_i[sky_i > 2*numpy.nanmedian(spec_i)] = numpy.nan
            #spec_i = spec_i

        hdulist.close()
    except (OSError, TypeError):
        print(full_path)
        spec_i = numpy.zeros(common_wave.shape)*numpy.nan

    return spec_i



def wavelength_ex(hdulist):
    """
    A function to get the wavelength grid in linear units
    :param hdulist:
    :return:
    """
    wave = hdulist[1].data['loglam']
    wave = 10**wave

    return wave

def deredden_spectrum(wl, spec, E_bv):
    """
    function dereddens a spectrum based on the given extinction_g value and Fitzpatric99 model
    IMPORTANT: the spectrum should be in the observer frame (do not correct for redshift)
    """
    # dust model
    wls = numpy.array([ 2600,  2700,  4110,  4670,  5470,  6000, 12200, 26500])
    a_l = numpy.array([ 6.591,  6.265,  4.315,  3.806,  3.055,  2.688,  0.829,  0.265])
    f_interp = interp1d(wls, a_l, kind="cubic")

    a_l_all = f_interp(wl)
    #E_bv = extinction_g / 3.793
    A_lambda = E_bv * a_l_all
    spec_real = spec * 10 ** (A_lambda / 2.5)

    return spec_real
